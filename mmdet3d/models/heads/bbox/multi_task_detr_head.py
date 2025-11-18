"""
Multi-Task DETR Head for End-to-End Perception
统一的多任务DETR头：目标检测、语义地图、轨迹跟踪

架构设计:
    BEV Features → Shared Transformer → Task-Specific Heads
                                         ├─ Detection Head
                                         ├─ Segmentation Head
                                         └─ Tracking Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss

__all__ = ["MultiTaskDETRHead"]


@HEADS.register_module()
class MultiTaskDETRHead(nn.Module):
    """
    统一的多任务DETR头
    
    支持的任务:
    - 3D目标检测 (Object Detection)
    - BEV语义分割 (Semantic Segmentation)
    - 多目标跟踪 (Multi-Object Tracking)
    
    Args:
        num_classes (int): 检测类别数
        num_seg_classes (int): 分割类别数
        in_channels (int): 输入通道数
        num_query_det (int): 检测query数量
        num_query_seg (int): 分割query数量
        transformer (dict): Transformer配置
        with_tracking (bool): 是否启用跟踪
        track_memory_len (int): 跟踪记忆长度
    """
    
    def __init__(
        self,
        num_classes=10,
        num_seg_classes=4,  # 可行驶区域、车道线、人行道等
        in_channels=256,
        num_query_det=900,
        num_query_seg=100,
        num_reg_fcs=2,
        transformer=None,
        with_tracking=True,
        track_memory_len=5,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        loss_track=dict(
            type='TrackingLoss',
            loss_weight=1.0
        ),
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.in_channels = in_channels
        self.num_query_det = num_query_det
        self.num_query_seg = num_query_seg
        self.with_tracking = with_tracking
        self.track_memory_len = track_memory_len
        self.fp16_enabled = False
        
        # ========== Shared Transformer ==========
        self.transformer = self._build_transformer(transformer or {})
        
        # ========== Query Embeddings ==========
        # 检测queries
        self.query_embed_det = nn.Embedding(num_query_det, in_channels)
        # 分割queries（用于语义地图）
        self.query_embed_seg = nn.Embedding(num_query_seg, in_channels)
        
        # ========== Task-Specific Heads ==========
        
        # 1. 检测头
        self.detection_head = self._build_detection_head(num_reg_fcs)
        
        # 2. 分割头
        self.segmentation_head = self._build_segmentation_head()
        
        # 3. 跟踪头
        if with_tracking:
            self.tracking_head = self._build_tracking_head()
            # 跟踪记忆库
            self.track_memory = None
        
        # ========== Loss Functions ==========
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_seg = build_loss(loss_seg)
        if with_tracking:
            self.loss_track = build_loss(loss_track) if isinstance(loss_track, dict) else None
    
    def _build_transformer(self, cfg):
        """构建共享的Transformer Decoder"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.in_channels,
            nhead=cfg.get('num_heads', 8),
            dim_feedforward=cfg.get('ffn_dim', 2048),
            dropout=cfg.get('dropout', 0.1),
            activation='relu',
            batch_first=True
        )
        
        decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.get('num_layers', 6)
        )
        
        return decoder
    
    def _build_detection_head(self, num_reg_fcs):
        """构建检测头"""
        return nn.ModuleDict({
            'cls': nn.Linear(self.in_channels, self.num_classes + 1),
            'reg': self._build_reg_branch(num_reg_fcs),
            'velocity': nn.Linear(self.in_channels, 2),  # vx, vy
        })
    
    def _build_reg_branch(self, num_fcs):
        """构建回归分支"""
        layers = []
        for _ in range(num_fcs):
            layers.extend([
                nn.Linear(self.in_channels, self.in_channels),
                nn.ReLU(inplace=True)
            ])
        # 输出: [x, y, z, w, h, l, sin(yaw), cos(yaw)]
        layers.append(nn.Linear(self.in_channels, 8))
        return nn.Sequential(*layers)
    
    def _build_segmentation_head(self):
        """构建语义分割头"""
        return nn.ModuleDict({
            # 将query特征投影到BEV空间
            'query_proj': nn.Sequential(
                nn.Linear(self.in_channels, self.in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_channels, self.in_channels),
            ),
            # 分割mask生成
            'mask_embed': nn.Sequential(
                nn.Linear(self.in_channels, self.in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_channels, self.in_channels),
            ),
            # 类别预测
            'cls_embed': nn.Linear(self.in_channels, self.num_seg_classes),
        })
    
    def _build_tracking_head(self):
        """构建跟踪头"""
        return nn.ModuleDict({
            # 目标embedding
            'obj_embed': nn.Sequential(
                nn.Linear(self.in_channels, self.in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_channels, 256),  # tracking embedding
            ),
            # 轨迹预测（预测未来位置）
            'traj_pred': nn.Sequential(
                nn.Linear(self.in_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 6 * 2),  # 预测未来6帧的(x,y)
            ),
            # ID分类（用于关联）
            'id_embed': nn.Linear(256, 256),
        })
    
    @auto_fp16(apply_to=('bev_features',))
    def forward(self, bev_features, img_metas=None, prev_frame_info=None):
        """
        前向传播
        
        Args:
            bev_features (dict): BEV特征字典
                - 'bev_flatten': [B, N, C] 展平的BEV特征
                - 'bev_features': [B, C, H, W] 2D BEV特征
            img_metas (list): 图像元信息
            prev_frame_info (dict): 前一帧信息（用于跟踪）
            
        Returns:
            dict: 多任务预测结果
                - 'detection': 检测结果
                - 'segmentation': 分割结果
                - 'tracking': 跟踪结果（可选）
        """
        # 处理输入
        if isinstance(bev_features, dict):
            bev_flatten = bev_features['bev_flatten']  # [B, N, C]
            bev_2d = bev_features['bev_features']  # [B, C, H, W]
        else:
            B, C, H, W = bev_features.shape
            bev_flatten = bev_features.flatten(2).permute(0, 2, 1)
            bev_2d = bev_features
        
        B, N, C = bev_flatten.shape
        
        # ========== 1. 检测分支 ==========
        detection_results = self._forward_detection(bev_flatten, img_metas)
        
        # ========== 2. 分割分支 ==========
        segmentation_results = self._forward_segmentation(bev_flatten, bev_2d, img_metas)
        
        # ========== 3. 跟踪分支（可选）==========
        tracking_results = None
        if self.with_tracking:
            tracking_results = self._forward_tracking(
                detection_results['query_features'],
                prev_frame_info
            )
        
        return {
            'detection': detection_results,
            'segmentation': segmentation_results,
            'tracking': tracking_results,
        }
    
    def _forward_detection(self, bev_flatten, img_metas):
        """检测分支前向传播"""
        B = bev_flatten.shape[0]
        
        # Query embedding
        query_embeds = self.query_embed_det.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Transformer
        hs = self.transformer(query_embeds, bev_flatten)
        
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)
        
        # 使用最后一层
        query_feat = hs[-1]  # [B, num_query_det, C]
        
        # 预测
        cls_scores = self.detection_head['cls'](query_feat)
        bbox_preds = self.detection_head['reg'](query_feat)
        velocity_preds = self.detection_head['velocity'](query_feat)
        
        return {
            'cls_scores': cls_scores,      # [B, num_query_det, num_classes+1]
            'bbox_preds': bbox_preds,      # [B, num_query_det, 8]
            'velocity_preds': velocity_preds,  # [B, num_query_det, 2]
            'query_features': query_feat,  # [B, num_query_det, C]
        }
    
    def _forward_segmentation(self, bev_flatten, bev_2d, img_metas):
        """分割分支前向传播"""
        B, C, H, W = bev_2d.shape
        
        # Segmentation query
        query_embeds = self.query_embed_seg.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Transformer
        seg_hs = self.transformer(query_embeds, bev_flatten)
        
        if seg_hs.dim() == 3:
            seg_hs = seg_hs.unsqueeze(0)
        
        seg_feat = seg_hs[-1]  # [B, num_query_seg, C]
        
        # Query投影
        seg_queries = self.segmentation_head['query_proj'](seg_feat)
        
        # Mask embedding
        mask_embed = self.segmentation_head['mask_embed'](seg_queries)  # [B, Q, C]
        
        # 与BEV特征交互生成mask
        bev_flat = bev_2d.flatten(2)  # [B, C, H*W]
        masks = torch.einsum('bqc,bchw->bqhw', mask_embed, bev_2d)  # [B, Q, H, W]
        
        # 类别预测
        seg_cls = self.segmentation_head['cls_embed'](seg_queries)  # [B, Q, num_seg_classes]
        
        return {
            'seg_masks': masks,        # [B, num_query_seg, H, W]
            'seg_classes': seg_cls,    # [B, num_query_seg, num_seg_classes]
            'seg_features': seg_feat,  # [B, num_query_seg, C]
        }
    
    def _forward_tracking(self, det_query_feat, prev_frame_info):
        """跟踪分支前向传播"""
        B, Q, C = det_query_feat.shape
        
        # 目标embedding（用于关联）
        obj_embeds = self.tracking_head['obj_embed'](det_query_feat)  # [B, Q, 256]
        id_embeds = self.tracking_head['id_embed'](obj_embeds)  # [B, Q, 256]
        
        # 轨迹预测
        traj_preds = self.tracking_head['traj_pred'](det_query_feat)  # [B, Q, 12]
        traj_preds = traj_preds.view(B, Q, 6, 2)  # [B, Q, 6, 2] - 未来6帧(x,y)
        
        # 与前一帧关联
        match_scores = None
        if prev_frame_info is not None and 'obj_embeds' in prev_frame_info:
            prev_embeds = prev_frame_info['obj_embeds']  # [B, Q_prev, 256]
            # 计算余弦相似度
            curr_norm = F.normalize(obj_embeds, dim=-1)
            prev_norm = F.normalize(prev_embeds, dim=-1)
            match_scores = torch.bmm(curr_norm, prev_norm.transpose(1, 2))  # [B, Q, Q_prev]
        
        return {
            'obj_embeds': obj_embeds,      # [B, Q, 256]
            'id_embeds': id_embeds,        # [B, Q, 256]
            'traj_preds': traj_preds,      # [B, Q, 6, 2]
            'match_scores': match_scores,  # [B, Q, Q_prev] or None
        }
    
    @force_fp32(apply_to=('preds_dict',))
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_seg_masks, gt_track_ids, preds_dict, **kwargs):
        """
        计算多任务损失
        
        Args:
            gt_bboxes_3d (list): GT 3D boxes
            gt_labels_3d (list): GT labels
            gt_seg_masks (Tensor): GT segmentation masks [B, num_classes, H, W]
            gt_track_ids (list): GT tracking IDs
            preds_dict (dict): 预测结果字典
            
        Returns:
            dict: 损失字典
        """
        losses = {}
        
        # 1. 检测损失
        det_losses = self._loss_detection(
            preds_dict['detection'],
            gt_bboxes_3d,
            gt_labels_3d
        )
        losses.update({f'det_{k}': v for k, v in det_losses.items()})
        
        # 2. 分割损失
        seg_losses = self._loss_segmentation(
            preds_dict['segmentation'],
            gt_seg_masks
        )
        losses.update({f'seg_{k}': v for k, v in seg_losses.items()})
        
        # 3. 跟踪损失
        if self.with_tracking and preds_dict['tracking'] is not None:
            track_losses = self._loss_tracking(
                preds_dict['tracking'],
                gt_track_ids,
                gt_bboxes_3d
            )
            losses.update({f'track_{k}': v for k, v in track_losses.items()})
        
        return losses
    
    def _loss_detection(self, det_pred, gt_bboxes_list, gt_labels_list):
        """检测损失"""
        cls_scores = det_pred['cls_scores']
        bbox_preds = det_pred['bbox_preds']
        
        # 简化版损失计算（实际需要匈牙利匹配）
        B = cls_scores.shape[0]
        
        # 这里需要实现完整的匈牙利匹配和损失计算
        # 为简化，返回占位损失
        loss_cls = cls_scores.sum() * 0
        loss_bbox = bbox_preds.sum() * 0
        
        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
        }
    
    def _loss_segmentation(self, seg_pred, gt_masks):
        """分割损失"""
        seg_masks = seg_pred['seg_masks']  # [B, Q, H, W]
        seg_cls = seg_pred['seg_classes']  # [B, Q, num_classes]
        
        # Mask损失（二值交叉熵）
        if gt_masks is not None:
            # 需要将query masks与GT masks匹配
            loss_mask = F.binary_cross_entropy_with_logits(
                seg_masks.flatten(0, 1),
                gt_masks.unsqueeze(1).repeat(1, seg_masks.shape[1], 1, 1, 1).flatten(0, 1)
            ) * 0  # 占位
        else:
            loss_mask = seg_masks.sum() * 0
        
        # 类别损失
        loss_cls = seg_cls.sum() * 0  # 占位
        
        return {
            'loss_mask': loss_mask,
            'loss_cls': loss_cls,
        }
    
    def _loss_tracking(self, track_pred, gt_track_ids, gt_bboxes_list):
        """跟踪损失"""
        obj_embeds = track_pred['obj_embeds']
        traj_preds = track_pred['traj_preds']
        
        # 1. ID一致性损失（对比学习）
        loss_id = obj_embeds.sum() * 0  # 占位
        
        # 2. 轨迹预测损失
        loss_traj = traj_preds.sum() * 0  # 占位
        
        return {
            'loss_id': loss_id,
            'loss_traj': loss_traj,
        }
    
    @force_fp32(apply_to=('preds_dict',))
    def get_results(self, preds_dict, img_metas):
        """
        生成最终结果
        
        Returns:
            list: 每个样本的结果
                - boxes_3d: 3D检测框
                - scores_3d: 置信度
                - labels_3d: 类别
                - seg_mask: 语义地图
                - track_ids: 跟踪ID
                - trajectories: 预测轨迹
        """
        B = preds_dict['detection']['cls_scores'].shape[0]
        results = []
        
        for batch_idx in range(B):
            result = {}
            
            # 1. 检测结果
            det_result = self._get_detection_results(
                preds_dict['detection'],
                batch_idx
            )
            result.update(det_result)
            
            # 2. 分割结果
            seg_result = self._get_segmentation_results(
                preds_dict['segmentation'],
                batch_idx
            )
            result.update(seg_result)
            
            # 3. 跟踪结果
            if self.with_tracking and preds_dict['tracking'] is not None:
                track_result = self._get_tracking_results(
                    preds_dict['tracking'],
                    batch_idx,
                    det_result
                )
                result.update(track_result)
            
            results.append(result)
        
        return results
    
    def _get_detection_results(self, det_pred, batch_idx):
        """获取检测结果"""
        cls_scores = det_pred['cls_scores'][batch_idx]
        bbox_preds = det_pred['bbox_preds'][batch_idx]
        velocity_preds = det_pred['velocity_preds'][batch_idx]
        
        # Softmax获取分数
        scores, labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        
        # 过滤
        keep = scores > 0.1
        scores = scores[keep]
        labels = labels[keep]
        bbox_preds = bbox_preds[keep]
        velocity_preds = velocity_preds[keep]
        
        # 转换bbox格式
        bboxes = self._bbox_pred_to_result(bbox_preds, velocity_preds)
        
        return {
            'boxes_3d': bboxes,
            'scores_3d': scores,
            'labels_3d': labels,
        }
    
    def _get_segmentation_results(self, seg_pred, batch_idx):
        """获取分割结果"""
        seg_masks = seg_pred['seg_masks'][batch_idx]  # [Q, H, W]
        seg_cls = seg_pred['seg_classes'][batch_idx]  # [Q, num_classes]
        
        # 获取每个query的类别
        seg_labels = seg_cls.argmax(dim=-1)  # [Q]
        
        # 聚合mask（同类别的mask合并）
        H, W = seg_masks.shape[-2:]
        final_mask = torch.zeros(self.num_seg_classes, H, W, device=seg_masks.device)
        
        for cls_id in range(self.num_seg_classes):
            cls_queries = (seg_labels == cls_id)
            if cls_queries.any():
                final_mask[cls_id] = seg_masks[cls_queries].sigmoid().max(dim=0)[0]
        
        return {
            'seg_mask': final_mask,  # [num_seg_classes, H, W]
        }
    
    def _get_tracking_results(self, track_pred, batch_idx, det_result):
        """获取跟踪结果"""
        obj_embeds = track_pred['obj_embeds'][batch_idx]
        traj_preds = track_pred['traj_preds'][batch_idx]
        match_scores = track_pred['match_scores']
        
        # 分配track ID（基于matching）
        if match_scores is not None:
            track_ids = match_scores[batch_idx].argmax(dim=-1)
        else:
            track_ids = torch.arange(len(obj_embeds), device=obj_embeds.device)
        
        return {
            'track_ids': track_ids,
            'trajectories': traj_preds,  # [N, 6, 2]
            'obj_embeds': obj_embeds,    # 保存用于下一帧
        }
    
    def _bbox_pred_to_result(self, bbox_preds, velocity_preds):
        """将预测转换为bbox格式"""
        # bbox_preds: [N, 8] -> [x, y, z, w, h, l, sin, cos]
        # velocity_preds: [N, 2] -> [vx, vy]
        
        x, y, z, w, h, l, sin_yaw, cos_yaw = bbox_preds.unbind(-1)
        vx, vy = velocity_preds.unbind(-1)
        
        yaw = torch.atan2(sin_yaw, cos_yaw)
        
        # [x, y, z, w, h, l, yaw, vx, vy]
        bboxes = torch.stack([x, y, z, w, h, l, yaw, vx, vy], dim=-1)
        
        return bboxes
