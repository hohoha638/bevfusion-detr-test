"""
DETR-based 3D Object Detection Head for BEVFusion
基于DETR的3D目标检测头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.builder import build_loss as build_loss_3d

__all__ = ["DETRHead3D"]


@HEADS.register_module()
class DETRHead3D(nn.Module):
    """
    基于DETR的3D目标检测头
    
    Args:
        num_classes (int): 类别数量
        in_channels (int): 输入通道数
        num_query (int): query数量
        transformer (dict): Transformer配置
        with_box_refine (bool): 是否使用box refinement
        num_reg_fcs (int): 回归分支的FC层数
    """
    
    def __init__(
        self,
        num_classes=10,
        in_channels=256,
        num_query=300,
        num_reg_fcs=2,
        transformer=None,
        bbox_coder=None,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        with_box_refine=False,
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.with_box_refine = with_box_refine
        self.fp16_enabled = False
        
        # Transformer
        if transformer is not None:
            self.transformer = self._build_transformer(transformer)
        else:
            # 默认Transformer配置
            self.transformer = self._build_default_transformer()
        
        # Query Embedding
        self.query_embedding = nn.Embedding(num_query, in_channels)
        
        # 分类头
        self.cls_branches = nn.ModuleList([
            nn.Linear(in_channels, num_classes + 1)  # +1 for background
        ])
        
        # 回归头 - 输出9个参数 [x, y, z, w, h, l, sin(yaw), cos(yaw), vx, vy]
        self.reg_branches = nn.ModuleList([
            self._build_reg_branch(in_channels, num_reg_fcs)
        ])
        
        # 如果使用box refine，为每个decoder layer创建分支
        if with_box_refine:
            num_pred = transformer.get('decoder', {}).get('num_layers', 6)
            self.cls_branches = nn.ModuleList([
                nn.Linear(in_channels, num_classes + 1) 
                for _ in range(num_pred)
            ])
            self.reg_branches = nn.ModuleList([
                self._build_reg_branch(in_channels, num_reg_fcs)
                for _ in range(num_pred)
            ])
        
        # 损失函数
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        
        # Bbox coder
        if bbox_coder is not None:
            from mmdet.core import build_bbox_coder
            self.bbox_coder = build_bbox_coder(bbox_coder)
        else:
            self.bbox_coder = None
    
    def _build_transformer(self, transformer_cfg):
        """构建Transformer"""
        from mmdet3d.models.utils.transformer import TransformerDecoder
        
        # 简化的Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.in_channels,
            nhead=transformer_cfg.get('num_heads', 8),
            dim_feedforward=transformer_cfg.get('ffn_dim', 2048),
            dropout=transformer_cfg.get('dropout', 0.1),
            activation='relu',
            batch_first=True
        )
        
        decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_cfg.get('num_layers', 6)
        )
        
        return decoder
    
    def _build_default_transformer(self):
        """构建默认Transformer"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.in_channels,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        return decoder
    
    def _build_reg_branch(self, in_channels, num_fcs):
        """构建回归分支"""
        layers = []
        for i in range(num_fcs):
            layers.extend([
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True)
            ])
        # 输出: [x, y, z, w, h, l, sin(yaw), cos(yaw), vx, vy] = 10维
        layers.append(nn.Linear(in_channels, 10))
        
        return nn.Sequential(*layers)
    
    @auto_fp16(apply_to=('bev_features',))
    def forward(self, bev_features, img_metas=None):
        """
        前向传播
        
        Args:
            bev_features (dict or Tensor): BEV特征
                如果是dict，应包含:
                    - 'bev_flatten': [B, N, C] 展平的BEV特征
                    - 'position_encoding': [B, C, H, W] 位置编码(可选)
                如果是Tensor: [B, C, H, W]
            img_metas (list): 图像元信息
            
        Returns:
            dict: 预测结果
                - 'all_cls_scores': list of [B, num_query, num_classes+1]
                - 'all_bbox_preds': list of [B, num_query, 10]
        """
        # 处理输入特征
        if isinstance(bev_features, dict):
            bev_flatten = bev_features['bev_flatten']  # [B, N, C]
            pos_encoding = bev_features.get('position_encoding', None)
        else:
            # 如果是Tensor，需要展平
            B, C, H, W = bev_features.shape
            bev_flatten = bev_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            pos_encoding = None
        
        B, N, C = bev_flatten.shape
        
        # Query embedding
        query_embeds = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_query, C]
        
        # 位置编码处理
        if pos_encoding is not None:
            pos_flatten = pos_encoding.flatten(2).permute(0, 2, 1)  # [B, N, C]
            memory = bev_flatten + pos_flatten
        else:
            memory = bev_flatten
        
        # Transformer Decoder
        hs = self.transformer(query_embeds, memory)  # [num_layers, B, num_query, C]
        
        # 如果不是多层输出，添加维度
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)
        
        # 预测分类和回归
        all_cls_scores = []
        all_bbox_preds = []
        
        num_layers = hs.shape[0]
        for lvl in range(num_layers):
            if lvl == 0 or self.with_box_refine:
                cls_branch = self.cls_branches[lvl] if self.with_box_refine else self.cls_branches[0]
                reg_branch = self.reg_branches[lvl] if self.with_box_refine else self.reg_branches[0]
            
            cls_scores = cls_branch(hs[lvl])  # [B, num_query, num_classes+1]
            bbox_preds = reg_branch(hs[lvl])  # [B, num_query, 10]
            
            all_cls_scores.append(cls_scores)
            all_bbox_preds.append(bbox_preds)
        
        return {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'query_features': hs[-1],  # 最后一层的query特征
        }
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """
        计算损失
        
        Args:
            gt_bboxes_3d (list[BaseInstance3DBoxes]): GT 3D boxes
            gt_labels_3d (list[Tensor]): GT labels
            preds_dicts (dict): 预测结果
            
        Returns:
            dict: 损失字典
        """
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        
        # 准备GT
        gt_bboxes_list = [bbox.tensor.to(all_cls_scores[0].device) for bbox in gt_bboxes_3d]
        gt_labels_list = [label.to(all_cls_scores[0].device) for label in gt_labels_3d]
        
        # 计算每一层的损失
        loss_dict = {}
        num_layers = len(all_cls_scores)
        
        for layer_idx in range(num_layers):
            cls_scores = all_cls_scores[layer_idx]
            bbox_preds = all_bbox_preds[layer_idx]
            
            # 分配GT (使用匈牙利匹配)
            cls_targets, bbox_targets, pos_inds, neg_inds = self._assign_targets(
                cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list
            )
            
            # 分类损失
            cls_scores_flatten = cls_scores.reshape(-1, self.num_classes + 1)
            cls_targets_flatten = cls_targets.reshape(-1)
            
            loss_cls = self.loss_cls(
                cls_scores_flatten,
                cls_targets_flatten,
                avg_factor=max(pos_inds.sum(), 1)
            )
            
            # 回归损失 (只计算正样本)
            if pos_inds.sum() > 0:
                bbox_preds_pos = bbox_preds.reshape(-1, 10)[pos_inds]
                bbox_targets_pos = bbox_targets.reshape(-1, 10)[pos_inds]
                
                loss_bbox = self.loss_bbox(
                    bbox_preds_pos,
                    bbox_targets_pos,
                    avg_factor=pos_inds.sum()
                )
            else:
                loss_bbox = bbox_preds.sum() * 0
            
            # 添加到损失字典
            loss_dict[f'loss_cls_{layer_idx}'] = loss_cls
            loss_dict[f'loss_bbox_{layer_idx}'] = loss_bbox
        
        return loss_dict
    
    def _assign_targets(self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list):
        """
        使用匈牙利算法分配GT
        
        这是一个简化版本，实际使用时需要实现完整的匈牙利匹配
        """
        B = cls_scores.shape[0]
        num_query = cls_scores.shape[1]
        
        # 创建目标张量
        cls_targets = torch.zeros_like(cls_scores[..., 0], dtype=torch.long)
        bbox_targets = torch.zeros_like(bbox_preds)
        
        pos_inds = torch.zeros(B * num_query, dtype=torch.bool, device=cls_scores.device)
        neg_inds = torch.ones(B * num_query, dtype=torch.bool, device=cls_scores.device)
        
        # 简化的分配策略（实际应使用HungarianMatcher）
        for batch_idx in range(B):
            num_gt = len(gt_bboxes_list[batch_idx])
            if num_gt > 0:
                # 简单分配前num_gt个query给GT
                num_assigned = min(num_gt, num_query)
                start_idx = batch_idx * num_query
                
                cls_targets[batch_idx, :num_assigned] = gt_labels_list[batch_idx][:num_assigned]
                bbox_targets[batch_idx, :num_assigned] = self._bbox_to_target(
                    gt_bboxes_list[batch_idx][:num_assigned]
                )
                
                pos_inds[start_idx:start_idx + num_assigned] = True
                neg_inds[start_idx:start_idx + num_assigned] = False
        
        return cls_targets, bbox_targets, pos_inds, neg_inds
    
    def _bbox_to_target(self, bboxes):
        """
        将bbox转换为目标格式 [x, y, z, w, h, l, sin(yaw), cos(yaw), vx, vy]
        """
        if bboxes.shape[-1] == 7:  # [x, y, z, w, h, l, yaw]
            x, y, z, w, h, l, yaw = bboxes.unbind(-1)
            # 添加速度为0
            vx = torch.zeros_like(x)
            vy = torch.zeros_like(y)
        elif bboxes.shape[-1] == 9:  # [x, y, z, w, h, l, yaw, vx, vy]
            x, y, z, w, h, l, yaw, vx, vy = bboxes.unbind(-1)
        else:
            raise ValueError(f"Unsupported bbox format with {bboxes.shape[-1]} dims")
        
        target = torch.stack([
            x, y, z, w, h, l,
            torch.sin(yaw), torch.cos(yaw),
            vx, vy
        ], dim=-1)
        
        return target
    
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """
        生成最终的bbox预测
        
        Args:
            preds_dicts (dict): 预测字典
            img_metas (list): 图像元信息
            rescale (bool): 是否rescale
            
        Returns:
            list: 每个样本的预测结果 [(boxes, scores, labels), ...]
        """
        # 使用最后一层的预测
        cls_scores = preds_dicts['all_cls_scores'][-1]  # [B, num_query, num_classes+1]
        bbox_preds = preds_dicts['all_bbox_preds'][-1]  # [B, num_query, 10]
        
        B = cls_scores.shape[0]
        results = []
        
        for batch_idx in range(B):
            cls_score = cls_scores[batch_idx]  # [num_query, num_classes+1]
            bbox_pred = bbox_preds[batch_idx]  # [num_query, 10]
            
            # 获取分类分数和标签
            scores, labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            
            # 过滤低置信度预测
            score_threshold = 0.1
            keep = scores > score_threshold
            
            scores = scores[keep]
            labels = labels[keep]
            bbox_pred = bbox_pred[keep]
            
            # NMS
            if len(scores) > 0:
                # 转换bbox格式
                bboxes = self._target_to_bbox(bbox_pred)
                
                # TODO: 实现3D NMS
                # keep_nms = self._nms_3d(bboxes, scores)
                # bboxes = bboxes[keep_nms]
                # scores = scores[keep_nms]
                # labels = labels[keep_nms]
            else:
                bboxes = bbox_pred.new_zeros((0, 7))
            
            results.append((bboxes, scores, labels))
        
        return results
    
    def _target_to_bbox(self, targets):
        """
        将预测目标转换为bbox格式 [x, y, z, w, h, l, yaw]
        """
        x, y, z, w, h, l, sin_yaw, cos_yaw, vx, vy = targets.unbind(-1)
        yaw = torch.atan2(sin_yaw, cos_yaw)
        
        bboxes = torch.stack([x, y, z, w, h, l, yaw], dim=-1)
        return bboxes
