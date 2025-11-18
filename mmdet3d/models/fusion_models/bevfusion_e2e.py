"""
BEVFusion End-to-End Perception System
完整的端到端多任务感知算法

支持的任务:
1. 3D目标检测 (Object Detection)
2. BEV语义地图 (Semantic Segmentation)  
3. 多目标跟踪 (Multi-Object Tracking)

架构:
    多模态输入 → BEVFusion编码器 → BEV特征融合 → BEV特征提取器
                                                        ↓
                                            多任务DETR头
                                    ├─ 检测分支 (Detection)
                                    ├─ 分割分支 (Segmentation)
                                    └─ 跟踪分支 (Tracking)
"""
from typing import Any, Dict, List, Optional

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.models import FUSIONMODELS

from .bevfusion_detr import BEVFusionDETR

__all__ = ["BEVFusionE2E"]


@FUSIONMODELS.register_module()
class BEVFusionE2E(BEVFusionDETR):
    """
    BEVFusion端到端多任务感知系统
    
    在BEVFusionDETR基础上扩展:
    1. 统一的多任务DETR头
    2. 语义地图分割
    3. 多目标跟踪
    4. 轨迹预测
    
    Args:
        encoders (dict): 编码器配置
        fuser (dict): 融合器配置
        decoder (dict): 解码器配置
        bev_extractor (dict): BEV特征提取器配置
        heads (dict): 多任务头配置
            - perception: 统一的多任务DETR头
        task_weights (dict): 任务损失权重
        enable_tracking (bool): 是否启用跟踪
    """
    
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        bev_extractor: Dict[str, Any],
        heads: Dict[str, Any],
        task_weights: Dict[str, float] = None,
        enable_tracking: bool = True,
        **kwargs,
    ) -> None:
        # 调用父类初始化
        super().__init__(encoders, fuser, decoder, bev_extractor, heads, **kwargs)
        
        # 任务权重
        if task_weights is None:
            task_weights = {
                'detection': 1.0,
                'segmentation': 1.0,
                'tracking': 0.5 if enable_tracking else 0.0,
            }
        self.task_weights = task_weights
        
        # 跟踪状态
        self.enable_tracking = enable_tracking
        self.prev_frame_cache = {}  # 缓存前一帧信息
        
        # 统计信息
        self.task_stats = {
            'detection': {'count': 0, 'avg_objs': 0},
            'segmentation': {'count': 0, 'avg_pixels': 0},
            'tracking': {'count': 0, 'avg_tracks': 0},
        }
    
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_seg_masks=None,
        gt_track_ids=None,
        **kwargs,
    ):
        """
        端到端多任务前向传播
        
        新增参数:
            gt_seg_masks: GT语义分割mask [B, num_classes, H, W]
            gt_track_ids: GT跟踪ID列表
        """
        # ========== 第一步：多模态特征提取 ==========
        features = []
        auxiliary_losses = {}
        
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img, points, radar,
                    camera2ego, lidar2ego, lidar2camera,
                    lidar2image, camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix, metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            
            features.append(feature)
        
        if not self.training:
            features = features[::-1]
        
        # ========== 第二步：特征融合 ==========
        if self.fuser is not None:
            fused_bev = self.fuser(features)
        else:
            assert len(features) == 1
            fused_bev = features[0]
        
        batch_size = fused_bev.shape[0]
        
        # ========== 第三步：BEV解码 ==========
        x = self.decoder["backbone"](fused_bev)
        x = self.decoder["neck"](x)
        
        # ========== 第四步：BEV特征提取 ==========
        if self.bev_extractor is not None and self.extract_bev_feat:
            bev_features = self.bev_extractor(x)
        else:
            if isinstance(x, (list, tuple)):
                bev_feat = x[0]
            else:
                bev_feat = x
            
            B, C, H, W = bev_feat.shape
            bev_features = {
                'bev_features': bev_feat,
                'bev_flatten': bev_feat.flatten(2).permute(0, 2, 1),
            }
        
        # ========== 第五步：多任务感知 ==========
        # 获取前一帧信息（用于跟踪）
        prev_frame_info = self._get_prev_frame_info(metas) if self.enable_tracking else None
        
        if self.training:
            # ===== 训练模式 =====
            outputs = {}
            
            # 多任务DETR头
            if 'perception' in self.heads:
                # 使用统一的多任务头
                preds_dict = self.heads['perception'](
                    bev_features,
                    metas,
                    prev_frame_info
                )
                
                # 计算多任务损失
                losses = self.heads['perception'].loss(
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    gt_seg_masks=gt_seg_masks,
                    gt_track_ids=gt_track_ids,
                    preds_dict=preds_dict,
                )
                
                # 添加任务权重
                for task_name, weight in self.task_weights.items():
                    for loss_name, loss_value in losses.items():
                        if loss_name.startswith(task_name):
                            if loss_value.requires_grad:
                                outputs[f"loss/{loss_name}"] = loss_value * weight
                            else:
                                outputs[f"stats/{loss_name}"] = loss_value
            
            # 兼容性：如果有单独的object/map头
            for type, head in self.heads.items():
                if type in ['object', 'map'] and type != 'perception':
                    if type == "object":
                        pred_dict = head(x, metas)
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    elif type == "map":
                        losses = head(x, gt_masks_bev)
                    
                    for name, val in losses.items():
                        if val.requires_grad:
                            outputs[f"loss/{type}/{name}"] = val * self.loss_scale.get(type, 1.0)
                        else:
                            outputs[f"stats/{type}/{name}"] = val
            
            # 深度损失
            if self.use_depth_loss and 'depth' in auxiliary_losses:
                outputs["loss/depth"] = auxiliary_losses['depth']
            
            return outputs
        
        else:
            # ===== 推理模式 =====
            outputs = [{} for _ in range(batch_size)]
            
            if 'perception' in self.heads:
                # 使用统一的多任务头
                preds_dict = self.heads['perception'](
                    bev_features,
                    metas,
                    prev_frame_info
                )
                
                # 获取结果
                results = self.heads['perception'].get_results(preds_dict, metas)
                
                for k, result in enumerate(results):
                    outputs[k].update({
                        # 检测结果
                        "boxes_3d": result['boxes_3d'].to("cpu"),
                        "scores_3d": result['scores_3d'].cpu(),
                        "labels_3d": result['labels_3d'].cpu(),
                        
                        # 分割结果
                        "seg_mask": result['seg_mask'].cpu(),
                        
                        # 跟踪结果（如果启用）
                        "track_ids": result.get('track_ids', None),
                        "trajectories": result.get('trajectories', None),
                    })
                    
                    # 缓存信息用于下一帧
                    if self.enable_tracking and 'obj_embeds' in result:
                        self._cache_frame_info(metas[k], result)
            
            # 兼容性：使用单独的头
            else:
                for type, head in self.heads.items():
                    if type == "object":
                        pred_dict = head(x, metas)
                        bboxes = head.get_bboxes(pred_dict, metas)
                        
                        for k, (boxes, scores, labels) in enumerate(bboxes):
                            outputs[k].update({
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            })
                    
                    elif type == "map":
                        logits = head(x)
                        for k in range(batch_size):
                            outputs[k].update({
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu() if gt_masks_bev is not None else None,
                            })
            
            return outputs
    
    def _get_prev_frame_info(self, metas: List[Dict]) -> Optional[Dict]:
        """获取前一帧信息用于跟踪"""
        if not self.enable_tracking or not metas:
            return None
        
        # 使用scene token作为key
        scene_token = metas[0].get('scene_token', None)
        if scene_token is None:
            return None
        
        return self.prev_frame_cache.get(scene_token, None)
    
    def _cache_frame_info(self, meta: Dict, result: Dict):
        """缓存当前帧信息用于下一帧跟踪"""
        scene_token = meta.get('scene_token', None)
        if scene_token is None:
            return
        
        # 缓存目标embedding
        self.prev_frame_cache[scene_token] = {
            'obj_embeds': result['obj_embeds'],
            'boxes': result['boxes_3d'],
            'track_ids': result.get('track_ids', None),
            'timestamp': meta.get('timestamp', 0),
        }
        
        # 限制缓存大小
        if len(self.prev_frame_cache) > 100:
            # 删除最旧的
            oldest_key = min(
                self.prev_frame_cache.keys(),
                key=lambda k: self.prev_frame_cache[k].get('timestamp', 0)
            )
            del self.prev_frame_cache[oldest_key]
    
    def reset_tracking(self):
        """重置跟踪状态"""
        self.prev_frame_cache.clear()
    
    def extract_multi_task_features(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
    ):
        """
        提取多任务特征（不进行预测）
        
        Returns:
            dict: 多任务特征字典
                - 'bev_features': BEV特征
                - 'detection_queries': 检测query特征
                - 'segmentation_queries': 分割query特征
                - 'tracking_features': 跟踪特征（可选）
        """
        with torch.no_grad():
            # 提取BEV特征
            bev_features = self.extract_bev_features_only(
                img, points,
                camera2ego, lidar2ego, lidar2camera,
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas,
                depths=depths, radar=radar,
            )
            
            # 如果有多任务头，提取各任务的query特征
            if 'perception' in self.heads:
                prev_info = self._get_prev_frame_info(metas) if self.enable_tracking else None
                preds_dict = self.heads['perception'](bev_features, metas, prev_info)
                
                bev_features.update({
                    'detection_features': preds_dict['detection']['query_features'],
                    'segmentation_features': preds_dict['segmentation']['seg_features'],
                    'tracking_features': preds_dict.get('tracking', {}).get('obj_embeds', None),
                })
            
            return bev_features
    
    def visualize_predictions(self, results: List[Dict], save_dir: str = None):
        """
        可视化多任务预测结果
        
        Args:
            results: 预测结果列表
            save_dir: 保存目录
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for idx, result in enumerate(results):
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # 1. BEV检测可视化
            ax = axes[0, 0]
            self._plot_detection(ax, result)
            ax.set_title('3D Object Detection')
            
            # 2. 语义地图可视化
            ax = axes[0, 1]
            self._plot_segmentation(ax, result)
            ax.set_title('Semantic Map')
            
            # 3. 跟踪可视化
            ax = axes[1, 0]
            self._plot_tracking(ax, result)
            ax.set_title('Multi-Object Tracking')
            
            # 4. 轨迹预测可视化
            ax = axes[1, 1]
            self._plot_trajectories(ax, result)
            ax.set_title('Trajectory Prediction')
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'perception_{idx:04d}.png'), dpi=150)
                plt.close()
            else:
                plt.show()
    
    def _plot_detection(self, ax, result):
        """绘制检测结果"""
        boxes = result.get('boxes_3d', None)
        if boxes is None or len(boxes) == 0:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center')
            return
        
        boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        
        # 绘制BEV视图的框
        for box in boxes:
            x, y, w, l, yaw = box[0], box[1], box[3], box[4], box[6]
            # 简化绘制
            rect = plt.Rectangle((x-w/2, y-l/2), w, l, fill=False, color='r')
            ax.add_patch(rect)
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True)
    
    def _plot_segmentation(self, ax, result):
        """绘制分割结果"""
        seg_mask = result.get('seg_mask', None)
        if seg_mask is None:
            ax.text(0.5, 0.5, 'No segmentation', ha='center', va='center')
            return
        
        if torch.is_tensor(seg_mask):
            seg_mask = seg_mask.cpu().numpy()
        
        # 可视化多类别mask
        combined = seg_mask.argmax(axis=0)
        ax.imshow(combined, cmap='tab10')
        ax.axis('off')
    
    def _plot_tracking(self, ax, result):
        """绘制跟踪结果"""
        boxes = result.get('boxes_3d', None)
        track_ids = result.get('track_ids', None)
        
        if boxes is None or track_ids is None:
            ax.text(0.5, 0.5, 'No tracking', ha='center', va='center')
            return
        
        boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        track_ids = track_ids.cpu().numpy() if torch.is_tensor(track_ids) else track_ids
        
        # 绘制带ID的框
        for box, tid in zip(boxes, track_ids):
            x, y, w, l = box[0], box[1], box[3], box[4]
            rect = plt.Rectangle((x-w/2, y-l/2), w, l, fill=False, color='g')
            ax.add_patch(rect)
            ax.text(x, y, f'ID:{tid}', color='white', fontsize=8)
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True)
    
    def _plot_trajectories(self, ax, result):
        """绘制轨迹预测"""
        boxes = result.get('boxes_3d', None)
        trajs = result.get('trajectories', None)
        
        if boxes is None or trajs is None:
            ax.text(0.5, 0.5, 'No trajectories', ha='center', va='center')
            return
        
        boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        trajs = trajs.cpu().numpy() if torch.is_tensor(trajs) else trajs
        
        # 绘制当前位置和未来轨迹
        for box, traj in zip(boxes, trajs):
            x, y = box[0], box[1]
            ax.plot(x, y, 'ro', markersize=8)  # 当前位置
            
            # 未来轨迹
            traj_x = traj[:, 0] + x
            traj_y = traj[:, 1] + y
            ax.plot(traj_x, traj_y, 'b--', alpha=0.6)
            ax.plot(traj_x, traj_y, 'bo', markersize=4)
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True)
