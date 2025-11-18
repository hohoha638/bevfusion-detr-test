"""
BEVFusion with DETR Integration
集成DETR的BEVFusion模型，用于提取融合BEV特征并使用DETR进行检测
"""
from typing import Any, Dict

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

from .bevfusion import BEVFusion

__all__ = ["BEVFusionDETR"]


@FUSIONMODELS.register_module()
class BEVFusionDETR(BEVFusion):
    """
    集成DETR的BEVFusion模型
    
    在原始BEVFusion基础上，添加了：
    1. BEV特征提取器
    2. DETR检测头
    
    Args:
        encoders (dict): 编码器配置
        fuser (dict): 融合器配置
        decoder (dict): 解码器配置
        bev_extractor (dict): BEV特征提取器配置
        heads (dict): 检测头配置
    """
    
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        bev_extractor: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        # 调用父类初始化
        super().__init__(encoders, fuser, decoder, heads, **kwargs)
        
        # BEV特征提取器
        if bev_extractor is not None:
            self.bev_extractor = build_neck(bev_extractor)
        else:
            self.bev_extractor = None
        
        # 是否提取BEV特征
        self.extract_bev_feat = kwargs.get('extract_bev_feat', True)
    
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
        **kwargs,
    ):
        """
        前向传播（单帧）
        
        与原始BEVFusion的主要区别：
        1. 在融合后提取BEV特征
        2. 将提取的特征传递给DETR头
        """
        # ========== 第一步：提取多模态特征 ==========
        features = []
        auxiliary_losses = {}
        
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
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
        
        # ========== 第二步：融合特征 ==========
        if self.fuser is not None:
            fused_bev = self.fuser(features)
        else:
            assert len(features) == 1, features
            fused_bev = features[0]
        
        batch_size = fused_bev.shape[0]
        
        # ========== 第三步：Decoder处理 ==========
        x = self.decoder["backbone"](fused_bev)
        x = self.decoder["neck"](x)
        
        # ========== 第四步：BEV特征提取（关键步骤）==========
        if self.bev_extractor is not None and self.extract_bev_feat:
            # 提取并处理BEV特征
            bev_features = self.bev_extractor(x)
        else:
            # 直接使用decoder输出
            if isinstance(x, (list, tuple)):
                # 如果是多尺度特征，取第一个
                bev_feat = x[0]
            else:
                bev_feat = x
            
            # 构建特征字典
            B, C, H, W = bev_feat.shape
            bev_features = {
                'bev_features': bev_feat,
                'bev_flatten': bev_feat.flatten(2).permute(0, 2, 1),
            }
        
        # ========== 第五步：检测头预测 ==========
        if self.training:
            outputs = {}
            
            for type, head in self.heads.items():
                if type == "object":
                    # DETR-based detection
                    if hasattr(head, 'forward') and 'bev_features' in head.forward.__code__.co_varnames:
                        # DETR头，传入BEV特征
                        pred_dict = head(bev_features, metas)
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    else:
                        # 传统检测头，使用原始特征
                        pred_dict = head(x, metas)
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                
                # 添加损失
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            
            # 添加深度损失
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            return outputs
        
        else:
            # ========== 推理模式 ==========
            outputs = [{} for _ in range(batch_size)]
            
            for type, head in self.heads.items():
                if type == "object":
                    # DETR-based detection
                    if hasattr(head, 'forward') and 'bev_features' in head.forward.__code__.co_varnames:
                        pred_dict = head(bev_features, metas)
                        bboxes = head.get_bboxes(pred_dict, metas)
                    else:
                        # 传统检测头
                        pred_dict = head(x, metas)
                        bboxes = head.get_bboxes(pred_dict, metas)
                    
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu() if gt_masks_bev is not None else None,
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            
            return outputs
    
    def extract_bev_features_only(
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
        仅提取BEV特征（不进行检测）
        
        用于特征可视化、分析或下游任务
        
        Returns:
            dict: BEV特征字典
                - 'fused_bev': 融合后的原始BEV特征 [B, C, H, W]
                - 'processed_bev': 处理后的BEV特征 [B, C, H, W]
                - 'bev_flatten': 展平的BEV特征 [B, N, C]
                - 'position_encoding': 位置编码 (可选)
        """
        with torch.no_grad():
            # 提取多模态特征
            features = []
            for sensor in list(self.encoders.keys()):
                if sensor == "camera":
                    feature = self.extract_camera_features(
                        img, points, radar,
                        camera2ego, lidar2ego, lidar2camera,
                        lidar2image, camera_intrinsics, camera2lidar,
                        img_aug_matrix, lidar_aug_matrix, metas,
                        gt_depths=depths,
                    )
                    if self.use_depth_loss:
                        feature = feature[0]
                elif sensor == "lidar":
                    feature = self.extract_features(points, sensor)
                elif sensor == "radar":
                    feature = self.extract_features(radar, sensor)
                
                features.append(feature)
            
            # 融合
            if self.fuser is not None:
                fused_bev = self.fuser(features)
            else:
                fused_bev = features[0]
            
            # Decoder
            x = self.decoder["backbone"](fused_bev)
            x = self.decoder["neck"](x)
            
            # BEV特征提取
            if self.bev_extractor is not None:
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
            
            # 添加原始融合特征
            bev_features['fused_bev'] = fused_bev
            bev_features['processed_bev'] = bev_features['bev_features']
            
            return bev_features
