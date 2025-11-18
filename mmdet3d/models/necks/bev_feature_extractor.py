"""
BEV Feature Extractor for DETR Integration
提取融合后的BEV特征并进行处理，用于DETR输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS

__all__ = ["BEVFeatureExtractor"]


@NECKS.register_module()
class BEVFeatureExtractor(BaseModule):
    """
    BEV特征提取器，用于处理融合后的BEV特征
    
    Args:
        in_channels (int): 输入特征通道数
        out_channels (int): 输出特征通道数
        num_layers (int): 处理层数
        feat_h (int): BEV特征高度
        feat_w (int): BEV特征宽度
        use_position_encoding (bool): 是否使用位置编码
    """
    
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        num_layers=3,
        feat_h=128,
        feat_w=128,
        use_position_encoding=True,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.use_position_encoding = use_position_encoding
        
        # 特征处理层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.conv_layers.append(
                ConvModule(
                    in_ch,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        
        # 位置编码
        if use_position_encoding:
            self.position_encoding = self._build_position_encoding(
                out_channels, feat_h, feat_w
            )
        else:
            self.position_encoding = None
    
    def _build_position_encoding(self, channels, h, w):
        """构建2D位置编码"""
        # 生成位置编码网格
        y_embed = torch.arange(h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        
        # 归一化到[-1, 1]
        y_embed = 2 * y_embed / (h - 1) - 1
        x_embed = 2 * x_embed / (w - 1) - 1
        
        # 构建位置编码
        dim_t = torch.arange(channels // 4, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (channels // 4))
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)
        
        return nn.Parameter(pos, requires_grad=False)
    
    @auto_fp16()
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 融合后的BEV特征 [B, C, H, W]
            
        Returns:
            dict: 包含处理后特征的字典
                - 'bev_features': BEV特征 [B, C, H, W]
                - 'bev_flatten': 展平的BEV特征 [B, H*W, C]
                - 'position_encoding': 位置编码 [1, C, H, W] (可选)
        """
        # 通过卷积层处理特征
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        B, C, H, W = x.shape
        
        # 准备输出字典
        output = {
            'bev_features': x,  # [B, C, H, W]
            'bev_flatten': x.flatten(2).permute(0, 2, 1),  # [B, H*W, C]
        }
        
        # 添加位置编码
        if self.use_position_encoding and self.position_encoding is not None:
            pos_encoding = self.position_encoding.unsqueeze(0).expand(B, -1, -1, -1)
            output['position_encoding'] = pos_encoding
            output['bev_features_with_pos'] = x + pos_encoding
        
        return output
    
    def extract_roi_features(self, x, rois):
        """
        从BEV特征中提取ROI特征
        
        Args:
            x (Tensor): BEV特征 [B, C, H, W]
            rois (Tensor): ROI坐标 [N, 5] (batch_idx, x1, y1, x2, y2)
            
        Returns:
            Tensor: ROI特征 [N, C, roi_h, roi_w]
        """
        from torchvision.ops import roi_align
        
        # 使用ROI Align提取特征
        roi_features = roi_align(
            x, 
            rois, 
            output_size=(7, 7),  # 可配置
            spatial_scale=1.0,
            sampling_ratio=2
        )
        
        return roi_features
