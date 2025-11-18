# BEVFusion-DETR 集成技术文档

## 目录

1. [概述](#概述)
2. [系统架构](#系统架构)
3. [核心模块](#核心模块)
4. [实现细节](#实现细节)
5. [使用指南](#使用指南)
6. [配置说明](#配置说明)
7. [训练与评估](#训练与评估)
8. [常见问题](#常见问题)

---

## 概述

### 目标

将MIT-BEVFusion中的图像与点云融合后的统一BEV特征提取出来，经过处理后用于DETR（DEtection TRansformer）的输入，实现端到端的3D目标检测。

### 特点

- ✅ **统一BEV特征提取**：从融合后的多模态BEV特征中提取统一表示
- ✅ **灵活的特征处理**：可配置的特征处理模块，支持多层卷积和位置编码
- ✅ **DETR集成**：基于Transformer的3D目标检测头
- ✅ **模块化设计**：各模块独立，易于扩展和修改
- ✅ **兼容性**：完全兼容原始BEVFusion框架

---

## 系统架构

### 整体流程

```
┌─────────────┐     ┌─────────────┐
│   图像输入   │     │  点云输入   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ Camera       │     │ LiDAR       │
│ Encoder      │     │ Encoder     │
└──────┬──────┘     └──────┬──────┘
       │                   │
       │    ┌─────────┐    │
       └───►│  Fuser  │◄───┘
            └────┬────┘
                 │ 融合BEV特征
                 ▼
            ┌─────────┐
            │ Decoder │
            │ Backbone│
            └────┬────┘
                 │
                 ▼
            ┌─────────┐
            │ Decoder │
            │  Neck   │
            └────┬────┘
                 │
                 ▼
       ┌─────────────────┐
       │ BEV Feature     │  ← 关键：特征提取
       │ Extractor       │
       └────┬────────────┘
            │
            ├─► bev_features (B, C, H, W)
            ├─► bev_flatten (B, H*W, C)
            └─► position_encoding (B, C, H, W)
                 │
                 ▼
       ┌─────────────────┐
       │ DETR Head       │
       │ - Transformer   │
       │ - Query Embed   │
       │ - Cls + Reg     │
       └────┬────────────┘
            │
            ▼
       ┌─────────────────┐
       │ 3D Bbox + Score │
       └─────────────────┘
```

### 核心组件

1. **BEVFeatureExtractor**：BEV特征提取和处理模块
2. **DETRHead3D**：基于DETR的3D检测头
3. **BEVFusionDETR**：集成上述模块的完整模型

---

## 核心模块

### 1. BEV特征提取器 (BEVFeatureExtractor)

**位置**: `mmdet3d/models/necks/bev_feature_extractor.py`

#### 功能

- 对融合后的BEV特征进行多层卷积处理
- 生成2D位置编码
- 提供多种特征输出格式（2D、1D展平）

#### 主要参数

```yaml
bev_extractor:
  type: BEVFeatureExtractor
  in_channels: 512        # 输入通道数
  out_channels: 256       # 输出通道数
  num_layers: 3           # 卷积层数
  feat_h: 180            # BEV特征高度
  feat_w: 180            # BEV特征宽度
  use_position_encoding: true  # 是否使用位置编码
```

#### 输出

```python
{
    'bev_features': Tensor,      # [B, C, H, W] - 2D BEV特征
    'bev_flatten': Tensor,       # [B, H*W, C] - 展平BEV特征
    'position_encoding': Tensor, # [B, C, H, W] - 位置编码
    'bev_features_with_pos': Tensor  # [B, C, H, W] - 带位置编码的特征
}
```

#### 位置编码

使用正弦-余弦位置编码，公式如下：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

其中：
- `pos`: 位置坐标 (x, y)
- `i`: 通道索引
- `d`: 特征维度

### 2. DETR检测头 (DETRHead3D)

**位置**: `mmdet3d/models/heads/bbox/detr_head.py`

#### 功能

- 使用Transformer Decoder处理BEV特征
- 生成可学习的object queries
- 预测3D bbox和类别

#### 主要参数

```yaml
heads:
  object:
    type: DETRHead3D
    num_classes: 10         # 类别数
    in_channels: 256        # 输入通道数
    num_query: 900          # query数量
    num_reg_fcs: 2          # 回归分支FC层数
    
    transformer:
      num_layers: 6         # Decoder层数
      num_heads: 8          # 注意力头数
      ffn_dim: 2048         # FFN维度
      dropout: 0.1          # Dropout率
    
    with_box_refine: true   # 是否使用box refinement
```

#### Bbox表示

预测10维向量：`[x, y, z, w, h, l, sin(θ), cos(θ), vx, vy]`

- `(x, y, z)`: 3D中心坐标
- `(w, h, l)`: 宽度、高度、长度
- `(sin(θ), cos(θ))`: 航向角的三角函数表示
- `(vx, vy)`: 速度（可选）

#### 损失函数

1. **分类损失**: Focal Loss
   ```python
   loss_cls = FocalLoss(
       use_sigmoid=True,
       gamma=2.0,
       alpha=0.25,
       loss_weight=2.0
   )
   ```

2. **回归损失**: L1 Loss
   ```python
   loss_bbox = L1Loss(loss_weight=0.25)
   ```

3. **IoU损失**: GIoU Loss (可选)
   ```python
   loss_iou = GIoULoss(loss_weight=0.0)
   ```

### 3. BEVFusionDETR模型

**位置**: `mmdet3d/models/fusion_models/bevfusion_detr.py`

#### 功能

继承自原始BEVFusion，添加：
- BEV特征提取功能
- DETR检测头集成
- 特征提取专用接口

#### 关键方法

```python
class BEVFusionDETR(BEVFusion):
    
    def forward_single(self, ...):
        """
        前向传播
        1. 提取多模态特征
        2. 融合特征
        3. Decoder处理
        4. BEV特征提取 ← 新增
        5. DETR检测
        """
        pass
    
    def extract_bev_features_only(self, ...):
        """
        仅提取BEV特征，不进行检测
        用于特征可视化、分析
        """
        pass
```

---

## 实现细节

### 特征维度变换

```
输入图像: [B, N, 3, H, W]  N=相机数量
    ↓ Camera Backbone + Neck
特征图: [B, N, C, H', W']
    ↓ View Transform
Camera BEV: [B, C1, H_bev, W_bev]

点云: [B, N_points, 3+features]
    ↓ Voxelize + Sparse Conv
LiDAR BEV: [B, C2, H_bev, W_bev]

    ↓ Fuser (Concat + Conv)
融合BEV: [B, C_fused, H_bev, W_bev]
    ↓ Decoder Backbone
    [B, C_dec, H_bev, W_bev]
    ↓ Decoder Neck (FPN)
    [B, C_neck, H_bev, W_bev]
    ↓ BEV Feature Extractor
    [B, C_out, H_bev, W_bev]
    ↓ Flatten
    [B, H_bev*W_bev, C_out]
    ↓ DETR Transformer
    [B, num_query, C_out]
    ↓ Classification + Regression Heads
预测: [B, num_query, num_classes+1+10]
```

### Transformer架构

```
Query Embedding: [B, num_query, C]
Memory (BEV Features): [B, H*W, C]

┌────────────────────────────────────┐
│  Transformer Decoder Layer × N     │
│                                    │
│  ┌─────────────────────────────┐  │
│  │ Self-Attention (Queries)    │  │
│  └─────────────────────────────┘  │
│             ↓                      │
│  ┌─────────────────────────────┐  │
│  │ Cross-Attention              │  │
│  │ (Queries ← BEV Features)    │  │
│  └─────────────────────────────┘  │
│             ↓                      │
│  ┌─────────────────────────────┐  │
│  │ Feed Forward Network        │  │
│  └─────────────────────────────┘  │
└────────────────────────────────────┘
          ↓
Updated Queries: [B, num_query, C]
          ↓
┌────────────────────────────────────┐
│  Classification Head               │
│  Output: [B, num_query, K+1]       │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│  Regression Head                   │
│  Output: [B, num_query, 10]        │
└────────────────────────────────────┘
```

---

## 使用指南

### 1. 环境准备

```bash
# 克隆仓库
cd /path/to/bevfusion

# 安装依赖
pip install -r requirements.txt

# 编译CUDA算子
cd mmdet3d/ops
python setup.py develop
```

### 2. 数据准备

```bash
# 准备nuScenes数据
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes
```

### 3. 训练

```bash
# 单GPU训练
python tools/train.py configs/nuscenes/det/bevfusion-detr.yaml

# 多GPU训练
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-detr.yaml \
    8  # GPU数量
```

### 4. 评估

```bash
python tools/test.py \
    configs/nuscenes/det/bevfusion-detr.yaml \
    work_dirs/bevfusion_detr/latest.pth \
    --eval bbox
```

### 5. BEV特征提取

```bash
# 提取并保存BEV特征
python examples/extract_bev_features_detr.py \
    configs/nuscenes/det/bevfusion-detr.yaml \
    work_dirs/bevfusion_detr/latest.pth \
    --out-dir output/bev_features \
    --save-features \
    --visualize
```

### 6. 推理

```python
import torch
from mmdet3d.models import build_model
from mmcv import Config

# 加载配置和模型
cfg = Config.fromfile('configs/nuscenes/det/bevfusion-detr.yaml')
model = build_model(cfg.model)
model.eval()

# 准备数据
# data = ...

# 提取BEV特征
with torch.no_grad():
    bev_features = model.extract_bev_features_only(
        img=data['img'],
        points=data['points'],
        # ... 其他参数
    )

# BEV特征可用于：
# 1. 可视化
# 2. 下游任务
# 3. 特征分析
```

---

## 配置说明

### 完整配置示例

详见 `configs/nuscenes/det/bevfusion-detr.yaml`

### 关键配置项

#### 1. BEV特征提取器

```yaml
bev_extractor:
  type: BEVFeatureExtractor
  in_channels: 512        # 必须匹配decoder neck的输出
  out_channels: 256       # DETR输入通道数
  num_layers: 3           # 特征处理层数，建议2-4层
  feat_h: 180            # BEV网格高度
  feat_w: 180            # BEV网格宽度
  use_position_encoding: true
```

#### 2. DETR检测头

```yaml
heads:
  object:
    type: DETRHead3D
    num_classes: 10
    in_channels: 256      # 必须匹配bev_extractor的out_channels
    num_query: 900        # query数量，影响检测能力和计算量
    
    transformer:
      num_layers: 6       # 更多层→更强表达能力，但更慢
      num_heads: 8        # 注意力头数
      ffn_dim: 2048       # FFN隐藏层维度
      dropout: 0.1
    
    with_box_refine: true  # 建议开启，提升精度
```

#### 3. 损失权重调整

```yaml
loss_scale:
  object: 1.0             # 检测损失权重
  map: 1.0                # 分割损失权重（如果使用）

# DETR损失权重
loss_cls:
  loss_weight: 2.0        # 分类损失
loss_bbox:
  loss_weight: 0.25       # 回归损失
loss_iou:
  loss_weight: 0.0        # IoU损失（可选）
```

---

## 训练与评估

### 训练策略

#### 1. 学习率设置

```yaml
optimizer:
  type: AdamW
  lr: 2.0e-4              # 基础学习率
  weight_decay: 0.01

lr_config:
  policy: CosineAnnealing
  warmup: linear
  warmup_iters: 500
  warmup_ratio: 0.33333333
  min_lr_ratio: 1.0e-3
```

#### 2. 数据增强

```yaml
# 2D增强
augment2d:
  resize: [[0.38, 0.55], [0.48, 0.48]]
  rotate: [-5.4, 5.4]
  gridmask:
    prob: 0.0
    fixed_prob: true

# 3D增强
augment3d:
  scale: [0.9, 1.1]
  rotate: [-0.78539816, 0.78539816]  # ±π/4
  translate: 0.5
```

#### 3. 训练技巧

- **渐进式训练**：先训练BEVFusion backbone，再fine-tune DETR head
- **冻结策略**：初期冻结encoder，只训练DETR
- **梯度裁剪**：使用梯度裁剪防止梯度爆炸

```yaml
optimizer_config:
  grad_clip:
    max_norm: 35
    norm_type: 2
```

### 评估指标

#### nuScenes检测指标

- **NDS** (nuScenes Detection Score): 综合评分
- **mAP**: 平均精度
- **mATE**: 平移误差
- **mASE**: 尺度误差
- **mAOE**: 方向误差
- **mAVE**: 速度误差
- **mAAE**: 属性误差

#### 示例输出

```
+----------+--------+--------+--------+--------+
| Category | AP     | ATE    | ASE    | AOE    |
+----------+--------+--------+--------+--------+
| car      | 0.856  | 0.245  | 0.152  | 0.098  |
| truck    | 0.723  | 0.312  | 0.178  | 0.125  |
| ...      | ...    | ...    | ...    | ...    |
+----------+--------+--------+--------+--------+
| Mean     | 0.789  | 0.278  | 0.165  | 0.112  |
+----------+--------+--------+--------+--------+

NDS: 0.712
mAP: 0.789
```

---

## 常见问题

### Q1: 如何调整query数量？

**A**: 修改配置中的`num_query`参数。建议：
- nuScenes: 900 (场景较大，目标多)
- KITTI: 300 (场景较小)
- 自定义数据集: 根据平均目标数×3-5倍

### Q2: 显存不足怎么办？

**A**: 尝试以下方法：
1. 减少batch size
2. 减少num_query
3. 减少transformer层数
4. 使用gradient checkpointing
5. 减小图像分辨率

```yaml
# 低显存配置
image_size: [224, 608]  # 从[256, 704]降低
num_query: 600          # 从900降低
transformer:
  num_layers: 4         # 从6降低
```

### Q3: BEV特征尺寸不匹配？

**A**: 确保以下参数一致：
- `bev_extractor.feat_h/feat_w` = BEV网格尺寸
- `point_cloud_range` / `voxel_size` 决定BEV尺寸

示例：
```python
# point_cloud_range = [-54, -54, -5, 54, 54, 3]
# voxel_size = [0.6, 0.6, 8]
# BEV size = (54-(-54))/0.6 = 180
feat_h: 180
feat_w: 180
```

### Q4: 如何可视化BEV特征？

**A**: 使用提供的可视化脚本：

```bash
python examples/extract_bev_features_detr.py \
    config.yaml checkpoint.pth \
    --visualize
```

或在代码中：

```python
import matplotlib.pyplot as plt
import numpy as np

# 提取特征
bev_feat = bev_features['bev_features'][0].cpu().numpy()  # [C, H, W]

# L2范数可视化
feat_norm = np.linalg.norm(bev_feat, axis=0)
plt.imshow(feat_norm, cmap='viridis')
plt.colorbar()
plt.show()
```

### Q5: 如何进行迁移学习？

**A**: 

```python
# 1. 加载预训练权重
checkpoint = torch.load('pretrained.pth')

# 2. 过滤DETR head权重（如果类别数不同）
state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
              if not k.startswith('heads.object')}

# 3. 加载
model.load_state_dict(state_dict, strict=False)

# 4. 冻结encoder（可选）
for param in model.encoders.parameters():
    param.requires_grad = False
```

---

## 性能优化

### 推理加速

#### 1. TensorRT部署

```bash
# 导出ONNX
python tools/export.py \
    config.yaml checkpoint.pth \
    --format onnx

# 转换TensorRT
trtexec --onnx=model.onnx \
    --saveEngine=model.trt \
    --fp16
```

#### 2. 量化

```python
# INT8量化
import torch.quantization as quantization

model.qconfig = quantization.get_default_qconfig('fbgemm')
model_prepared = quantization.prepare(model)
# ... 校准数据 ...
model_quantized = quantization.convert(model_prepared)
```

### 训练加速

#### 1. 混合精度训练

```yaml
# 在配置中启用
fp16:
  loss_scale: 512.0
```

#### 2. 分布式训练

```bash
# 多节点训练
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    tools/train.py config.yaml
```

---

## 扩展与定制

### 自定义BEV特征提取器

```python
from mmdet.models.builder import NECKS

@NECKS.register_module()
class CustomBEVExtractor(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 自定义实现
    
    def forward(self, x):
        # 返回与BEVFeatureExtractor相同格式
        return {
            'bev_features': ...,
            'bev_flatten': ...,
        }
```

### 自定义DETR Head

```python
from mmdet.models.builder import HEADS

@HEADS.register_module()
class CustomDETRHead(DETRHead3D):
    def __init__(self, ...):
        super().__init__(...)
        # 添加自定义模块
    
    def forward(self, bev_features, img_metas):
        # 自定义前向传播
        pass
```

---

## 参考文献

1. BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation
2. End-to-End Object Detection with Transformers (DETR)
3. Deformable DETR: Deformable Transformers for End-to-End Object Detection
4. nuScenes: A multimodal dataset for autonomous driving

---

## 更新日志

- **2024-11**: 初始版本
  - BEV特征提取器
  - DETR检测头
  - 完整文档

---

## 联系与支持

如有问题或建议，请提交Issue或Pull Request。
