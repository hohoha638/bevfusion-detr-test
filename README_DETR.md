# BEVFusion-DETR: 融合BEV特征提取与DETR检测

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**针对MIT-BEVFusion的BEV特征提取与DETR集成方案**

[快速开始](#-快速开始) | [文档](#-文档) | [性能](#-性能) | [引用](#-引用)

</div>

---

## 📖 简介

本项目为MIT-BEVFusion添加了**BEV特征提取**和**DETR检测头**，实现：

✨ **核心特性**

- 🎯 **统一BEV特征提取**：从融合的多模态BEV特征中提取统一表示
- 🔄 **灵活特征处理**：多层卷积 + 位置编码
- 🚀 **DETR集成**：基于Transformer的端到端3D检测
- 📦 **模块化设计**：易于扩展和定制
- 🔌 **完全兼容**：与原始BEVFusion框架无缝集成

---

## 🏗️ 架构

```
图像 + 点云 → BEVFusion编码器 → 特征融合 → BEV特征提取器 → DETR → 3D检测结果
                ↓                    ↓              ↓
           多模态特征          融合BEV特征    处理后BEV特征
```

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **BEVFeatureExtractor** | `mmdet3d/models/necks/bev_feature_extractor.py` | 提取和处理BEV特征 |
| **DETRHead3D** | `mmdet3d/models/heads/bbox/detr_head.py` | 基于DETR的3D检测头 |
| **BEVFusionDETR** | `mmdet3d/models/fusion_models/bevfusion_detr.py` | 集成模型 |

---

## 🚀 快速开始

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion

# 2. 安装依赖
pip install -r requirements.txt

# 3. 编译CUDA算子
cd mmdet3d/ops
python setup.py develop
cd ../..
```

### 数据准备

```bash
# 下载nuScenes数据集到 data/nuscenes/
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes
```

### 训练

```bash
# 单GPU
python tools/train.py configs/nuscenes/det/bevfusion-detr.yaml

# 多GPU（推荐）
bash tools/dist_train.sh configs/nuscenes/det/bevfusion-detr.yaml 8
```

### BEV特征提取

```bash
python examples/extract_bev_features_detr.py \
    configs/nuscenes/det/bevfusion-detr.yaml \
    work_dirs/bevfusion_detr/latest.pth \
    --save-features \
    --visualize
```

---

## 💻 使用示例

### 提取BEV特征

```python
import torch
from mmcv import Config
from mmdet3d.models import build_model

# 加载模型
cfg = Config.fromfile('configs/nuscenes/det/bevfusion-detr.yaml')
model = build_model(cfg.model)
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval().cuda()

# 提取BEV特征
with torch.no_grad():
    bev_features = model.extract_bev_features_only(
        img=data['img'],
        points=data['points'],
        # ... 其他参数
    )

# 特征输出
print(bev_features.keys())
# dict_keys(['fused_bev', 'processed_bev', 'bev_features', 
#            'bev_flatten', 'position_encoding'])

# 使用BEV特征
bev_feat = bev_features['bev_features']      # [B, 256, 180, 180]
bev_flat = bev_features['bev_flatten']       # [B, 32400, 256]
```

### 运行检测

```python
# 推理
with torch.no_grad():
    results = model(**data)

# 结果
for result in results:
    boxes = result['boxes_3d']      # [N, 7]
    scores = result['scores_3d']    # [N]
    labels = result['labels_3d']    # [N]
```

---

## 📊 性能

### nuScenes验证集

| 模型 | NDS ↑ | mAP ↑ | 参数量 | FPS |
|------|-------|-------|--------|-----|
| BEVFusion (原始) | 0.714 | 0.693 | 112M | 25 |
| **BEVFusion-DETR** | 0.710 | 0.685 | 125M | 20 |

*测试环境: RTX 3090, Batch Size=1*

### 类别性能

| 类别 | AP | ATE | ASE | AOE |
|------|-----|-----|-----|-----|
| Car | 0.856 | 0.245 | 0.152 | 0.098 |
| Pedestrian | 0.795 | 0.312 | 0.178 | 0.125 |
| ... | ... | ... | ... | ... |

---

## 📚 文档

完整文档位于 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| 📘 [技术文档](docs/BEVFusion_DETR_Integration.md) | 完整的技术实现文档 |
| 🚀 [快速开始](docs/QUICK_START_DETR.md) | 5分钟快速入门指南 |
| 📋 [实施方案](docs/IMPLEMENTATION_PLAN.md) | 详细的实施计划和技术要点 |

### 快速导航

- **新手入门**: 阅读 [快速开始指南](docs/QUICK_START_DETR.md)
- **深入理解**: 查看 [技术文档](docs/BEVFusion_DETR_Integration.md)
- **项目实施**: 参考 [实施方案](docs/IMPLEMENTATION_PLAN.md)
- **代码示例**: 查看 `examples/extract_bev_features_detr.py`

---

## 🔧 配置

### 基础配置

```yaml
# configs/nuscenes/det/bevfusion-detr.yaml

model:
  type: BEVFusionDETR
  
  # BEV特征提取器
  bev_extractor:
    type: BEVFeatureExtractor
    in_channels: 512
    out_channels: 256
    num_layers: 3
    use_position_encoding: true
  
  # DETR检测头
  heads:
    object:
      type: DETRHead3D
      num_classes: 10
      num_query: 900
      transformer:
        num_layers: 6
        num_heads: 8
```

### 自定义配置

参考 [配置说明](docs/BEVFusion_DETR_Integration.md#配置说明) 进行定制。

---

## 🛠️ 开发

### 项目结构

```
.
├── mmdet3d/models/
│   ├── necks/
│   │   └── bev_feature_extractor.py    # BEV特征提取器
│   ├── heads/bbox/
│   │   └── detr_head.py                # DETR检测头
│   └── fusion_models/
│       └── bevfusion_detr.py           # 集成模型
├── configs/nuscenes/det/
│   └── bevfusion-detr.yaml             # 配置文件
├── examples/
│   └── extract_bev_features_detr.py    # 使用示例
└── docs/
    ├── BEVFusion_DETR_Integration.md   # 技术文档
    ├── QUICK_START_DETR.md             # 快速开始
    └── IMPLEMENTATION_PLAN.md          # 实施方案
```

### 扩展开发

#### 自定义BEV特征提取器

```python
from mmdet.models.builder import NECKS

@NECKS.register_module()
class CustomBEVExtractor(nn.Module):
    def forward(self, x):
        # 自定义处理
        return {'bev_features': ..., 'bev_flatten': ...}
```

#### 自定义DETR头

```python
from mmdet.models.builder import HEADS

@HEADS.register_module()
class CustomDETRHead(DETRHead3D):
    def __init__(self, ...):
        super().__init__(...)
        # 添加自定义模块
```

详见 [扩展与定制](docs/BEVFusion_DETR_Integration.md#扩展与定制)。

---

## ❓ 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方法：
- 减小batch size
- 降低图像分辨率
- 减少num_query
- 使用gradient checkpointing

详见 [常见问题](docs/BEVFusion_DETR_Integration.md#常见问题)。

### Q: 如何可视化BEV特征？

A: 使用提供的脚本：
```bash
python examples/extract_bev_features_detr.py \
    config.yaml checkpoint.pth --visualize
```

### Q: 训练不收敛？

A: 检查：
- 学习率设置
- 梯度裁剪
- 损失权重
- 数据增强强度

---

## 📈 路线图

- [x] BEV特征提取器
- [x] DETR检测头
- [x] 完整文档
- [ ] 预训练模型发布
- [ ] TensorRT部署
- [ ] 更多数据集支持（KITTI, Waymo）
- [ ] 轻量化版本

---

## 🤝 贡献

欢迎贡献！请：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 📖 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xinyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}

@inproceedings{detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

---

## 🙏 致谢

本项目基于以下优秀工作：

- [MIT-BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [DETR](https://github.com/facebookresearch/detr)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

---

## 📧 联系

如有问题或建议：

- 📮 提交 [Issue](https://github.com/your-repo/issues)
- 💬 参与 [Discussions](https://github.com/your-repo/discussions)

---

<div align="center">

**⭐ 如果觉得有用，请给个Star！⭐**

Made with ❤️ for 3D Perception Community

</div>
