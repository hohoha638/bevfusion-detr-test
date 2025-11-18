# BEVFusion 端到端多任务感知系统 - 完整安装指南

## 📋 目录

- [系统要求](#系统要求)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [模型配置](#模型配置)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

---

## 🖥️ 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **GPU** | NVIDIA RTX 2080 Ti (11GB) | NVIDIA RTX 3090 (24GB) |
| **CPU** | 8核 | 16核+ |
| **内存** | 32GB | 64GB+ |
| **存储** | 200GB SSD | 500GB+ NVMe SSD |

### 软件要求

- **操作系统**: Ubuntu 18.04 / 20.04 / 22.04
- **CUDA**: 11.1 / 11.3 / 11.7
- **Python**: 3.8 / 3.9
- **PyTorch**: 1.9+ (推荐1.10.0)

---

## 🚀 环境安装

### 步骤 1: 创建 Conda 环境

```bash
# 创建新环境
conda create -n bevfusion-e2e python=3.8 -y
conda activate bevfusion-e2e

# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# CUDA 11.1
# conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

### 步骤 2: 安装 MMDetection3D 依赖

```bash
# 安装 MMEngine
pip install openmim
mim install mmengine

# 安装 MMCV
mim install "mmcv-full>=1.4.0,<1.7.0"

# 安装 MMDetection
mim install "mmdet>=2.24.0,<3.0.0"

# 安装 MMSegmentation (用于分割任务)
mim install "mmsegmentation>=0.20.0,<1.0.0"
```

### 步骤 3: 克隆仓库

```bash
# 克隆BEVFusion仓库
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion

# 或使用你的定制版本
# cd /path/to/your/bevfusion
```

### 步骤 4: 安装项目依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装额外的依赖（端到端系统需要）
pip install scipy scikit-learn matplotlib seaborn \
    opencv-python pillow tensorboard \
    numba nuscenes-devkit motmetrics
```

### 步骤 5: 编译 CUDA 扩展

```bash
# 编译自定义算子
cd mmdet3d/ops
python setup.py develop
cd ../..

# 验证编译
python -c "import mmdet3d; print(mmdet3d.__version__)"
```

### 步骤 6: 安装额外工具（可选但推荐）

```bash
# 可视化工具
pip install open3d mayavi vtk

# 性能分析
pip install tensorboard wandb

# 视频处理
pip install imageio imageio-ffmpeg

# Jupyter支持
pip install jupyter ipywidgets
```

---

## 📊 数据准备

### nuScenes 数据集

#### 1. 下载数据

```bash
# 创建数据目录
mkdir -p data/nuscenes
cd data/nuscenes

# 下载数据（需要在 https://www.nuscenes.org/nuscenes 注册）
# 下载以下文件：
# - Full dataset (v1.0): 
#   * Metadata (All)
#   * Sensor blobs (Camera, LiDAR, Radar)
# - Mini dataset (v1.0): 用于快速测试

# 使用wget或其他工具下载
# wget <download_url>
```

#### 2. 数据结构

确保数据按以下结构组织：

```
data/nuscenes/
├── maps/                   # 地图文件
├── samples/                # 关键帧数据
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   └── RADAR_FRONT/
├── sweeps/                 # 中间帧数据
│   ├── CAM_FRONT/
│   ├── ...
│   └── LIDAR_TOP/
├── v1.0-trainval/         # 标注元数据
│   ├── attribute.json
│   ├── category.json
│   ├── instance.json
│   ├── scene.json
│   ├── sample.json
│   └── ...
└── v1.0-test/             # 测试集元数据
```

#### 3. 预处理数据

```bash
# 返回项目根目录
cd ../..

# 创建数据信息文件
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval

# 生成BEV分割标注（用于语义地图任务）
python tools/create_bev_seg_gt.py \
    --dataroot ./data/nuscenes \
    --version v1.0-trainval \
    --out-dir ./data/nuscenes/bev_seg
```

#### 4. 验证数据

```bash
# 检查数据完整性
python tools/misc/browse_dataset.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    --show-interval 1
```

---

## ⚙️ 模型配置

### 1. 下载预训练权重

```bash
# 创建预训练模型目录
mkdir -p pretrained

# 下载 Swin Transformer 预训练权重（Camera backbone）
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth \
    -O pretrained/swin_tiny_patch4_window7_224.pth

# 下载 BEVFusion 预训练模型（可选，用于微调）
# 从 https://github.com/mit-han-lab/bevfusion 下载
```

### 2. 配置文件检查

```bash
# 检查配置文件
cat configs/nuscenes/det/bevfusion-e2e-perception.yaml

# 确保以下路径正确：
# - data_root: data/nuscenes/
# - pretrained weights路径
# - work_dir: 输出目录
```

### 3. 修改配置（如需要）

编辑 `configs/nuscenes/det/bevfusion-e2e-perception.yaml`:

```yaml
# 根据你的GPU调整
data:
  samples_per_gpu: 2  # Batch size per GPU
  workers_per_gpu: 4  # DataLoader workers

# 根据显存调整
model:
  heads:
    perception:
      num_query_det: 600  # 降低query数量节省显存
      num_query_seg: 50
```

---

## 🏋️ 训练模型

### 单GPU训练

```bash
python tools/train.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    --work-dir work_dirs/bevfusion_e2e_v1
```

### 多GPU训练（推荐）

```bash
# 4个GPU
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    4 \
    --work-dir work_dirs/bevfusion_e2e_v1

# 8个GPU
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    8 \
    --work-dir work_dirs/bevfusion_e2e_v1
```

### 从检查点恢复

```bash
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    4 \
    --work-dir work_dirs/bevfusion_e2e_v1 \
    --resume-from work_dirs/bevfusion_e2e_v1/latest.pth
```

### 使用预训练模型微调

```bash
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    4 \
    --work-dir work_dirs/bevfusion_e2e_finetune \
    --load-from pretrained/bevfusion_pretrained.pth
```

---

## 🔍 模型评估

### 完整评估

```bash
# 评估所有任务
python tools/test.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --eval bbox segm tracking
```

### 单任务评估

```bash
# 仅评估检测
python tools/test.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --eval bbox

# 仅评估分割
python tools/test.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --eval segm
```

---

## 🎨 可视化推理

### 运行端到端推理并可视化

```bash
python examples/run_e2e_perception.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --out-dir output/visualization \
    --visualize \
    --save-results \
    --num-samples 20
```

### 生成视频

```bash
python tools/visualize_video.py \
    --results output/visualization/results \
    --output output/video/perception.mp4 \
    --fps 10
```

---

## ✅ 验证安装

### 快速测试

创建测试脚本 `test_installation.py`:

```python
#!/usr/bin/env python3
import torch
import mmcv
import mmdet
import mmdet3d

print("=" * 60)
print("环境检查")
print("=" * 60)

# PyTorch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")

# MMCV
print(f"✓ MMCV: {mmcv.__version__}")

# MMDetection
print(f"✓ MMDetection: {mmdet.__version__}")

# MMDetection3D
print(f"✓ MMDetection3D: {mmdet3d.__version__}")

# 测试自定义算子
try:
    from mmdet3d.ops import Voxelization
    print("✓ Custom CUDA operators compiled successfully")
except:
    print("✗ Custom CUDA operators not available")

# 测试模型构建
try:
    from mmcv import Config
    from mmdet3d.models import build_model
    
    cfg = Config.fromfile('configs/nuscenes/det/bevfusion-e2e-perception.yaml')
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    print("✓ Model build successful")
    
    # 测试前向传播
    dummy_img = torch.randn(1, 6, 3, 256, 704).cuda()
    dummy_points = torch.randn(1, 10000, 5).cuda()
    print("✓ Dummy data created")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")

print("=" * 60)
print("安装验证完成！")
print("=" * 60)
```

运行测试：

```bash
python test_installation.py
```

### 最小示例测试

```bash
# 使用mini数据集快速测试
python tools/test.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --eval bbox \
    --show-dir output/test_vis \
    --cfg-options data.test.ann_file=data/nuscenes/nuscenes_infos_val_mini.pkl
```

---

## 🐳 Docker 部署（可选）

### 构建 Docker 镜像

创建 `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    git wget curl vim \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# 创建环境
RUN conda create -n bevfusion python=3.8 -y
SHELL ["conda", "run", "-n", "bevfusion", "/bin/bash", "-c"]

# 安装PyTorch
RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# 安装MMDetection3D相关
RUN pip install openmim && \
    mim install mmengine && \
    mim install "mmcv-full>=1.4.0,<1.7.0" && \
    mim install "mmdet>=2.24.0,<3.0.0" && \
    mim install "mmsegmentation>=0.20.0,<1.0.0"

# 复制项目
WORKDIR /workspace
COPY . /workspace/bevfusion

# 安装依赖
RUN cd bevfusion && \
    pip install -r requirements.txt && \
    cd mmdet3d/ops && python setup.py develop

# 设置工作目录
WORKDIR /workspace/bevfusion

CMD ["/bin/bash"]
```

构建和运行：

```bash
# 构建镜像
docker build -t bevfusion-e2e:latest .

# 运行容器
docker run --gpus all -it --rm \
    -v /path/to/data:/workspace/bevfusion/data \
    -v /path/to/output:/workspace/bevfusion/output \
    bevfusion-e2e:latest
```

---

## ❓ 常见问题

### Q1: CUDA out of memory

**解决方案**:
```yaml
# 降低batch size
data:
  samples_per_gpu: 1

# 降低query数量
model:
  heads:
    perception:
      num_query_det: 300
      num_query_seg: 30
```

### Q2: 编译CUDA扩展失败

**解决方案**:
```bash
# 确保CUDA版本匹配
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 清理重新编译
cd mmdet3d/ops
rm -rf build/
python setup.py clean
python setup.py develop
```

### Q3: 数据加载慢

**解决方案**:
```yaml
# 增加workers
data:
  workers_per_gpu: 8

# 使用SSD存储数据
# 启用预加载
data:
  train:
    dataset:
      pipeline:
        - type: LoadMultiViewImageFromFiles
          to_float32: true
          prefetch: true
```

### Q4: 训练不稳定

**解决方案**:
```yaml
# 降低学习率
optimizer:
  lr: 1.0e-4

# 增加warmup
lr_config:
  warmup_iters: 1000

# 使用梯度累积
optimizer_config:
  grad_clip:
    max_norm: 10
```

### Q5: 可视化结果不显示

**解决方案**:
```bash
# 安装显示依赖
pip install opencv-python-headless matplotlib

# 使用保存而非显示
python examples/run_e2e_perception.py ... --save-results
```

---

## 📝 检查清单

安装完成后，确保以下所有项都已完成：

- [ ] CUDA和PyTorch正确安装
- [ ] MMDetection3D及依赖安装
- [ ] CUDA扩展编译成功
- [ ] nuScenes数据集下载并组织
- [ ] 数据预处理完成
- [ ] 配置文件检查无误
- [ ] 模型可以成功构建
- [ ] 训练脚本可以运行
- [ ] 推理脚本可以运行
- [ ] 可视化结果正常生成

---

## 🎉 下一步

安装完成后，你可以：

1. **训练模型**: 按照训练指南开始训练
2. **运行推理**: 使用预训练模型进行推理
3. **可视化结果**: 生成多任务感知可视化
4. **性能优化**: 根据硬件调整配置

参考其他文档获取更多信息：
- [使用指南](docs/E2E_PERCEPTION_GUIDE.md)
- [部署方案](DEPLOYMENT_E2E.md)
- [可视化工具](tools/visualize_e2e.py)

---

**安装支持**: 如遇问题，请查看[常见问题](#常见问题)或提交Issue。
