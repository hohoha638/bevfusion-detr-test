# BEVFusion 端到端多任务感知 - 快速部署指南 🚀

> **一站式部署方案：从安装到可视化结果生成**

---

## 📚 文档导航

| 文档 | 说明 | 适用场景 |
|------|------|----------|
| 📘 [本指南](QUICK_DEPLOY_GUIDE.md) | 快速部署流程 | **首次使用必读** |
| 🔧 [安装指南](INSTALLATION_E2E.md) | 详细安装步骤 | 深入了解安装细节 |
| 🚀 [部署方案](DEPLOYMENT_E2E.md) | 生产部署方案 | 生产环境部署 |
| 📖 [完整文档](README_E2E.md) | 系统完整说明 | 全面了解系统 |

---

## ⚡ 快速开始（5分钟）

### 方式1: 自动安装脚本（推荐）

#### Linux/Mac:
```bash
# 1. 下载并运行快速启动脚本
chmod +x quick_start_e2e.sh
./quick_start_e2e.sh
```

#### Windows:
```batch
REM 1. 双击运行或在命令行执行
quick_start_e2e.bat
```

### 方式2: 手动安装

```bash
# 1. 创建环境
conda create -n bevfusion-e2e python=3.8 -y
conda activate bevfusion-e2e

# 2. 安装PyTorch
conda install pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch

# 3. 安装MMDetection3D
pip install openmim
mim install mmengine mmcv-full mmdet mmsegmentation

# 4. 安装依赖
pip install -r requirements.txt
pip install scipy matplotlib opencv-python imageio tqdm

# 5. 编译CUDA算子
cd mmdet3d/ops && python setup.py develop && cd ../..

# 6. 验证安装
python -c "import mmdet3d; print('✓ 安装成功')"
```

---

## 📊 完整部署流程

### 第一步：环境准备 ⚙️

#### 系统要求
- **GPU**: NVIDIA RTX 2080Ti+ (推荐RTX 3090)
- **内存**: 32GB+ RAM
- **存储**: 200GB+ SSD
- **系统**: Ubuntu 18.04+ / Windows 10+

#### 软件要求
```bash
# 检查版本
python --version    # 3.8+
nvcc --version     # CUDA 11.1/11.3
nvidia-smi         # 显示GPU信息
```

### 第二步：数据准备 📦

#### nuScenes数据集

```bash
# 1. 创建数据目录
mkdir -p data/nuscenes

# 2. 下载数据（从 https://www.nuscenes.org/nuscenes）
# - Full dataset (v1.0-trainval): ~350GB
# - Mini dataset (v1.0-mini): ~5GB (用于快速测试)

# 3. 组织数据结构
data/nuscenes/
├── maps/
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_RIGHT/
│   ├── ...
│   └── LIDAR_TOP/
├── sweeps/
└── v1.0-trainval/

# 4. 预处理数据
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes

# 5. 生成BEV分割标注
python tools/create_bev_seg_gt.py \
    --dataroot ./data/nuscenes \
    --out-dir ./data/nuscenes/bev_seg
```

### 第三步：模型配置 🔧

#### 下载预训练权重

```bash
# 创建目录
mkdir -p pretrained

# 下载Swin Transformer权重
wget -O pretrained/swin_tiny_patch4_window7_224.pth \
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

#### 配置文件说明

配置文件位置: `configs/nuscenes/det/bevfusion-e2e-perception.yaml`

关键参数：
```yaml
# 根据GPU显存调整
data:
  samples_per_gpu: 2      # Batch size (24GB GPU: 2, 12GB GPU: 1)
  workers_per_gpu: 4

# 根据任务需求调整
model:
  task_weights:
    detection: 1.0        # 检测任务权重
    segmentation: 1.0     # 分割任务权重
    tracking: 0.5         # 跟踪任务权重
  
  heads:
    perception:
      num_query_det: 900  # 检测query数量
      num_query_seg: 100  # 分割query数量
```

### 第四步：训练模型 🏋️

#### 单GPU训练
```bash
python tools/train.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    --work-dir work_dirs/bevfusion_e2e_v1
```

#### 多GPU训练（推荐）
```bash
# 4个GPU
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    4

# 8个GPU
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    8
```

#### 监控训练
```bash
# TensorBoard
tensorboard --logdir work_dirs/bevfusion_e2e_v1

# 查看日志
tail -f work_dirs/bevfusion_e2e_v1/$(date +%Y%m%d_%H%M%S).log
```

### 第五步：运行推理 🔍

#### 基础推理

```bash
python examples/run_e2e_perception.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --out-dir output/results \
    --num-samples 20
```

#### 推理 + 可视化

```bash
python examples/run_e2e_perception.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --out-dir output/results \
    --visualize \
    --save-results \
    --num-samples 20
```

#### 批量推理

```bash
# 测试集推理
python tools/test.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --eval bbox segm tracking
```

### 第六步：生成可视化 🎨

#### 单个结果可视化

```bash
python tools/visualize_e2e.py \
    --results output/results/result_0000.npz \
    --output output/visualization
```

#### 批量可视化

```bash
python tools/visualize_e2e.py \
    --results output/results \
    --output output/visualization \
    --num-samples 50
```

#### 生成视频

```bash
python tools/visualize_e2e.py \
    --results output/results \
    --output output/visualization \
    --video \
    --fps 10
```

输出文件：
- `output/visualization/combined/` - 单帧可视化
- `output/visualization/perception_video.mp4` - 视频
- `output/visualization/perception_video.gif` - GIF动图

---

## 🎯 可视化结果说明

### 生成的可视化包含四个子图：

1. **左上 - 3D检测 (BEV视图)**
   - 蓝色矩形：自车
   - 彩色多边形：检测到的目标
   - 箭头：目标朝向
   - 标签：类别 + 置信度

2. **右上 - 语义地图**
   - 紫色：可行驶区域
   - 品红：车道线
   - 灰色：人行道
   - 蓝灰：其他

3. **左下 - 多目标跟踪**
   - 彩色框：不同目标
   - ID标签：跟踪ID
   - 颜色一致：同一目标

4. **右下 - 轨迹预测**
   - 实心圆：当前位置
   - 虚线：预测轨迹
   - 空心圆：未来位置点
   - t+N标签：时间步

### 可视化示例

```
┌──────────────────────┬──────────────────────┐
│   3D检测(BEV视图)     │    语义地图分割       │
│                      │                      │
│  [车辆、行人等检测框] │ [道路、车道线等分割]  │
│                      │                      │
├──────────────────────┼──────────────────────┤
│   多目标跟踪          │    轨迹预测          │
│                      │                      │
│ [带ID的跟踪框]        │ [未来运动轨迹]        │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

---

## 📈 性能优化建议

### 显存不足？

```yaml
# 方案1: 减小batch size
data:
  samples_per_gpu: 1

# 方案2: 减少query数量
model:
  heads:
    perception:
      num_query_det: 600  # 从900减少
      num_query_seg: 50   # 从100减少

# 方案3: 禁用某些任务
model:
  task_weights:
    tracking: 0.0  # 禁用跟踪
```

### 推理速度慢？

```bash
# 方案1: TensorRT加速
python tools/deployment/export_onnx.py config.yaml checkpoint.pth
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# 方案2: 量化
python tools/deployment/quantize_model.py \
    --config config.yaml \
    --checkpoint checkpoint.pth \
    --output model_int8.pth

# 方案3: 混合精度
# 在推理时使用 --fp16 标志
```

---

## 🐛 常见问题快速解决

### Q1: CUDA编译失败
```bash
# 清理重编译
cd mmdet3d/ops
rm -rf build/
python setup.py clean
python setup.py develop
```

### Q2: 内存溢出
```bash
# 增加系统swap
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Q3: 数据加载慢
```yaml
# 增加workers
data:
  workers_per_gpu: 8
  persistent_workers: true
```

### Q4: 可视化不显示
```bash
# 安装无头版OpenCV
pip uninstall opencv-python
pip install opencv-python-headless

# 使用保存而非显示
python tools/visualize_e2e.py --save-only
```

---

## 📊 性能基准测试

运行基准测试：

```bash
python examples/run_e2e_perception.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    checkpoint.pth \
    --benchmark \
    --num-samples 100
```

预期性能（RTX 3090）：

| 指标 | 值 |
|------|-----|
| 检测 NDS | 0.708 |
| 检测 mAP | 0.680 |
| 分割 mIoU | 0.652 |
| 跟踪 MOTA | 0.534 |
| 推理速度 | 15 FPS |
| GPU显存 | ~14GB |

---

## 🎓 完整使用示例

### 示例1: 端到端训练和评估

```bash
#!/bin/bash

# 1. 训练模型
bash tools/dist_train.sh \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    4 \
    --work-dir work_dirs/my_experiment

# 2. 评估模型
python tools/test.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/my_experiment/latest.pth \
    --eval bbox segm tracking \
    --out results.pkl

# 3. 可视化结果
python tools/visualize_e2e.py \
    --results results.pkl \
    --output vis_output \
    --video
```

### 示例2: 在线推理服务

```bash
# 启动推理服务
python tools/inference_server.py \
    --config configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    --checkpoint work_dirs/my_experiment/latest.pth \
    --port 8080

# 客户端请求
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d @sample_data.json
```

### 示例3: ROS节点部署

```bash
# 启动ROS节点
rosrun bevfusion bevfusion_e2e_node.py \
    _model_path:=checkpoint.pth \
    _config_path:=config.yaml

# 查看话题
rostopic list
# /bevfusion/detections
# /bevfusion/segmentation
# /bevfusion/tracking
```

---

## 📦 Docker快速部署

```bash
# 1. 构建镜像
docker build -t bevfusion-e2e:latest .

# 2. 运行容器
docker run --gpus all -it --rm \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/output:/workspace/output \
    -p 8080:8080 \
    bevfusion-e2e:latest

# 3. 在容器内运行
python examples/run_e2e_perception.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    checkpoint.pth \
    --visualize
```

---

## 🎯 检查清单

部署完成后，确保以下所有项都已完成：

- [ ] **环境安装**
  - [ ] Python 3.8+ 安装
  - [ ] CUDA 11.1/11.3 安装
  - [ ] PyTorch 1.10+ 安装
  - [ ] MMDetection3D安装
  - [ ] CUDA算子编译成功

- [ ] **数据准备**
  - [ ] nuScenes数据下载
  - [ ] 数据结构正确
  - [ ] 数据预处理完成
  - [ ] BEV分割标注生成

- [ ] **模型训练**
  - [ ] 配置文件检查
  - [ ] 预训练权重下载
  - [ ] 训练正常运行
  - [ ] 模型收敛

- [ ] **推理验证**
  - [ ] 推理脚本运行成功
  - [ ] 输出结果正确
  - [ ] 性能达标

- [ ] **可视化**
  - [ ] 单帧可视化正常
  - [ ] 视频生成成功
  - [ ] 结果符合预期

---

## 🚀 下一步行动

1. **开始训练**: 运行快速启动脚本，开始第一次训练
2. **查看结果**: 使用可视化工具查看多任务输出
3. **优化模型**: 根据结果调整配置参数
4. **部署应用**: 将模型部署到生产环境

---

## 📞 获取帮助

- **文档**: 查看[完整文档](README_E2E.md)
- **示例**: 运行`examples/`目录下的示例代码
- **问题**: 提交GitHub Issue
- **讨论**: 加入社区讨论

---

## 🎉 恭喜！

如果你已经完成以上步骤，说明你已经成功部署了**完整的端到端多任务感知系统**！

现在你可以：
- ✅ 检测3D目标
- ✅ 生成语义地图
- ✅ 跟踪多个目标
- ✅ 预测运动轨迹
- ✅ 可视化所有结果

**祝你使用愉快！** 🚗💨

---

**最后更新**: 2025年11月  
**维护者**: BEVFusion Team  
**许可证**: Apache 2.0
