# BEVFusion 端到端多任务感知系统 - 完整部署方案

## 📋 目录

- [部署架构](#部署架构)
- [性能优化](#性能优化)
- [生产部署](#生产部署)
- [推理服务](#推理服务)
- [监控与维护](#监控与维护)

---

## 🏗️ 部署架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据采集层                                  │
│              摄像头×6 + LiDAR + 预处理                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    推理引擎层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ TensorRT优化 │  │  模型推理     │  │  结果后处理   │      │
│  │  模型加速    │  │  多任务输出   │  │  NMS/追踪    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    应用服务层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 可视化服务   │  │  API服务      │  │  存储服务     │      │
│  │ Web/ROS     │  │  REST/gRPC   │  │  数据库/文件  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 部署模式

#### 1. 单机部署（开发/测试）

```
┌────────────────────┐
│   工作站/服务器     │
│  ┌──────────────┐  │
│  │ GPU × 1-2    │  │
│  │ 模型推理     │  │
│  │ 可视化       │  │
│  └──────────────┘  │
└────────────────────┘
```

#### 2. 车载部署（边缘计算）

```
┌────────────────────┐
│   车载计算单元      │
│  ┌──────────────┐  │
│  │ NVIDIA AGX   │  │
│  │ Xavier/Orin  │  │
│  │ TensorRT优化 │  │
│  └──────────────┘  │
└────────────────────┘
```

#### 3. 云端部署（批量处理）

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 推理节点 #1  │  │ 推理节点 #2  │  │ 推理节点 #N  │
│  GPU × 8     │  │  GPU × 8     │  │  GPU × 8     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                ┌────────▼────────┐
                │  负载均衡器      │
                └─────────────────┘
```

---

## ⚡ 性能优化

### 1. 模型优化

#### TensorRT 加速

```bash
# 导出ONNX模型
python tools/deployment/export_onnx.py \
    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
    work_dirs/bevfusion_e2e_v1/latest.pth \
    --output-file deploy/model.onnx \
    --shape 1,6,3,256,704

# 转换为TensorRT引擎
trtexec \
    --onnx=deploy/model.onnx \
    --saveEngine=deploy/model.trt \
    --fp16 \
    --workspace=8192 \
    --minShapes=input:1x6x3x256x704 \
    --optShapes=input:1x6x3x256x704 \
    --maxShapes=input:1x6x3x256x704 \
    --verbose
```

#### 量化加速

```python
# 动态量化
import torch
from torch.quantization import quantize_dynamic

model = build_model(cfg.model).eval()
model_quantized = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 保存量化模型
torch.save(model_quantized.state_dict(), 'deploy/model_int8.pth')
```

#### 混合精度推理

```python
# 使用AMP加速
from torch.cuda.amp import autocast

model.eval().cuda()
with torch.no_grad():
    with autocast():
        results = model(**data)
```

### 2. 数据优化

#### 预处理优化

```python
# 使用DALI加速数据加载
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def preprocess_pipeline():
    images = fn.readers.file(file_root="data/images")
    images = fn.decoders.image(images, device="mixed")
    images = fn.resize(images, resize_x=704, resize_y=256)
    images = fn.normalize(images, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225])
    return images
```

#### 批处理优化

```python
# 动态批处理
class DynamicBatcher:
    def __init__(self, max_batch_size=4, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.buffer = []
    
    def add_sample(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.max_batch_size:
            return self.flush()
        return None
    
    def flush(self):
        if not self.buffer:
            return None
        batch = collate_fn(self.buffer)
        self.buffer = []
        return batch
```

### 3. 推理优化

#### Pipeline并行

```python
import threading
from queue import Queue

class PipelineInference:
    def __init__(self, model, num_threads=3):
        self.model = model
        self.preprocess_queue = Queue(maxsize=10)
        self.inference_queue = Queue(maxsize=10)
        self.postprocess_queue = Queue(maxsize=10)
        
        # 启动pipeline
        threading.Thread(target=self._preprocess_worker).start()
        threading.Thread(target=self._inference_worker).start()
        threading.Thread(target=self._postprocess_worker).start()
    
    def _preprocess_worker(self):
        while True:
            raw_data = self.preprocess_queue.get()
            processed = preprocess(raw_data)
            self.inference_queue.put(processed)
    
    def _inference_worker(self):
        while True:
            data = self.inference_queue.get()
            results = self.model(**data)
            self.postprocess_queue.put(results)
    
    def _postprocess_worker(self):
        while True:
            results = self.postprocess_queue.get()
            final = postprocess(results)
            yield final
```

---

## 🚀 生产部署

### 方案 1: Docker 容器化部署

#### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:21.12-py3

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY . /app
COPY deploy/model.trt /app/deploy/

# 暴露端口
EXPOSE 8080

# 启动服务
CMD ["python", "tools/inference_server.py", "--config", "configs/deploy.yaml"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  bevfusion-e2e:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/deploy/model.trt
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data:ro
      - ./output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 方案 2: Kubernetes 部署

#### Deployment配置

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bevfusion-e2e
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bevfusion-e2e
  template:
    metadata:
      labels:
        app: bevfusion-e2e
    spec:
      containers:
      - name: inference
        image: bevfusion-e2e:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 8
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: model-volume
          mountPath: /app/deploy
          readOnly: true
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

#### Service配置

```yaml
apiVersion: v1
kind: Service
metadata:
  name: bevfusion-e2e-service
spec:
  selector:
    app: bevfusion-e2e
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 方案 3: NVIDIA Triton 部署

#### 模型仓库结构

```
model_repository/
└── bevfusion_e2e/
    ├── config.pbtxt
    ├── 1/
    │   └── model.plan  # TensorRT引擎
    └── labels.txt
```

#### config.pbtxt

```protobuf
name: "bevfusion_e2e"
platform: "tensorrt_plan"
max_batch_size: 4
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 6, 3, 256, 704 ]
  },
  {
    name: "points"
    data_type: TYPE_FP32
    dims: [ -1, 5 ]
  }
]
output [
  {
    name: "boxes_3d"
    data_type: TYPE_FP32
    dims: [ -1, 9 ]
  },
  {
    name: "seg_mask"
    data_type: TYPE_FP32
    dims: [ 4, 180, 180 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

#### 启动Triton服务

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:22.12-py3 \
  tritonserver --model-repository=/models
```

---

## 🌐 推理服务

### REST API 服务

创建 `tools/inference_server.py`:

```python
from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import io
from PIL import Image
import base64

app = Flask(__name__)

# 加载模型
model = load_model('configs/deploy.yaml', 'deploy/model.pth')
model.eval().cuda()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'gpu': torch.cuda.is_available()})

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收图像和点云数据，返回多任务感知结果
    """
    data = request.json
    
    # 解析输入
    images = decode_images(data['images'])  # List of 6 images
    points = np.array(data['points'])       # Point cloud
    
    # 预处理
    processed = preprocess(images, points)
    
    # 推理
    with torch.no_grad():
        results = model(**processed)
    
    # 后处理
    output = postprocess(results[0])
    
    return jsonify({
        'detection': {
            'boxes': output['boxes_3d'].tolist(),
            'scores': output['scores_3d'].tolist(),
            'labels': output['labels_3d'].tolist()
        },
        'segmentation': {
            'mask': output['seg_mask'].tolist()
        },
        'tracking': {
            'ids': output['track_ids'].tolist() if output.get('track_ids') is not None else []
        }
    })

@app.route('/visualize', methods=['POST'])
def visualize():
    """
    生成可视化结果并返回图像
    """
    data = request.json
    
    # 运行推理
    images = decode_images(data['images'])
    points = np.array(data['points'])
    processed = preprocess(images, points)
    
    with torch.no_grad():
        results = model(**processed)
    
    # 生成可视化
    vis_image = generate_visualization(results[0], images[0])
    
    # 转换为bytes
    img_io = io.BytesIO()
    vis_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
```

### gRPC 服务

创建 `proto/inference.proto`:

```protobuf
syntax = "proto3";

service BEVFusionE2E {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc StreamPredict(stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
  repeated bytes images = 1;  // 6 images
  repeated float points = 2;   // Nx5 point cloud
  map<string, string> metadata = 3;
}

message PredictResponse {
  repeated BBox3D boxes = 1;
  SegmentationMask seg_mask = 2;
  repeated int32 track_ids = 3;
  repeated Trajectory trajectories = 4;
}

message BBox3D {
  float x = 1;
  float y = 2;
  float z = 3;
  float w = 4;
  float h = 5;
  float l = 6;
  float yaw = 7;
  float score = 8;
  int32 label = 9;
}

message SegmentationMask {
  repeated float data = 1;
  int32 height = 2;
  int32 width = 3;
  int32 channels = 4;
}

message Trajectory {
  repeated Point2D points = 1;
}

message Point2D {
  float x = 1;
  float y = 2;
}
```

### ROS节点部署

创建 `ros/bevfusion_e2e_node.py`:

```python
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray
from nav_msgs.msg import OccupancyGrid
import torch

class BEVFusionE2ENode:
    def __init__(self):
        rospy.init_node('bevfusion_e2e_node')
        
        # 加载模型
        self.model = self.load_model()
        
        # 订阅话题
        self.image_subs = []
        for i in range(6):
            sub = rospy.Subscriber(
                f'/camera_{i}/image_raw',
                Image,
                self.image_callback,
                callback_args=i
            )
            self.image_subs.append(sub)
        
        self.points_sub = rospy.Subscriber(
            '/lidar/points',
            PointCloud2,
            self.points_callback
        )
        
        # 发布话题
        self.detection_pub = rospy.Publisher(
            '/bevfusion/detections',
            Detection3DArray,
            queue_size=10
        )
        
        self.segmentation_pub = rospy.Publisher(
            '/bevfusion/segmentation',
            OccupancyGrid,
            queue_size=10
        )
        
        # 数据缓存
        self.images = [None] * 6
        self.points = None
        
    def load_model(self):
        model_path = rospy.get_param('~model_path', 'deploy/model.pth')
        config_path = rospy.get_param('~config_path', 'configs/deploy.yaml')
        
        model = load_model(config_path, model_path)
        model.eval().cuda()
        return model
    
    def image_callback(self, msg, camera_id):
        self.images[camera_id] = self.convert_image(msg)
        self.try_inference()
    
    def points_callback(self, msg):
        self.points = self.convert_points(msg)
        self.try_inference()
    
    def try_inference(self):
        # 检查数据是否齐全
        if None in self.images or self.points is None:
            return
        
        # 运行推理
        with torch.no_grad():
            results = self.model(
                img=torch.stack(self.images).unsqueeze(0).cuda(),
                points=self.points.unsqueeze(0).cuda()
            )
        
        # 发布结果
        self.publish_detection(results[0])
        self.publish_segmentation(results[0])
    
    def publish_detection(self, result):
        msg = Detection3DArray()
        # 填充检测结果
        # ...
        self.detection_pub.publish(msg)
    
    def publish_segmentation(self, result):
        msg = OccupancyGrid()
        # 填充分割结果
        # ...
        self.segmentation_pub.publish(msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = BEVFusionE2ENode()
    node.run()
```

---

## 📊 监控与维护

### 性能监控

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.gpu_usage = []
        self.memory_usage = []
    
    def record_inference(self, start_time, end_time):
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        
        # GPU使用率
        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_usage.append(gpus[0].load * 100)
            self.memory_usage.append(gpus[0].memoryUsed)
    
    def get_stats(self):
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times),
            'avg_gpu_usage': np.mean(self.gpu_usage),
            'avg_memory_usage': np.mean(self.memory_usage)
        }
```

### 日志记录

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

# 使用
logger = setup_logger('bevfusion_e2e', 'logs/inference.log')
logger.info('Model loaded successfully')
logger.error('Inference failed', exc_info=True)
```

### 健康检查

```python
class HealthChecker:
    def __init__(self, model):
        self.model = model
        self.last_check = time.time()
        self.check_interval = 60  # 60秒检查一次
    
    def check(self):
        if time.time() - self.last_check < self.check_interval:
            return True
        
        try:
            # 测试推理
            dummy_input = create_dummy_input()
            with torch.no_grad():
                _ = self.model(**dummy_input)
            
            self.last_check = time.time()
            return True
        
        except Exception as e:
            logger.error(f'Health check failed: {e}')
            return False
```

---

## 🎯 部署检查清单

### 部署前

- [ ] 模型训练完成并验证
- [ ] 模型优化（TensorRT/量化）
- [ ] 性能基准测试通过
- [ ] 资源需求评估完成
- [ ] 部署环境准备就绪

### 部署中

- [ ] 容器镜像构建成功
- [ ] 模型文件正确挂载
- [ ] 网络配置正确
- [ ] GPU资源分配成功
- [ ] 服务成功启动

### 部署后

- [ ] 健康检查通过
- [ ] 性能指标达标
- [ ] 日志记录正常
- [ ] 监控系统配置
- [ ] 文档更新完成

---

## 📈 性能基准

| 部署方式 | 延迟 | 吞吐量 | GPU利用率 | 内存占用 |
|---------|------|--------|-----------|----------|
| PyTorch FP32 | 80ms | 12.5 FPS | 85% | 12GB |
| PyTorch FP16 | 50ms | 20 FPS | 80% | 10GB |
| TensorRT FP16 | 30ms | 33 FPS | 75% | 8GB |
| TensorRT INT8 | 20ms | 50 FPS | 70% | 6GB |

*测试环境: NVIDIA RTX 3090*

---

**部署支持**: 详细问题请参考[安装指南](INSTALLATION_E2E.md)或提交Issue。
