"""
End-to-End Multi-Task Perception Demo
演示完整的端到端多任务感知：检测 + 分割 + 跟踪
"""
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='E2E Multi-Task Perception')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型checkpoint路径')
    parser.add_argument('--out-dir', default='output/e2e_perception', help='输出目录')
    parser.add_argument('--device', default='cuda:0', help='使用的设备')
    parser.add_argument('--visualize', action='store_true', help='是否可视化')
    parser.add_argument('--save-results', action='store_true', help='是否保存结果')
    parser.add_argument('--num-samples', type=int, default=10, help='处理样本数')
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device='cuda:0'):
    """加载模型"""
    print("🔄 加载模型...")
    cfg = Config.fromfile(config_path)
    
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    return model, cfg


def run_inference(model, data, device='cuda:0'):
    """运行推理"""
    with torch.no_grad():
        # 移动数据到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        # 前向传播
        results = model(**data)
    
    return results


def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("📊 感知结果摘要")
    print("="*60)
    
    for idx, result in enumerate(results):
        print(f"\n样本 {idx+1}:")
        
        # 1. 检测统计
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            scores = result['scores_3d']
            labels = result['labels_3d']
            
            print(f"  🎯 检测: {len(boxes)} 个目标")
            if len(boxes) > 0:
                print(f"     平均置信度: {scores.mean():.3f}")
                print(f"     类别分布: {np.bincount(labels.numpy())}")
        
        # 2. 分割统计
        if 'seg_mask' in result:
            seg_mask = result['seg_mask']
            print(f"  🗺️  分割: {seg_mask.shape[0]} 个类别")
            for cls_id in range(seg_mask.shape[0]):
                mask = seg_mask[cls_id] > 0.5
                ratio = mask.float().mean().item()
                print(f"     类别{cls_id}: {ratio*100:.1f}% 像素")
        
        # 3. 跟踪统计
        if 'track_ids' in result and result['track_ids'] is not None:
            track_ids = result['track_ids']
            unique_ids = torch.unique(track_ids)
            print(f"  🎬 跟踪: {len(unique_ids)} 个轨迹")
        
        # 4. 轨迹预测
        if 'trajectories' in result and result['trajectories'] is not None:
            trajs = result['trajectories']
            print(f"  🔮 轨迹: 预测未来 {trajs.shape[1]} 帧")
    
    print("="*60 + "\n")


def save_results(results, save_dir):
    """保存结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, result in enumerate(results):
        save_path = os.path.join(save_dir, f'result_{idx:04d}.npz')
        
        # 转换为numpy
        result_np = {}
        for key, value in result.items():
            if value is not None:
                if torch.is_tensor(value):
                    result_np[key] = value.cpu().numpy()
                else:
                    result_np[key] = value
        
        np.savez(save_path, **result_np)
    
    print(f"✅ 结果已保存到: {save_dir}")


def visualize_results(model, results, save_dir=None):
    """可视化结果"""
    print("\n🎨 生成可视化...")
    model.visualize_predictions(results, save_dir)
    print(f"✅ 可视化完成")


def analyze_perception_quality(results):
    """分析感知质量"""
    print("\n" + "="*60)
    print("📈 感知质量分析")
    print("="*60)
    
    # 统计信息
    total_detections = 0
    high_conf_detections = 0
    total_seg_pixels = 0
    total_tracks = set()
    
    for result in results:
        # 检测质量
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            scores = result['scores_3d']
            total_detections += len(boxes)
            high_conf_detections += (scores > 0.5).sum().item()
        
        # 分割质量
        if 'seg_mask' in result:
            seg_mask = result['seg_mask']
            total_seg_pixels += (seg_mask > 0.5).sum().item()
        
        # 跟踪质量
        if 'track_ids' in result and result['track_ids'] is not None:
            track_ids = result['track_ids'].cpu().numpy()
            total_tracks.update(track_ids)
    
    # 打印统计
    print(f"\n检测质量:")
    print(f"  总检测数: {total_detections}")
    print(f"  高置信度检测 (>0.5): {high_conf_detections}")
    print(f"  高置信度比例: {high_conf_detections/max(total_detections,1)*100:.1f}%")
    
    print(f"\n分割质量:")
    print(f"  分割像素总数: {total_seg_pixels}")
    
    print(f"\n跟踪质量:")
    print(f"  唯一轨迹数: {len(total_tracks)}")
    
    print("="*60 + "\n")


def benchmark_performance(model, data_loader, device, num_samples=10):
    """性能基准测试"""
    print("\n" + "="*60)
    print("⚡ 性能基准测试")
    print("="*60)
    
    import time
    
    model.eval()
    
    # 预热
    print("预热中...")
    for i, data in enumerate(data_loader):
        if i >= 3:
            break
        _ = run_inference(model, data, device)
    
    # 测试
    print("测试中...")
    times = []
    
    for i, data in enumerate(data_loader):
        if i >= num_samples:
            break
        
        torch.cuda.synchronize()
        start = time.time()
        
        _ = run_inference(model, data, device)
        
        torch.cuda.synchronize()
        end = time.time()
        
        times.append(end - start)
        print(f"  样本 {i+1}/{num_samples}: {times[-1]*1000:.2f} ms")
    
    # 统计
    times = np.array(times)
    print(f"\n性能统计:")
    print(f"  平均时间: {times.mean()*1000:.2f} ms")
    print(f"  中位数: {np.median(times)*1000:.2f} ms")
    print(f"  标准差: {times.std()*1000:.2f} ms")
    print(f"  FPS: {1.0/times.mean():.2f}")
    
    print("="*60 + "\n")


def extract_multi_task_features(model, data, device='cuda:0'):
    """提取多任务特征"""
    print("\n🔍 提取多任务特征...")
    
    with torch.no_grad():
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        features = model.extract_multi_task_features(**data)
    
    print("✅ 特征提取完成")
    print(f"\n特征信息:")
    for key, value in features.items():
        if torch.is_tensor(value):
            print(f"  {key:30s}: {list(value.shape)}")
    
    return features


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*60)
    print("🚀 BEVFusion 端到端多任务感知系统")
    print("="*60)
    
    # 1. 加载模型
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    
    # 2. 准备数据
    print("\n📦 准备数据...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    print(f"✅ 数据集: {len(dataset)} 个样本")
    
    # 3. 运行推理
    print(f"\n⚙️  运行推理 (处理 {args.num_samples} 个样本)...")
    all_results = []
    
    for i, data in enumerate(data_loader):
        if i >= args.num_samples:
            break
        
        print(f"  处理样本 {i+1}/{args.num_samples}...")
        results = run_inference(model, data, args.device)
        all_results.extend(results)
    
    print("✅ 推理完成")
    
    # 4. 打印结果摘要
    print_results_summary(all_results)
    
    # 5. 分析感知质量
    analyze_perception_quality(all_results)
    
    # 6. 保存结果
    if args.save_results:
        results_dir = os.path.join(args.out_dir, 'results')
        save_results(all_results, results_dir)
    
    # 7. 可视化
    if args.visualize:
        vis_dir = os.path.join(args.out_dir, 'visualizations')
        visualize_results(model, all_results, vis_dir)
    
    # 8. 性能测试
    print("\n是否运行性能基准测试? (y/n): ", end='')
    if input().lower() == 'y':
        benchmark_performance(model, data_loader, args.device, min(args.num_samples, 10))
    
    # 9. 特征提取示例
    print("\n是否提取多任务特征? (y/n): ", end='')
    if input().lower() == 'y':
        data = next(iter(data_loader))
        features = extract_multi_task_features(model, data, args.device)
        
        # 保存特征
        if args.save_results:
            feat_dir = os.path.join(args.out_dir, 'features')
            os.makedirs(feat_dir, exist_ok=True)
            
            features_np = {}
            for key, value in features.items():
                if torch.is_tensor(value) and value is not None:
                    features_np[key] = value.cpu().numpy()
            
            np.savez(os.path.join(feat_dir, 'multi_task_features.npz'), **features_np)
            print(f"✅ 特征已保存")
    
    print("\n" + "="*60)
    print("✅ 所有任务完成！")
    print(f"📁 输出目录: {args.out_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
