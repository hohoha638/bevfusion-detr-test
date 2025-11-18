"""
生成演示结果数据

用于在没有真实模型的情况下测试可视化系统
"""
import numpy as np
import argparse
from pathlib import Path
import torch


def generate_demo_detection(num_objects=15):
    """生成演示检测结果"""
    # 随机生成检测框
    boxes_3d = []
    scores_3d = []
    labels_3d = []
    
    for _ in range(num_objects):
        # 位置 (x, y, z)
        x = np.random.uniform(-40, 40)
        y = np.random.uniform(-40, 40)
        z = np.random.uniform(-1, 1)
        
        # 尺寸 (w, h, l)
        w = np.random.uniform(1.5, 2.5)
        h = np.random.uniform(1.3, 1.8)
        l = np.random.uniform(3.5, 5.0)
        
        # 朝向 (yaw)
        yaw = np.random.uniform(-np.pi, np.pi)
        
        # 速度 (vx, vy)
        vx = np.random.uniform(-5, 5)
        vy = np.random.uniform(-5, 5)
        
        box = [x, y, z, w, h, l, yaw, vx, vy]
        boxes_3d.append(box)
        
        # 置信度
        score = np.random.uniform(0.5, 0.99)
        scores_3d.append(score)
        
        # 类别 (0-9)
        label = np.random.randint(0, 10)
        labels_3d.append(label)
    
    return {
        'boxes_3d': torch.tensor(boxes_3d, dtype=torch.float32),
        'scores_3d': torch.tensor(scores_3d, dtype=torch.float32),
        'labels_3d': torch.tensor(labels_3d, dtype=torch.long)
    }


def generate_demo_segmentation():
    """生成演示分割结果"""
    H, W = 180, 180
    num_classes = 4
    
    seg_mask = torch.zeros(num_classes, H, W, dtype=torch.float32)
    
    # 生成圆形区域（模拟道路）
    center_x, center_y = W // 2, H // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # 可行驶区域（大圆）
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    seg_mask[0] = (dist < 70).float()
    
    # 车道线（环形）
    seg_mask[1] = ((dist > 30) & (dist < 35)).float() | ((dist > 60) & (dist < 65)).float()
    
    # 人行道（边缘）
    seg_mask[2] = ((dist > 70) & (dist < 80)).float()
    
    # 其他区域
    seg_mask[3] = (dist >= 80).float()
    
    return seg_mask


def generate_demo_tracking(num_objects=15):
    """生成演示跟踪结果"""
    # 跟踪ID
    track_ids = torch.arange(num_objects, dtype=torch.long)
    
    return track_ids


def generate_demo_trajectories(num_objects=15, future_steps=6):
    """生成演示轨迹预测"""
    trajectories = []
    
    for _ in range(num_objects):
        # 生成平滑轨迹
        traj = []
        vx = np.random.uniform(-2, 2)
        vy = np.random.uniform(-2, 2)
        
        for t in range(future_steps):
            # 带有一点随机扰动的直线运动
            dx = vx * (t + 1) + np.random.normal(0, 0.5)
            dy = vy * (t + 1) + np.random.normal(0, 0.5)
            traj.append([dx, dy])
        
        trajectories.append(traj)
    
    return torch.tensor(trajectories, dtype=torch.float32)


def generate_demo_result(num_objects=15):
    """生成完整的演示结果"""
    result = {}
    
    # 检测结果
    detection = generate_demo_detection(num_objects)
    result.update(detection)
    
    # 分割结果
    result['seg_mask'] = generate_demo_segmentation()
    
    # 跟踪结果
    result['track_ids'] = generate_demo_tracking(num_objects)
    
    # 轨迹预测
    result['trajectories'] = generate_demo_trajectories(num_objects)
    
    return result


def generate_sequence(num_frames=20, num_objects=15, output_dir='output/demo_results'):
    """生成连续帧序列"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成 {num_frames} 帧演示数据...")
    
    # 初始化目标状态
    objects = []
    for i in range(num_objects):
        obj = {
            'id': i,
            'x': np.random.uniform(-40, 40),
            'y': np.random.uniform(-40, 40),
            'z': np.random.uniform(-1, 1),
            'w': np.random.uniform(1.5, 2.5),
            'h': np.random.uniform(1.3, 1.8),
            'l': np.random.uniform(3.5, 5.0),
            'yaw': np.random.uniform(-np.pi, np.pi),
            'vx': np.random.uniform(-3, 3),
            'vy': np.random.uniform(-3, 3),
            'label': np.random.randint(0, 10)
        }
        objects.append(obj)
    
    for frame_id in range(num_frames):
        # 更新目标状态
        boxes_3d = []
        scores_3d = []
        labels_3d = []
        track_ids = []
        trajectories = []
        
        for obj in objects:
            # 更新位置（简单运动模型）
            obj['x'] += obj['vx'] * 0.1
            obj['y'] += obj['vy'] * 0.1
            
            # 边界检查
            if abs(obj['x']) > 50 or abs(obj['y']) > 50:
                # 重置到随机位置
                obj['x'] = np.random.uniform(-40, 40)
                obj['y'] = np.random.uniform(-40, 40)
            
            # 添加检测结果
            box = [obj['x'], obj['y'], obj['z'], obj['w'], obj['h'], 
                   obj['l'], obj['yaw'], obj['vx'], obj['vy']]
            boxes_3d.append(box)
            scores_3d.append(np.random.uniform(0.7, 0.99))
            labels_3d.append(obj['label'])
            track_ids.append(obj['id'])
            
            # 生成轨迹
            traj = []
            for t in range(6):
                dx = obj['vx'] * (t + 1) * 0.1
                dy = obj['vy'] * (t + 1) * 0.1
                traj.append([dx, dy])
            trajectories.append(traj)
        
        # 构建结果
        result = {
            'boxes_3d': np.array(boxes_3d, dtype=np.float32),
            'scores_3d': np.array(scores_3d, dtype=np.float32),
            'labels_3d': np.array(labels_3d, dtype=np.int64),
            'seg_mask': generate_demo_segmentation().numpy(),
            'track_ids': np.array(track_ids, dtype=np.int64),
            'trajectories': np.array(trajectories, dtype=np.float32)
        }
        
        # 保存
        save_path = output_dir / f'result_{frame_id:04d}.npz'
        np.savez(save_path, **result)
        
        if (frame_id + 1) % 5 == 0:
            print(f"  已生成 {frame_id + 1}/{num_frames} 帧")
    
    print(f"\n✓ 完成！演示数据已保存到: {output_dir}")
    print(f"\n下一步:")
    print(f"  1. 可视化单帧:")
    print(f"     python tools/visualize_e2e.py --results {output_dir}/result_0000.npz")
    print(f"  2. 生成视频:")
    print(f"     python tools/visualize_e2e.py --results {output_dir} --video")


def main():
    parser = argparse.ArgumentParser(description='生成演示结果数据')
    parser.add_argument('--num-frames', type=int, default=20, help='生成帧数')
    parser.add_argument('--num-objects', type=int, default=15, help='每帧目标数')
    parser.add_argument('--output', type=str, default='output/demo_results', help='输出目录')
    parser.add_argument('--single', action='store_true', help='仅生成单帧')
    
    args = parser.parse_args()
    
    if args.single:
        # 生成单帧
        result = generate_demo_result(args.num_objects)
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为numpy并保存
        result_np = {}
        for key, value in result.items():
            if torch.is_tensor(value):
                result_np[key] = value.numpy()
            else:
                result_np[key] = value
        
        save_path = output_dir / 'demo_result.npz'
        np.savez(save_path, **result_np)
        
        print(f"✓ 单帧演示数据已保存: {save_path}")
        print(f"\n可视化:")
        print(f"  python tools/visualize_e2e.py --results {save_path}")
    else:
        # 生成序列
        generate_sequence(args.num_frames, args.num_objects, args.output)


if __name__ == '__main__':
    main()
