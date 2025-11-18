"""
BEVFusion 端到端多任务感知可视化工具

功能:
- 3D检测框可视化（BEV视图和透视图）
- 语义地图可视化
- 多目标跟踪可视化
- 轨迹预测可视化
- 生成视频和GIF
"""

import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
import argparse
from pathlib import Path
from tqdm import tqdm
import imageio
from PIL import Image, ImageDraw, ImageFont

# 颜色方案
COLORS = {
    'car': (255, 158, 0),
    'truck': (255, 99, 71),
    'bus': (255, 140, 0),
    'trailer': (255, 127, 80),
    'construction_vehicle': (233, 150, 70),
    'pedestrian': (0, 0, 230),
    'motorcycle': (255, 61, 99),
    'bicycle': (220, 20, 60),
    'traffic_cone': (255, 255, 0),
    'barrier': (112, 128, 144),
}

SEGMENTATION_COLORS = {
    0: (128, 64, 128),   # Drivable area (紫色)
    1: (244, 35, 232),   # Lane (品红)
    2: (70, 70, 70),     # Sidewalk (灰色)
    3: (102, 102, 156),  # Other (蓝灰)
}

CLASS_NAMES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]


class E2EVisualizer:
    """端到端多任务感知可视化器"""
    
    def __init__(self, output_dir='output/visualization', dpi=150, figsize=(20, 15)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        # 创建子目录
        (self.output_dir / 'detection').mkdir(exist_ok=True)
        (self.output_dir / 'segmentation').mkdir(exist_ok=True)
        (self.output_dir / 'tracking').mkdir(exist_ok=True)
        (self.output_dir / 'combined').mkdir(exist_ok=True)
    
    def visualize_sample(self, result, img=None, frame_id=0, save=True):
        """
        可视化单个样本的所有任务
        
        Args:
            result: 模型输出结果字典
            img: 原始图像（可选）
            frame_id: 帧ID
            save: 是否保存
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 3D检测可视化 (左上)
        self._plot_detection_bev(axes[0, 0], result)
        axes[0, 0].set_title('3D Object Detection (BEV View)', fontsize=14, fontweight='bold')
        
        # 2. 语义地图可视化 (右上)
        self._plot_segmentation(axes[0, 1], result)
        axes[0, 1].set_title('Semantic Map', fontsize=14, fontweight='bold')
        
        # 3. 多目标跟踪可视化 (左下)
        self._plot_tracking(axes[1, 0], result)
        axes[1, 0].set_title('Multi-Object Tracking', fontsize=14, fontweight='bold')
        
        # 4. 轨迹预测可视化 (右下)
        self._plot_trajectories(axes[1, 1], result)
        axes[1, 1].set_title('Trajectory Prediction', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'combined' / f'frame_{frame_id:04d}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
        
        return fig
    
    def _plot_detection_bev(self, ax, result):
        """绘制BEV检测结果"""
        boxes = result.get('boxes_3d', None)
        scores = result.get('scores_3d', None)
        labels = result.get('labels_3d', None)
        
        if boxes is None or len(boxes) == 0:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=20)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            return
        
        # 转换为numpy
        boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        scores = scores.cpu().numpy() if torch.is_tensor(scores) else scores
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        # 绘制自车
        ego_rect = Rectangle((-2, -1), 4, 2, linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.5)
        ax.add_patch(ego_rect)
        ax.text(0, 0, 'EGO', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 绘制检测框
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:  # 过滤低置信度
                continue
            
            x, y, z, w, h, l, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]
            
            # 获取颜色
            class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else 'unknown'
            color = np.array(COLORS.get(class_name, (128, 128, 128))) / 255.0
            
            # 计算四个角点
            corners = self._get_box_corners(x, y, w, l, yaw)
            
            # 绘制框
            poly = Polygon(corners, closed=True, linewidth=2, 
                          edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(poly)
            
            # 绘制朝向箭头
            arrow_length = l / 2
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            ax.arrow(x, y, dx, dy, head_width=1.0, head_length=0.5, 
                    fc=color, ec=color, linewidth=2)
            
            # 标注类别和置信度
            ax.text(x, y + h/2 + 1, f'{class_name}\n{score:.2f}', 
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                   color='white', fontweight='bold')
        
        # 设置坐标轴
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', 
                      markerfacecolor=np.array(COLORS[name])/255.0, 
                      markersize=10, label=name.capitalize())
            for name in ['car', 'pedestrian', 'truck', 'bicycle']
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_segmentation(self, ax, result):
        """绘制语义分割地图"""
        seg_mask = result.get('seg_mask', None)
        
        if seg_mask is None:
            ax.text(0.5, 0.5, 'No segmentation', ha='center', va='center', fontsize=20)
            return
        
        # 转换为numpy
        if torch.is_tensor(seg_mask):
            seg_mask = seg_mask.cpu().numpy()
        
        # 创建RGB图像
        num_classes, H, W = seg_mask.shape
        rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 应用颜色
        for cls_id in range(num_classes):
            mask = seg_mask[cls_id] > 0.5
            color = SEGMENTATION_COLORS.get(cls_id, (128, 128, 128))
            rgb_mask[mask] = color
        
        # 显示
        ax.imshow(rgb_mask, origin='lower')
        ax.axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_labels = ['Drivable', 'Lane', 'Sidewalk', 'Other']
        legend_elements = [
            Patch(facecolor=np.array(SEGMENTATION_COLORS[i])/255.0, label=legend_labels[i])
            for i in range(min(num_classes, len(legend_labels)))
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _plot_tracking(self, ax, result):
        """绘制跟踪结果"""
        boxes = result.get('boxes_3d', None)
        track_ids = result.get('track_ids', None)
        
        if boxes is None or track_ids is None or len(boxes) == 0:
            ax.text(0.5, 0.5, 'No tracking', ha='center', va='center', fontsize=20)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            return
        
        # 转换为numpy
        boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        track_ids = track_ids.cpu().numpy() if torch.is_tensor(track_ids) else track_ids
        
        # 绘制自车
        ego_rect = Rectangle((-2, -1), 4, 2, linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.5)
        ax.add_patch(ego_rect)
        
        # 为每个track ID分配颜色
        unique_ids = np.unique(track_ids)
        cmap = plt.cm.get_cmap('tab20')
        id_colors = {uid: cmap(i % 20) for i, uid in enumerate(unique_ids)}
        
        # 绘制跟踪框
        for box, tid in zip(boxes, track_ids):
            x, y, w, l, yaw = box[0], box[1], box[3], box[5], box[6]
            
            color = id_colors[tid]
            corners = self._get_box_corners(x, y, w, l, yaw)
            
            # 绘制框
            poly = Polygon(corners, closed=True, linewidth=3,
                          edgecolor=color, facecolor='none')
            ax.add_patch(poly)
            
            # 绘制ID
            ax.text(x, y, f'ID:{tid}', ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.8),
                   color='white')
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
    
    def _plot_trajectories(self, ax, result):
        """绘制轨迹预测"""
        boxes = result.get('boxes_3d', None)
        trajs = result.get('trajectories', None)
        labels = result.get('labels_3d', None)
        
        if boxes is None or trajs is None or len(boxes) == 0:
            ax.text(0.5, 0.5, 'No trajectories', ha='center', va='center', fontsize=20)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            return
        
        # 转换为numpy
        boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        trajs = trajs.cpu().numpy() if torch.is_tensor(trajs) else trajs
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        # 绘制自车
        ego_rect = Rectangle((-2, -1), 4, 2, linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.5)
        ax.add_patch(ego_rect)
        
        # 绘制当前位置和轨迹
        for box, traj, label in zip(boxes, trajs, labels):
            x, y = box[0], box[1]
            
            # 获取颜色
            class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else 'unknown'
            color = np.array(COLORS.get(class_name, (128, 128, 128))) / 255.0
            
            # 当前位置
            ax.plot(x, y, 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=2)
            
            # 轨迹点（相对坐标）
            traj_x = traj[:, 0] + x
            traj_y = traj[:, 1] + y
            
            # 绘制轨迹线
            ax.plot(traj_x, traj_y, '--', color=color, linewidth=2, alpha=0.7)
            
            # 绘制轨迹点
            ax.plot(traj_x, traj_y, 'o', color=color, markersize=6, alpha=0.7)
            
            # 标注时间步
            for t, (tx, ty) in enumerate(zip(traj_x, traj_y)):
                ax.text(tx, ty, f't+{t+1}', fontsize=7, ha='center', va='bottom',
                       color='black', bbox=dict(boxstyle='round,pad=0.2', 
                       facecolor='white', alpha=0.7))
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
    
    def _get_box_corners(self, x, y, w, l, yaw):
        """计算box的四个角点"""
        # 中心点
        cx, cy = x, y
        
        # 未旋转的角点（相对于中心）
        corners_local = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])
        
        # 旋转矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # 旋转并平移
        corners = corners_local @ rot_matrix.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        
        return corners
    
    def create_video(self, results, output_path='output/perception_video.mp4', 
                     fps=10, imgs=None):
        """
        创建可视化视频
        
        Args:
            results: 结果列表
            output_path: 输出视频路径
            fps: 帧率
            imgs: 原始图像列表（可选）
        """
        print(f"\n🎬 创建视频: {output_path}")
        
        # 生成帧
        frames = []
        for i, result in enumerate(tqdm(results, desc="渲染帧")):
            # 生成可视化
            fig = self.visualize_sample(result, 
                                       img=imgs[i] if imgs else None,
                                       frame_id=i, 
                                       save=False)
            
            # 转换为图像
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)
        
        # 写入视频
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"✓ 视频已保存: {output_path}")
        
        # 同时保存GIF（前30帧）
        gif_path = output_path.with_suffix('.gif')
        imageio.mimsave(str(gif_path), frames[:min(30, len(frames))], fps=fps//2)
        print(f"✓ GIF已保存: {gif_path}")
    
    def create_comparison_grid(self, results_list, labels, output_path='output/comparison.png'):
        """
        创建对比网格图
        
        Args:
            results_list: 多个模型的结果列表
            labels: 模型标签
            output_path: 输出路径
        """
        n_models = len(results_list)
        fig, axes = plt.subplots(n_models, 4, figsize=(20, 5*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (results, label) in enumerate(zip(results_list, labels)):
            result = results[0]  # 取第一个样本
            
            # 绘制四个任务
            self._plot_detection_bev(axes[i, 0], result)
            self._plot_segmentation(axes[i, 1], result)
            self._plot_tracking(axes[i, 2], result)
            self._plot_trajectories(axes[i, 3], result)
            
            # 添加模型标签
            axes[i, 0].set_ylabel(label, fontsize=14, fontweight='bold')
        
        # 添加列标题
        axes[0, 0].set_title('Detection', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Segmentation', fontsize=14, fontweight='bold')
        axes[0, 2].set_title('Tracking', fontsize=14, fontweight='bold')
        axes[0, 3].set_title('Trajectories', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 对比图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='BEVFusion E2E 可视化工具')
    parser.add_argument('--results', type=str, required=True, help='结果文件路径（.npz）')
    parser.add_argument('--output', type=str, default='output/visualization', help='输出目录')
    parser.add_argument('--video', action='store_true', help='生成视频')
    parser.add_argument('--fps', type=int, default=10, help='视频帧率')
    parser.add_argument('--num-samples', type=int, default=-1, help='可视化样本数（-1表示全部）')
    
    args = parser.parse_args()
    
    # 加载结果
    print(f"📂 加载结果: {args.results}")
    
    if args.results.endswith('.npz'):
        # 单个文件
        data = np.load(args.results, allow_pickle=True)
        results = [data]
    else:
        # 目录
        result_files = sorted(Path(args.results).glob('*.npz'))
        results = [np.load(f, allow_pickle=True) for f in result_files]
    
    if args.num_samples > 0:
        results = results[:args.num_samples]
    
    print(f"✓ 加载了 {len(results)} 个结果")
    
    # 创建可视化器
    visualizer = E2EVisualizer(output_dir=args.output)
    
    # 可视化每个样本
    print("\n🎨 生成可视化...")
    for i, result_data in enumerate(tqdm(results)):
        # 转换为字典格式
        result = {key: result_data[key] for key in result_data.files}
        
        # 可视化
        visualizer.visualize_sample(result, frame_id=i, save=True)
    
    # 生成视频
    if args.video and len(results) > 1:
        results_list = [{key: r[key] for key in r.files} for r in results]
        visualizer.create_video(results_list, 
                               output_path=f'{args.output}/perception_video.mp4',
                               fps=args.fps)
    
    print("\n✅ 可视化完成！")
    print(f"📁 输出目录: {args.output}")


if __name__ == '__main__':
    main()
