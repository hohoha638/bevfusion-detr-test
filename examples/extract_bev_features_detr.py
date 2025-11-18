"""
BEV特征提取和DETR检测示例
演示如何使用BEVFusionDETR模型提取融合后的BEV特征并进行检测
"""
import torch
import numpy as np
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BEV Feature Extraction with DETR')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型checkpoint路径')
    parser.add_argument('--out-dir', default='output/bev_features', help='输出目录')
    parser.add_argument('--device', default='cuda:0', help='使用的设备')
    parser.add_argument('--save-features', action='store_true', help='是否保存特征')
    parser.add_argument('--visualize', action='store_true', help='是否可视化')
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device='cuda:0'):
    """
    加载模型
    
    Args:
        config_path (str): 配置文件路径
        checkpoint_path (str): checkpoint路径
        device (str): 设备
        
    Returns:
        model: 加载的模型
        cfg: 配置对象
    """
    # 加载配置
    cfg = Config.fromfile(config_path)
    
    # 构建模型
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功: {checkpoint_path}")
    
    return model, cfg


def extract_bev_features(model, data, device='cuda:0'):
    """
    提取BEV特征
    
    Args:
        model: BEVFusionDETR模型
        data (dict): 输入数据
        device (str): 设备
        
    Returns:
        dict: BEV特征字典
            - 'fused_bev': 融合后的原始BEV特征
            - 'processed_bev': 处理后的BEV特征
            - 'bev_flatten': 展平的BEV特征
            - 'position_encoding': 位置编码
    """
    with torch.no_grad():
        # 将数据移到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        # 提取BEV特征
        bev_features = model.extract_bev_features_only(
            img=data['img'],
            points=data['points'],
            camera2ego=data['camera2ego'],
            lidar2ego=data['lidar2ego'],
            lidar2camera=data['lidar2camera'],
            lidar2image=data['lidar2image'],
            camera_intrinsics=data['camera_intrinsics'],
            camera2lidar=data['camera2lidar'],
            img_aug_matrix=data['img_aug_matrix'],
            lidar_aug_matrix=data['lidar_aug_matrix'],
            metas=data['img_metas'],
            depths=data.get('depths', None),
            radar=data.get('radar', None),
        )
    
    return bev_features


def run_detection(model, data, device='cuda:0'):
    """
    运行检测
    
    Args:
        model: BEVFusionDETR模型
        data (dict): 输入数据
        device (str): 设备
        
    Returns:
        list: 检测结果
    """
    with torch.no_grad():
        # 将数据移到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        # 前向传播
        results = model(**data)
    
    return results


def save_features(bev_features, save_path):
    """
    保存BEV特征
    
    Args:
        bev_features (dict): BEV特征字典
        save_path (str): 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将tensor转换为numpy
    features_to_save = {}
    for key, value in bev_features.items():
        if isinstance(value, torch.Tensor):
            features_to_save[key] = value.cpu().numpy()
    
    # 保存
    np.savez(save_path, **features_to_save)
    print(f"✅ 特征已保存到: {save_path}")


def visualize_bev_features(bev_features, save_path=None):
    """
    可视化BEV特征
    
    Args:
        bev_features (dict): BEV特征字典
        save_path (str): 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    # 获取BEV特征
    bev_feat = bev_features['bev_features'][0].cpu().numpy()  # [C, H, W]
    
    # 计算特征的L2范数作为可视化
    feat_norm = np.linalg.norm(bev_feat, axis=0)  # [H, W]
    
    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始融合特征
    if 'fused_bev' in bev_features:
        fused_feat = bev_features['fused_bev'][0].cpu().numpy()
        fused_norm = np.linalg.norm(fused_feat, axis=0)
        axes[0].imshow(fused_norm, cmap='viridis')
        axes[0].set_title('Fused BEV Features')
        axes[0].axis('off')
    
    # 处理后的BEV特征
    axes[1].imshow(feat_norm, cmap='viridis')
    axes[1].set_title('Processed BEV Features')
    axes[1].axis('off')
    
    # 特征通道的均值
    feat_mean = np.mean(bev_feat, axis=0)
    axes[2].imshow(feat_mean, cmap='viridis')
    axes[2].set_title('BEV Features (Channel Mean)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_feature_info(bev_features):
    """
    打印BEV特征信息
    
    Args:
        bev_features (dict): BEV特征字典
    """
    print("\n" + "="*60)
    print("📊 BEV特征信息")
    print("="*60)
    
    for key, value in bev_features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:25s}: shape={list(value.shape)}, "
                  f"dtype={value.dtype}, device={value.device}")
            print(f"  {'':25s}  min={value.min().item():.4f}, "
                  f"max={value.max().item():.4f}, "
                  f"mean={value.mean().item():.4f}")
    
    print("="*60 + "\n")


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("🚀 开始BEV特征提取...")
    
    # 1. 加载模型
    print("\n📥 加载模型...")
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    
    # 2. 准备数据（这里使用第一个样本作为示例）
    print("\n📥 准备数据...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    
    # 3. 处理数据
    print("\n⚙️  处理数据...")
    for i, data in enumerate(data_loader):
        print(f"\n处理样本 {i+1}/{len(data_loader)}")
        
        # 提取BEV特征
        print("  🔍 提取BEV特征...")
        bev_features = extract_bev_features(model, data, args.device)
        
        # 打印特征信息
        print_feature_info(bev_features)
        
        # 运行检测
        print("  🎯 运行检测...")
        results = run_detection(model, data, args.device)
        print(f"  ✅ 检测完成，检测到 {len(results[0]['boxes_3d'])} 个目标")
        
        # 保存特征
        if args.save_features:
            save_path = os.path.join(args.out_dir, f'bev_features_{i:04d}.npz')
            save_features(bev_features, save_path)
        
        # 可视化
        if args.visualize:
            vis_path = os.path.join(args.out_dir, f'bev_visualization_{i:04d}.png')
            visualize_bev_features(bev_features, vis_path)
        
        # 只处理前几个样本作为示例
        if i >= 2:
            break
    
    print("\n✅ 处理完成！")
    print(f"📁 结果保存在: {args.out_dir}")


if __name__ == '__main__':
    main()
