#!/usr/bin/env python3
"""
完整的可视化测试脚本

一键测试端到端多任务感知系统的可视化功能
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse


def print_header(text):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    print(f"   命令: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✓ {description} 完成\n")
        return True
    else:
        print(f"✗ {description} 失败\n")
        return False


def test_environment():
    """测试环境"""
    print_header("步骤 1: 环境检查")
    
    checks = [
        ("python --version", "Python版本"),
        ("python -c 'import torch; print(f\"PyTorch {torch.__version__}\")'", "PyTorch"),
        ("python -c 'import mmdet3d; print(f\"MMDetection3D {mmdet3d.__version__}\")'", "MMDetection3D"),
        ("python -c 'import matplotlib; print(f\"Matplotlib {matplotlib.__version__}\")'", "Matplotlib"),
        ("python -c 'import cv2; print(f\"OpenCV {cv2.__version__}\")'", "OpenCV"),
    ]
    
    all_passed = True
    for cmd, name in checks:
        if not run_command(cmd, f"检查 {name}"):
            all_passed = False
    
    return all_passed


def generate_demo_data(output_dir='output/demo_results', num_frames=20):
    """生成演示数据"""
    print_header("步骤 2: 生成演示数据")
    
    cmd = f"python tools/generate_demo_results.py --num-frames {num_frames} --output {output_dir}"
    return run_command(cmd, f"生成 {num_frames} 帧演示数据")


def test_single_frame_visualization(results_dir='output/demo_results', output_dir='output/test_vis'):
    """测试单帧可视化"""
    print_header("步骤 3: 测试单帧可视化")
    
    # 找到第一个结果文件
    results_path = Path(results_dir)
    result_files = sorted(results_path.glob('result_*.npz'))
    
    if not result_files:
        print("✗ 未找到结果文件")
        return False
    
    first_result = result_files[0]
    cmd = f"python tools/visualize_e2e.py --results {first_result} --output {output_dir}"
    
    return run_command(cmd, "单帧可视化")


def test_batch_visualization(results_dir='output/demo_results', output_dir='output/test_vis_batch'):
    """测试批量可视化"""
    print_header("步骤 4: 测试批量可视化")
    
    cmd = f"python tools/visualize_e2e.py --results {results_dir} --output {output_dir} --num-samples 10"
    return run_command(cmd, "批量可视化（10帧）")


def test_video_generation(results_dir='output/demo_results', output_dir='output/test_video'):
    """测试视频生成"""
    print_header("步骤 5: 测试视频生成")
    
    cmd = f"python tools/visualize_e2e.py --results {results_dir} --output {output_dir} --video --fps 5"
    return run_command(cmd, "视频生成")


def check_outputs(output_dirs):
    """检查输出文件"""
    print_header("步骤 6: 检查输出文件")
    
    all_good = True
    
    for output_dir in output_dirs:
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"✗ 目录不存在: {output_dir}")
            all_good = False
            continue
        
        # 检查文件
        png_files = list(output_path.glob('**/*.png'))
        mp4_files = list(output_path.glob('**/*.mp4'))
        gif_files = list(output_path.glob('**/*.gif'))
        
        print(f"\n📁 {output_dir}:")
        print(f"   PNG图像: {len(png_files)} 个")
        print(f"   MP4视频: {len(mp4_files)} 个")
        print(f"   GIF动图: {len(gif_files)} 个")
        
        if png_files:
            print(f"   ✓ 找到可视化图像")
            # 显示第一个文件的路径
            print(f"   示例: {png_files[0]}")
        
        if mp4_files:
            print(f"   ✓ 找到视频文件")
            print(f"   示例: {mp4_files[0]}")
        
        if gif_files:
            print(f"   ✓ 找到GIF文件")
            print(f"   示例: {gif_files[0]}")
    
    return all_good


def open_results(output_dir='output/test_vis/combined'):
    """尝试打开结果"""
    print_header("步骤 7: 查看结果")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"✗ 输出目录不存在: {output_dir}")
        return
    
    png_files = sorted(output_path.glob('*.png'))
    if not png_files:
        print("✗ 未找到PNG文件")
        return
    
    first_png = png_files[0]
    print(f"📷 第一个可视化结果: {first_png}")
    
    # 尝试用系统默认程序打开
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', str(first_png)])
        elif sys.platform == 'win32':  # Windows
            os.startfile(str(first_png))
        else:  # Linux
            subprocess.run(['xdg-open', str(first_png)])
        
        print("✓ 已使用默认程序打开图像")
    except:
        print(f"⚠ 无法自动打开，请手动查看: {first_png}")


def print_summary(results):
    """打印测试总结"""
    print_header("测试总结")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print()
    
    for test_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {test_name}")
    
    print()
    
    if passed == total:
        print("🎉 所有测试通过！可视化系统工作正常！")
        return True
    else:
        print("⚠ 部分测试失败，请检查错误信息")
        return False


def main():
    parser = argparse.ArgumentParser(description='测试可视化系统')
    parser.add_argument('--skip-env', action='store_true', help='跳过环境检查')
    parser.add_argument('--skip-demo', action='store_true', help='跳过演示数据生成')
    parser.add_argument('--num-frames', type=int, default=20, help='演示数据帧数')
    parser.add_argument('--open-result', action='store_true', help='自动打开结果')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  BEVFusion 端到端多任务感知 - 可视化系统测试")
    print("="*70)
    
    results = {}
    
    # 测试环境
    if not args.skip_env:
        results['环境检查'] = test_environment()
        if not results['环境检查']:
            print("\n✗ 环境检查失败，请先安装必要的依赖")
            print("  运行: pip install -r requirements.txt")
            return
    
    # 生成演示数据
    if not args.skip_demo:
        results['演示数据生成'] = generate_demo_data(num_frames=args.num_frames)
        if not results['演示数据生成']:
            print("\n✗ 演示数据生成失败")
            return
    
    # 测试可视化
    results['单帧可视化'] = test_single_frame_visualization()
    results['批量可视化'] = test_batch_visualization()
    results['视频生成'] = test_video_generation()
    
    # 检查输出
    output_dirs = [
        'output/test_vis',
        'output/test_vis_batch',
        'output/test_video'
    ]
    results['输出文件检查'] = check_outputs(output_dirs)
    
    # 打开结果
    if args.open_result:
        open_results()
    
    # 打印总结
    all_passed = print_summary(results)
    
    # 显示下一步
    if all_passed:
        print("\n" + "="*70)
        print("  下一步操作")
        print("="*70)
        print("\n1. 查看可视化结果:")
        print("   - 单帧: output/test_vis/combined/")
        print("   - 批量: output/test_vis_batch/combined/")
        print("   - 视频: output/test_video/perception_video.mp4")
        print()
        print("2. 使用真实模型:")
        print("   python examples/run_e2e_perception.py \\")
        print("       configs/nuscenes/det/bevfusion-e2e-perception.yaml \\")
        print("       checkpoint.pth --visualize")
        print()
        print("3. 自定义可视化:")
        print("   python tools/visualize_e2e.py --help")
        print()
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
