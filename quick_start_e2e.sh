#!/bin/bash
# BEVFusion 端到端多任务感知系统 - 快速启动脚本

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "  BEVFusion 端到端多任务感知系统 - 快速启动"
echo "======================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印成功消息
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 函数：打印信息
info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 函数：打印警告
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 函数：打印错误
error() {
    echo -e "${RED}✗ $1${NC}"
}

# 检查必要工具
check_requirements() {
    info "检查系统要求..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        error "Python未安装"
        exit 1
    fi
    success "Python: $(python --version)"
    
    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        warning "nvidia-smi未找到，GPU可能不可用"
    else
        success "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
    
    # 检查conda
    if ! command -v conda &> /dev/null; then
        warning "Conda未安装，建议使用conda环境"
    else
        success "Conda已安装"
    fi
    
    echo ""
}

# 创建环境
setup_environment() {
    info "设置环境..."
    
    if command -v conda &> /dev/null; then
        # 使用conda
        if conda env list | grep -q bevfusion-e2e; then
            success "环境 'bevfusion-e2e' 已存在"
        else
            info "创建conda环境..."
            conda create -n bevfusion-e2e python=3.8 -y
            success "环境创建完成"
        fi
        
        info "激活环境..."
        eval "$(conda shell.bash hook)"
        conda activate bevfusion-e2e
        success "环境已激活"
    else
        # 使用venv
        if [ -d "venv" ]; then
            success "虚拟环境已存在"
        else
            info "创建虚拟环境..."
            python -m venv venv
            success "虚拟环境创建完成"
        fi
        
        source venv/bin/activate
        success "虚拟环境已激活"
    fi
    
    echo ""
}

# 安装依赖
install_dependencies() {
    info "安装依赖包..."
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    # 安装PyTorch (根据CUDA版本调整)
    if nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        info "检测到CUDA版本: $CUDA_VERSION"
        
        if [ "$CUDA_VERSION" == "11.3" ]; then
            pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
        elif [ "$CUDA_VERSION" == "11.1" ]; then
            pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        else
            pip install torch torchvision
        fi
    else
        pip install torch torchvision
    fi
    
    # 安装MMDetection3D生态
    pip install openmim
    mim install mmengine
    mim install "mmcv-full>=1.4.0,<1.7.0"
    mim install "mmdet>=2.24.0,<3.0.0"
    mim install "mmsegmentation>=0.20.0,<1.0.0"
    
    # 安装其他依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # 额外工具
    pip install scipy scikit-learn matplotlib seaborn \
        opencv-python pillow tensorboard \
        numba nuscenes-devkit motmetrics \
        imageio imageio-ffmpeg tqdm
    
    success "依赖安装完成"
    echo ""
}

# 编译CUDA算子
compile_ops() {
    info "编译CUDA算子..."
    
    if [ -d "mmdet3d/ops" ]; then
        cd mmdet3d/ops
        python setup.py develop
        cd ../..
        success "CUDA算子编译完成"
    else
        warning "未找到mmdet3d/ops目录，跳过编译"
    fi
    
    echo ""
}

# 下载数据
download_data() {
    info "数据准备..."
    
    if [ -d "data/nuscenes" ]; then
        success "数据目录已存在: data/nuscenes"
    else
        warning "数据目录不存在"
        echo ""
        echo "请执行以下步骤准备数据:"
        echo "1. 从 https://www.nuscenes.org/nuscenes 下载数据集"
        echo "2. 解压到 data/nuscenes/ 目录"
        echo "3. 运行预处理脚本:"
        echo "   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes"
        echo ""
        read -p "是否已准备好数据? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "请先准备数据后再继续"
            exit 1
        fi
    fi
    
    echo ""
}

# 下载预训练模型
download_pretrained() {
    info "下载预训练模型..."
    
    mkdir -p pretrained
    
    # Swin Transformer预训练权重
    if [ -f "pretrained/swin_tiny_patch4_window7_224.pth" ]; then
        success "Swin Transformer权重已存在"
    else
        info "下载Swin Transformer权重..."
        wget -O pretrained/swin_tiny_patch4_window7_224.pth \
            https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
        success "下载完成"
    fi
    
    echo ""
}

# 验证安装
verify_installation() {
    info "验证安装..."
    
    python -c "
import torch
import mmcv
import mmdet
import mmdet3d

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
print('✓ MMCV:', mmcv.__version__)
print('✓ MMDetection:', mmdet.__version__)
print('✓ MMDetection3D:', mmdet3d.__version__)

# 测试CUDA算子
try:
    from mmdet3d.ops import Voxelization
    print('✓ CUDA operators compiled')
except:
    print('✗ CUDA operators not available')
"
    
    if [ $? -eq 0 ]; then
        success "安装验证通过"
    else
        error "安装验证失败"
        exit 1
    fi
    
    echo ""
}

# 运行演示
run_demo() {
    info "运行演示..."
    
    echo ""
    echo "请选择运行模式:"
    echo "1) 训练模型"
    echo "2) 运行推理（需要checkpoint）"
    echo "3) 可视化结果"
    echo "4) 跳过演示"
    echo ""
    read -p "请选择 (1-4): " choice
    
    case $choice in
        1)
            info "启动训练..."
            python tools/train.py configs/nuscenes/det/bevfusion-e2e-perception.yaml
            ;;
        2)
            read -p "请输入checkpoint路径: " checkpoint
            if [ -f "$checkpoint" ]; then
                info "运行推理..."
                python examples/run_e2e_perception.py \
                    configs/nuscenes/det/bevfusion-e2e-perception.yaml \
                    "$checkpoint" \
                    --visualize \
                    --save-results \
                    --num-samples 10
            else
                error "Checkpoint文件不存在: $checkpoint"
            fi
            ;;
        3)
            read -p "请输入结果目录: " results_dir
            if [ -d "$results_dir" ]; then
                info "生成可视化..."
                python tools/visualize_e2e.py \
                    --results "$results_dir" \
                    --output output/visualization \
                    --video \
                    --fps 10
            else
                error "结果目录不存在: $results_dir"
            fi
            ;;
        4)
            info "跳过演示"
            ;;
        *)
            warning "无效选择"
            ;;
    esac
    
    echo ""
}

# 主流程
main() {
    echo ""
    info "开始安装和配置..."
    echo ""
    
    # 检查要求
    check_requirements
    
    # 询问用户要执行的步骤
    echo "请选择要执行的步骤:"
    echo "1) 完整安装（推荐首次使用）"
    echo "2) 仅验证安装"
    echo "3) 运行演示"
    echo ""
    read -p "请选择 (1-3): " install_choice
    
    case $install_choice in
        1)
            setup_environment
            install_dependencies
            compile_ops
            download_data
            download_pretrained
            verify_installation
            
            echo ""
            echo "======================================================================"
            success "安装完成！"
            echo "======================================================================"
            echo ""
            echo "下一步:"
            echo "1. 训练模型:"
            echo "   bash tools/dist_train.sh configs/nuscenes/det/bevfusion-e2e-perception.yaml 4"
            echo ""
            echo "2. 运行推理:"
            echo "   python examples/run_e2e_perception.py \\"
            echo "       configs/nuscenes/det/bevfusion-e2e-perception.yaml \\"
            echo "       checkpoint.pth --visualize --save-results"
            echo ""
            echo "3. 可视化结果:"
            echo "   python tools/visualize_e2e.py --results output/results --video"
            echo ""
            
            # 询问是否运行演示
            read -p "是否运行演示? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                run_demo
            fi
            ;;
        2)
            verify_installation
            ;;
        3)
            run_demo
            ;;
        *)
            error "无效选择"
            exit 1
            ;;
    esac
    
    echo ""
    info "完成！祝使用愉快！"
    echo ""
}

# 运行主流程
main
