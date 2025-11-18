@echo off
REM BEVFusion 端到端多任务感知系统 - Windows快速启动脚本

echo ======================================================================
echo   BEVFusion 端到端多任务感知系统 - 快速启动 (Windows)
echo ======================================================================
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python未安装
    pause
    exit /b 1
)
echo [成功] Python已安装

REM 检查NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] nvidia-smi未找到，GPU可能不可用
) else (
    echo [成功] 检测到NVIDIA GPU
)

echo.
echo 请选择操作:
echo 1. 完整安装 (首次使用)
echo 2. 安装依赖包
echo 3. 编译CUDA算子
echo 4. 验证安装
echo 5. 运行训练
echo 6. 运行推理
echo 7. 生成可视化
echo 8. 退出
echo.

set /p choice="请输入选项 (1-8): "

if "%choice%"=="1" goto :full_install
if "%choice%"=="2" goto :install_deps
if "%choice%"=="3" goto :compile_ops
if "%choice%"=="4" goto :verify
if "%choice%"=="5" goto :train
if "%choice%"=="6" goto :inference
if "%choice%"=="7" goto :visualize
if "%choice%"=="8" goto :end

:full_install
echo.
echo ====== 执行完整安装 ======
echo.

REM 创建conda环境
echo [信息] 检查conda环境...
conda env list | findstr "bevfusion-e2e" >nul 2>&1
if %errorlevel% neq 0 (
    echo [信息] 创建conda环境 bevfusion-e2e...
    call conda create -n bevfusion-e2e python=3.8 -y
    echo [成功] 环境创建完成
) else (
    echo [成功] 环境已存在
)

echo [信息] 激活环境...
call conda activate bevfusion-e2e

REM 安装依赖
call :install_deps

REM 编译算子
call :compile_ops

REM 验证安装
call :verify

echo.
echo [成功] 完整安装完成！
echo.
echo 下一步:
echo 1. 训练模型: python tools/train.py configs/nuscenes/det/bevfusion-e2e-perception.yaml
echo 2. 运行推理: python examples/run_e2e_perception.py config.yaml checkpoint.pth --visualize
echo 3. 生成可视化: python tools/visualize_e2e.py --results output/results --video
echo.
goto :menu

:install_deps
echo.
echo ====== 安装依赖包 ======
echo.

echo [信息] 升级pip...
python -m pip install --upgrade pip setuptools wheel

echo [信息] 安装PyTorch...
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

echo [信息] 安装MMDetection3D生态...
pip install openmim
call mim install mmengine
call mim install "mmcv-full>=1.4.0,<1.7.0"
call mim install "mmdet>=2.24.0,<3.0.0"
call mim install "mmsegmentation>=0.20.0,<1.0.0"

echo [信息] 安装项目依赖...
if exist requirements.txt (
    pip install -r requirements.txt
)

echo [信息] 安装额外工具...
pip install scipy scikit-learn matplotlib seaborn opencv-python pillow tensorboard numba nuscenes-devkit motmetrics imageio imageio-ffmpeg tqdm

echo [成功] 依赖安装完成
goto :menu

:compile_ops
echo.
echo ====== 编译CUDA算子 ======
echo.

if exist mmdet3d\ops (
    echo [信息] 编译中...
    cd mmdet3d\ops
    python setup.py develop
    cd ..\..
    echo [成功] CUDA算子编译完成
) else (
    echo [警告] 未找到 mmdet3d\ops 目录
)
goto :menu

:verify
echo.
echo ====== 验证安装 ======
echo.

python -c "import torch; import mmcv; import mmdet; import mmdet3d; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MMCV:', mmcv.__version__); print('MMDetection:', mmdet.__version__); print('MMDetection3D:', mmdet3d.__version__)"

if %errorlevel% equ 0 (
    echo [成功] 安装验证通过
) else (
    echo [错误] 安装验证失败
)
goto :menu

:train
echo.
echo ====== 训练模型 ======
echo.

if not exist configs\nuscenes\det\bevfusion-e2e-perception.yaml (
    echo [错误] 配置文件不存在
    goto :menu
)

echo [信息] 启动训练...
python tools\train.py configs\nuscenes\det\bevfusion-e2e-perception.yaml

goto :menu

:inference
echo.
echo ====== 运行推理 ======
echo.

set /p checkpoint="请输入checkpoint路径: "
if not exist "%checkpoint%" (
    echo [错误] Checkpoint不存在: %checkpoint%
    goto :menu
)

echo [信息] 运行推理...
python examples\run_e2e_perception.py ^
    configs\nuscenes\det\bevfusion-e2e-perception.yaml ^
    "%checkpoint%" ^
    --visualize ^
    --save-results ^
    --num-samples 10

echo [成功] 推理完成
goto :menu

:visualize
echo.
echo ====== 生成可视化 ======
echo.

set /p results_dir="请输入结果目录路径: "
if not exist "%results_dir%" (
    echo [错误] 结果目录不存在: %results_dir%
    goto :menu
)

echo [信息] 生成可视化...
python tools\visualize_e2e.py ^
    --results "%results_dir%" ^
    --output output\visualization ^
    --video ^
    --fps 10

echo [成功] 可视化完成
echo [信息] 输出目录: output\visualization
goto :menu

:menu
echo.
echo 按任意键返回菜单...
pause >nul
cls
goto :eof

:end
echo.
echo 感谢使用！
echo.
pause
