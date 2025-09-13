### 仿真脚本取数据集采集及可视化

## 环境配置

### 系统要求
- Linux系统 (推荐Ubuntu 20.04+)
- NVIDIA GPU (支持CUDA 11.8)
- Miniconda/Anaconda

### Conda环境设置
```bash
# 创建conda环境
conda create -n genesis-env python=3.10

# 激活环境
conda activate genesis-env

# 安装PyTorch (CUDA版本)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装Genesis仿真框架
pip install genesis-world==0.2.1

# 安装其他依赖
pip install matplotlib numpy lxml requests psutil
```

### 主要依赖包版本
- Python: 3.10.18
- torch: 2.7.1+cu118
- genesis-world: 0.2.1
- numpy: 1.26.4
- matplotlib: 3.10.3
- CUDA: 11.8

### 验证安装
```bash
conda activate genesis-env
python -c "import genesis as gs; print('Genesis安装成功!')"
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

## 项目模块
- 控制脚本模块：`robotcontroller.py`
- 初始化模型模块：`robotinitial.py`
- 数据记录以及可视化模块: `logger.py`
- 滤波器模块: `processer.py`

## 使用说明
1. 确保已按照上述步骤配置好conda环境
2. 激活genesis-env环境：`conda activate genesis-env`
3. 运行相应的Python脚本进行机器人仿真

## 重要文件说明
- `urdf/red-duo-fixed.urdf`: 修复后的机器人URDF文件 (网格路径已修复)
- `urdf/red-duo.urdf`: 原始URDF文件 (包含ROS包路径格式)
- `log/`: 数据日志目录
- `meshes/`: 机器人3D网格文件目录

## 故障排除

### URDF文件路径问题
如果遇到"Asset file not found"错误，请确保使用`urdf/red-duo-fixed.urdf`文件，该文件已将ROS包路径格式转换为相对路径格式。

### GPU/CUDA问题
```bash
# 检查CUDA是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import genesis as gs; gs.init(backend=gs.gpu)"
```

### 显示问题
如果仿真窗口无法显示，请检查：
- X11转发设置 (如果使用SSH)
- 显卡驱动是否正确安装
