# Jetson 双目测距 & YOLO 分割调试工具

本项目旨在解决双目视差图噪点过多、深度提取不准的问题，并在 Jetson 平台上进行 TensorRT 优化加速。

## 🌟 功能特性
- **工业级滤波**: 集成 VPI 视差计算 + 时域平滑 (EMA) + 边缘保留滤波 (WLS)。
- **YOLOv11 优化**: 支持 TensorRT Engine 自动编译，支持 FP16 精度和动态分辨率。
- **鲁棒性设计**: 支持配置文件加载 (`config.yaml`)、自动编译模型、自动回退 PyTorch 推理。
- **深度提取增强**: 使用掩码腐蚀和 IQR 统计滤波剔除深度离群点。

## 🛠️ 环境准备 (Jetson)
确保您的 Jetson 已安装 JetPack 5.x/6.x 以及以下组件：
- **CUDA 11.4/12.2**
- **TensorRT 8.x/10.x**
- **VPI 2.x/3.x**
- **Python 3.8/3.10**

### 安装依赖
```bash
pip3 install ultralytics pyyaml opencv-python numpy py-cpuinfo
# 如果需要 WLS 滤波
pip3 install opencv-contrib-python
```

## 🚀 快速开始

### 1. 模型编译
如果您只有 `.pt` 权重文件，可以通过以下命令一键编译 TensorRT Engine：
```bash
chmod +x build_engine.sh
./build_engine.sh
```
这将在当前目录生成 `yolo11n-seg-dyn.engine`。

### 2. 运行测距脚本
```bash
# 使用默认配置运行
python3 depth_wls_yolo.py

# 指定配置文件运行
python3 depth_wls_yolo.py --config config.yaml
```

## 📂 核心文件说明
- `depth_wls_yolo.py`: 主程序，包含相机采集、VPI 推理、YOLO 预测、深度计算和 Web 预览。
- `build_engine.py`: Python 版模型转换工具。
- `build_engine.sh`: Shell 版一键转换脚本，针对 Jetson 优化了环境变量和参数。
- `config.yaml`: 配置文件，支持相机参数、标定文件路径、YOLO 模型路径等设置。
- `test_depth_wls.py`: 单元测试脚本。

## 🧪 单元测试
运行以下命令进行逻辑验证：
```bash
python3 test_depth_wls.py
```

## ⚠️ 注意事项
- 如果出现 `FileNotFoundError`，脚本会自动寻找同名的 `.pt` 文件并触发 `build_engine.sh`。
- 确保您的 `calib_data/stereo_calib.npz` 标定文件路径正确。
- VPI 默认使用 `CUDA` 后端，如果显存紧张可以尝试在配置中切换为 `PVA`。
