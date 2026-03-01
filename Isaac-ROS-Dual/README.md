# Isaac-ROS-Dual: VPI 硬件加速双目深度估计

在 Jetson Orin Nano Super 上用 **Python + VPI** 跑双目深度——不需要 ROS，不需要 Docker。

## 核心思路

```
MX219 CSI × 2 → nvarguscamerasrc → BGR → 校正(remap) → VPI stereodisp (CUDA/PVA) → 深度图
       ^                                                    ^
    Python 打开                                        硬件加速，零CPU运算
```

Python 只做**指挥官**（开相机、收结果、推流），VPI 在 GPU/PVA 上做**苦力**（像素级视差计算）。

## 环境要求

系统自带，不需要 pip install 任何东西：

| 组件 | 版本 | 来源 |
|------|------|------|
| VPI  | 3.2.4 | JetPack 预装 |
| OpenCV | 4.10.0 (CUDA) | JetPack 预装 |
| Python | 3.10 | 系统 |
| CUDA | 12.6 | JetPack 预装 |

## 快速开始

```bash
cd ~/Desktop/Isaac-ROS-Dual

# 基本用法（默认 CUDA 后端，1280×720@30fps）
python3 stereo_depth.py

# 指定标定文件和 VPI 后端
python3 stereo_depth.py \
    --calib ../MX219-DUAL/calib_data/stereo_calib.npz \
    --vpi-backend CUDA \
    --max-disparity 128

# 更低分辨率（如果跑不动）
python3 stereo_depth.py --width 640 --height 480

# 用 OFA 后端（Orin 专有，效率最高但精度稍低）
python3 stereo_depth.py --vpi-backend OFA
```

然后在你电脑的浏览器打开 `http://<jetson-ip>:8080`，会看到四路画面：

| 画面 | 说明 |
|------|------|
| Left (rectified) | 校正后的左目图像 |
| Right (rectified) | 校正后的右目图像 |
| Disparity (VPI) | VPI 计算的伪彩色视差图 |
| Anaglyph 3D | 红蓝立体图（戴 3D 眼镜可看） |

## 命令行参数

```
相机:
  --cam0 INT          左目 sensor-id (默认: 0)
  --cam1 INT          右目 sensor-id (默认: 1)
  --width INT         宽度 (默认: 1280)
  --height INT        高度 (默认: 720)
  --fps INT           帧率 (默认: 30)
  --rotate0 INT       左目旋转角度 0/90/180/270 (默认: 180)
  --rotate1 INT       右目旋转角度 (默认: 180)
  --swap-lr           交换左右 (默认: 开)
  --no-swap-lr        不交换

标定:
  --calib PATH        stereo_calib.npz 路径

VPI:
  --max-disparity INT 最大视差, 16 的倍数 (默认: 128)
  --vpi-backend STR   CUDA / PVA / OFA / CPU (默认: CUDA)

HTTP 服务:
  --host STR          监听地址 (默认: 0.0.0.0)
  --port INT          端口 (默认: 8080)
  --jpeg-quality INT  JPEG 质量 30-95 (默认: 80)
```

## VPI 后端选择

| 后端 | 特点 | 推荐场景 |
|------|------|----------|
| **CUDA** | 精度最高，参数可调 | 默认首选 |
| **OFA** | Orin 专有硬件引擎，几乎不占 GPU | 需要 GPU 同时跑推理 |
| **PVA** | 固定功能加速器 | 低功耗场景 |
| **CPU** | 最慢，仅供对比 | 调试 |

## 为什么不用 ROS / Isaac ROS？

| 问题 | 实际情况 |
|------|----------|
| Isaac ROS NITROS 包 | 你系统上没装，只有 Docker 镜像 (27 GB) |
| Docker 内运行 | 需要挂载 CSI 设备、X11、额外配置，复杂度远高于收益 |
| ROS 2 开销 | 对于纯双目深度估计，ROS 的 pub/sub 框架完全多余 |

VPI 直接调硬件，跟 Isaac ROS 里 `EssDisparityNode` 底层用的是同一套加速器。
区别只是少了 ROS 消息传输那层——对你这个场景来说，那层是纯开销。

---

## 双目标定工具 (`stereo_calibrate.py`)

标定质量直接决定深度图效果。旧标定数据 `reproj_error=53.85` 导致图像全黑。

### 标定流程

```
打印棋盘格 → capture（Web 采集 25+ 对） → calibrate（自动剔除离群值） → verify（极线验证）
```

### 快速开始

```bash
# 一步到位：采集 + 标定 + 验证
python3 stereo_calibrate.py all

# 或者分步操作
python3 stereo_calibrate.py capture        # 在浏览器中采集图像对
python3 stereo_calibrate.py calibrate      # 计算标定参数
python3 stereo_calibrate.py verify         # 查看校正结果

# 使用 ChArUco 板（还是用的棋盘格默认就行）
python3 stereo_calibrate.py all --board-type charuco
```

采集时浏览器打开 `http://<jetson-ip>:8080`，会看到双目画面 + 角点检测标记。

### 旧版 5 大问题 → 全部修复

| 问题 | 旧版 | 新版 |
|------|------|------|
| 远程不可用 | `cv2.imshow` 依赖 X11 | Web 界面，SSH 下可用 |
| 模型过拟合 | `CALIB_RATIONAL_MODEL` 8 系数 | 标准 5 系数畸变模型 |
| 采集门槛低 | `min_corners=6` 接受垃圾检测 | 棋盘格强制全部角点/ChArUco ≥12 |
| 无离群值过滤 | 全部图像对直接用 | Per-pair 重投影误差剔除 (阈值 1.5px) |
| 无结果验证 | 盲信数字 | Web 极线可视化，一眼看出标定好坏 |

### 标定板要求

默认配置：**9×6 棋盘格**，格子边长 25mm

- 建议 A3 打印，贴平在硬板上
- 采集时从多个角度、距离、位置拍摄
- 至少 25 对（程序达标后自动停止）
- 避免反光、模糊、标定板弯曲

### 标定质量判断

| reproj_error | 评级 | 含义 |
|--------------|------|------|
| < 0.5 | EXCELLENT | 标定极好 |
| 0.5 - 1.0 | GOOD | 正常水平 |
| 1.0 - 2.0 | ACCEPTABLE | 勉强可用 |
| > 2.0 | POOR | 需要重新采集 |

### 标定参数

```
标定板:
  --board-type          chessboard / charuco (默认: chessboard)
  --board-cols INT      内角点列数 (默认: 9)
  --board-rows INT      内角点行数 (默认: 6)
  --square-size FLOAT   格子边长·米 (默认: 0.025)

采集:
  --min-pairs INT       最少采集对数 (默认: 25)
  --no-auto             关闭自动采集，仅手动
  --stability-frames    连续检测帧数才保存 (默认: 8)
  --capture-interval    两次保存最小间隔·秒 (默认: 1.5)

质量:
  --max-reproj-error    per-pair 剔除阈值 (默认: 1.5)
```

### 输出格式

`calib_data/stereo_calib.npz` 包含：

| 字段 | 说明 |
|------|------|
| `camera_matrix_left/right` | 内参矩阵 3×3 |
| `dist_coeffs_left/right` | 畸变系数 (5 个) |
| `R, T, E, F` | 外参：旋转、平移、本质、基础矩阵 |
| `R1, R2, P1, P2, Q` | 校正参数 |
| `map1x, map1y, map2x, map2y` | 预计算 remap 映射表 |

与 `stereo_depth.py` 完全兼容，标定完直接用：

```bash
python3 stereo_depth.py --calib calib_data/stereo_calib.npz
```
