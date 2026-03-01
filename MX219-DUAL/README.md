# MX219 双目调用与对焦尝试（Jetson Orin）

## 1) 先决条件

- 已安装 OpenCV（Python）
- 推荐安装 `v4l-utils`（提供 `v4l2-ctl`），用于检测/设置对焦控件

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv v4l-utils
```

## 2) 运行双目预览（Argus / CSI）

> 适合 Jetson 上 2x15pin MIPI CSI 双目（常见 `sensor-id=0` 和 `sensor-id=1`）

```bash
python3 dual_mx219_focus.py --source argus --cam0 0 --cam1 1
```

默认参数已调整为更稳的高质量配置：`1640x1232 @ 30fps`（双路更不容易爆内存）。

若遇到 `InsufficientMemory` 或 `Failed to create CaptureSession`，请手动降级参数，例如：

```bash
python3 dual_mx219_focus.py --source argus --cam0 0 --cam1 1 --width 1920 --height 1080 --fps 30
```

如果你想强行拉满再试：

```bash
python3 dual_mx219_focus.py --source argus --cam0 0 --cam1 1 --width 3280 --height 2464 --fps 21
```

窗口内会显示：
- 实时清晰度分数
- 清晰度进度条
- 中心区域放大框（更容易观察是否对焦到位）

按键：
- `q`：退出
- `f`：对支持可编程焦点的镜头执行一次扫焦

## 2.1) 无界面（SSH/TTY）模式

如果你在纯终端（`DISPLAY` 为空）运行，OpenCV 无法弹窗，可使用：

```bash
python3 dual_mx219_focus.py --source argus --cam0 0 --cam1 1 --no-gui
```

此模式会持续输出两路清晰度分数，适合远程调试。

## 2.2) 必须弹窗时怎么跑

- 在 Jetson 本机桌面终端运行（推荐），而不是纯 tty。
- 若通过 SSH 远程弹窗，需要开启 X11 转发并安装本地 X Server（网络环境差时不稳定）。

## 3) 启动时自动尝试扫焦（若硬件支持）

先确认你的设备节点（示例 `/dev/video0`、`/dev/video1`）：

```bash
v4l2-ctl --list-devices
```

运行：

```bash
python3 dual_mx219_focus.py \
  --source argus --cam0 0 --cam1 1 \
  --focus-dev0 /dev/video0 --focus-dev1 /dev/video1 \
  --try-focus-on-start
```

## 4) 重要说明

- 大多数原生 IMX219 模组是**固定焦**，没有 `focus_absolute` 控件。
- 这种情况下程序会自动降级为：
  - 同时显示两路画面
  - 显示清晰度评分（Laplacian 方差）
  - 你可手动旋镜头后，观察评分增大来判断是否更清晰。

## 5) 网页预览服务器（在电脑浏览器看画面）

新增脚本：`web_stereo_server.py`。

在 Jetson 上启动：

```bash
python3 web_stereo_server.py --source argus --cam0 0 --cam1 1 --host 0.0.0.0 --port 8080
```

然后在你的电脑浏览器访问：

```text
http://<jetson_ip>:8080
```

可选参数：
- 降低带宽/负载：`--width 1280 --height 720 --fps 30`
- 控制网页码率：`--jpeg-quality 70`
- 关闭叠加层：`--no-overlay`

## 6) 双目标定（新脚本）

新增脚本：`stereo_calibrate.py`，支持采集 + 求解。

### 6.1 准备标定板

- 标定板类型：`ChArUco Board`
- 字典：`cv2.aruco.DICT_4X4_50`
- 网格：`squaresX=6, squaresY=9`
- 尺寸：`squareLength=0.025m, markerLength=0.018m`
- 打印在硬纸板上，尽量平整

脚本默认按以上参数创建 ChArUco 检测器。

### 6.2 采集图像对

```bash
python3 stereo_calibrate.py \
  --mode capture \
  --source v4l2 --cam0 /dev/video0 --cam1 /dev/video1 \
  --width 1640 --height 1232 --fps 30 \
  --squares-x 6 --squares-y 9 \
  --square-length 0.025 --marker-length 0.018 \
  --pairs 20 --output-dir calib_data
```

采集窗口中按键：
- `s`：保存一对图像（左右都检测到足够 ChArUco 角点才会保存）
- `q`：退出采集

采集建议：
- 不要只在画面中心，尽量覆盖四角与边缘
- 角度要变化（俯仰、偏航、轻微滚转）
- 距离要变化（近、中、远）
- 至少 15 对，推荐 20~30 对

### 6.3 求解标定参数

```bash
python3 stereo_calibrate.py \
  --mode calibrate \
  --squares-x 6 --squares-y 9 \
  --square-length 0.025 --marker-length 0.018 \
  --output-dir calib_data
```

输出文件：
- `calib_data/stereo_calib.npz`（内参、畸变、外参、校正映射）
- `calib_data/calibration_summary.json`（使用图像数与误差摘要）

### 6.4 一条命令完成采集+求解

```bash
python3 stereo_calibrate.py \
  --mode all \
  --source v4l2 --cam0 /dev/video0 --cam1 /dev/video1 \
  --squares-x 6 --squares-y 9 \
  --square-length 0.025 --marker-length 0.018 \
  --pairs 20 --output-dir calib_data
```
