# 请先看根目录 `ai_memo.md`

本目录当前主入口是 `segment_depth_norender.py`，用于 Jetson 上的双目视差 + YOLO 分割深度估计（无深度图渲染面板，仅保留 YOLO 预览和对象深度表格）。

## 当前实现特性

- 三进程架构：`CameraWorker`、`InferWorker`、`WebWorker`
- 采集：GStreamer `nvarguscamerasrc` 双路相机
- 校正：支持 `--rectify-backend vpi|cpu`（VPI 失败自动回退 CPU remap）
- 视差：VPI `stereodisp`（启动时强依赖 VPI，失败会阻止整体启动）
- 分割：Ultralytics YOLO，支持 `engine/pt` 运行时切换
- 深度统计：分割掩码腐蚀 + IQR 过滤，输出每个目标的 `depth_mm`
- Web：MJPEG 预览 + `/api/state` + `/api/set`（可在线调 `yolo_size/model_mode/lite_mode`）

## 目录关键文件

- `segment_depth_norender.py`：当前主程序（推荐）
- `segment_depth.py`：旧版流程
- `segment_depth_debug.py`：调试版本
- `stereo_calibrate.py`：双目标定工具
- `strategy_mono_with_verify.py`：单目策略实验脚本
- `strategy_stereo_stacked.py`：双目堆叠策略实验脚本
- `calib_data/stereo_calib.npz`：标定文件

## 运行环境

- Jetson + JetPack（含 CUDA / TensorRT / VPI）
- Python 3.10（项目虚拟环境建议）
- 依赖（最小）：

```bash
pip3 install ultralytics opencv-python numpy
```

## 快速启动

```bash
python3 segment_depth_norender.py
```

默认会尝试加载本目录模型（按优先级）：

- `yolo11n-seg-<size>.engine`（如 256/512/640）
- `yolo11n-seg-dyn.engine`
- `yolo11n-seg.engine`（仅 `size=640` 时安全使用）
- `yolo11n-seg.pt`

启动后访问：

- `http://<jetson-ip>:8088`

## 常用参数

### 相机与标定

- `--cam0/--cam1`：双目设备编号
- `--width/--height/--fps`：采集参数
- `--flip-method`：GStreamer 翻转模式
- `--swap-lr` / `--no-swap-lr`：左右交换
- `--calib <path>`：标定文件
- `--no-rectify`：关闭校正
- `--rectify-backend vpi|cpu`：校正后端

### 视差与分割

- `--vpi-backend CUDA|PVA|OFA|CPU`
- `--downscale`：视差计算缩放
- `--max-disparity`
- `--model`：模型路径（engine 或 pt）
- `--force-pt`：强制 pt 推理
- `--seg-size`：YOLO 输入尺寸（160~1280）
- `--conf`：置信度阈值
- `--max-det`：最大目标数
- `--lite-mode`：预览 1fps 模式（识别仍按全速）

### Web

- `--host`
- `--port`
- `--jpeg-quality`
- `--preview-width`

## Web 控制接口

- `GET /api/state`：获取当前状态与对象深度结果
- `GET /api/set?yolo=640`：切换 YOLO size
- `GET /api/set?model=engine` 或 `model=pt`：切换推理模式
- `GET /api/set?lite=1`：开启 1fps 预览模式
- `GET /yolo`：MJPEG 预览流

## 诊断与排障

- 启动日志会输出：
  - `CAM-PERF`（采集侧帧率）
  - `PERF / PERF-DETAIL / PERF-MODE-10S`（推理侧时延与分解）
  - `WEB-PERF`（编码与码率）
  - `HEALTH`（主进程心跳）
- 若出现 Argus/CUDA 内存异常，请参考仓库根目录：
  - `JETSON_内存清理与相机重置.md`
