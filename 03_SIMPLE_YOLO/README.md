# 弹幕 YOLO（单文件版）

这是一个可直接运行的单文件演示：`main.py`

## 运行

```bash
cd /home/jetson/Desktop/MX219/03_Isaac_ROS_VSLAM
python3 main.py
```

按 `q` 或 `ESC` 退出。

## VSCode 一键运行

在 VSCode 打开 `main.py` 后，直接点右上角 `Run Python File` 即可。

## 可选环境变量

- `YOLO_CAM_INDEX`：摄像头索引，默认 `0`
- `YOLO_MODEL`：模型路径，默认自动尝试：
  - `/home/jetson/Desktop/MX219/yolo11n.pt`
  - `/home/jetson/Desktop/MX219/yolo11n-seg.pt`
  - `yolo11n.pt`（自动下载）
- `YOLO_CONF`：置信度阈值，默认 `0.35`
- `YOLO_IMGSZ`：推理尺寸，默认 `640`
- `YOLO_INFER_EVERY`：每 N 帧推理一次，默认 `1`
- `DANMAKU_MAX`：弹幕最大条数，默认 `22`
- `DANMAKU_LANES`：弹幕轨道数，默认 `8`

## 依赖

若缺包，请安装：

```bash
pip3 install ultralytics opencv-python
```
