# IMX219 摄像头坐标系与方向定义

## 坐标系定义
- **Z轴**: 水平向前（镜头光轴指向前方）。
- **Y轴**: 竖直向下（图像垂直方向）。
- **X轴**: 
  - **用户视角**（面对镜头时）：水平向左。
  - **相机视角**（图像坐标系）：水平向右。
  - 符合标准右手坐标系。

## 画面修正
- **现状**: 默认输出画面是倒置的（旋转了 180 度）。
- **修正方法**: 在 GStreamer 管道的 `nvvidconv` 插件中设置 `flip-method=2`。
  - `flip-method=0`: 不旋转（默认）
  - `flip-method=2`: 旋转 180 度（上下颠倒 + 左右颠倒 = 中心对称旋转）

## 涉及文件
- `02_ORB_SLAM3_Pipeline/run_orbslam_camera.py`
- `01_Stereo_Depth_Baseline/segment_depth_debug.py`
- `01_Stereo_Depth_Baseline/segment_depth.py`
- `01_Stereo_Depth_Baseline/stereo_calibrate.py`
