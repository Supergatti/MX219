import cv2
import numpy as np
import time
import os
import threading
from flask import Flask, Response

app = Flask(__name__)
# 全局变量用于在线程间共享帧
global_frame = None
frame_lock = threading.Lock()

def get_gst_pipeline(sensor_id=0, width=640, height=480, fps=30, flip_method=2):
    """
    Generate GStreamer pipeline for IMX219 on Jetson.
    Using rotate=2 (180 deg) as per user instructions.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} bufapi-version=1 ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=True max-buffers=1"
    )

def generate_frames():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is None:
                # 生成一个黑色背景等待图片
                blank_image = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank_image, "Waiting for Camera...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_image)
                frame_bytes = buffer.tobytes()
            else:
                # 编码为 JPEG
                ret, buffer = cv2.imencode('.jpg', global_frame)
                frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # 限制 Web 刷新率，避免占用过多 CPU

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_app():
    # 允许所有 IP 访问，端口 5000
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

def main():
    global global_frame
    
    # 使用 640x480 (4:3) 以匹配 IMX219 的 1640x1232 模式，避免画面拉伸
    width = 640
    height = 480
    fps = 30  # 尝试恢复到 30fps
    flip_method = 2
    swap_lr = True

    # 初始化 ORB 特征提取器 (演示用)
    orb = cv2.ORB_create(nfeatures=1000)

    print("Opening Cameras...")
    # 尝试打开第一个摄像头
    cap0 = cv2.VideoCapture(get_gst_pipeline(0, width, height, fps, flip_method), cv2.CAP_GSTREAMER)
    
    if cap0.isOpened():
        print("Camera 0 (Left) opened successfully.")
    else:
        print("Error: Camera 0 failed to open.")

    print("Waiting 2 seconds before opening Camera 1...")
    time.sleep(2.0)  # 稍微缩短等待时间

    # 尝试打开第二个摄像头
    cap1 = cv2.VideoCapture(get_gst_pipeline(1, width, height, fps, flip_method), cv2.CAP_GSTREAMER)

    if cap1.isOpened():
        print("Camera 1 (Right) opened successfully.")
    else:
        print("Error: Camera 1 failed to open.")

    if not cap0.isOpened() and not cap1.isOpened():
        print("FATAL: Both cameras failed. Please restart nvargus-daemon.")
        return

    print("Cameras initialization complete.")
    
    # 获取本机 IP 地址提示用户
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
        s.close()
        print(f"\n==============================================")
        print(f"  Web Monitor Available at: http://{ip_addr}:5000")
        print(f"==============================================\n")
    except Exception:
        print("Could not detect IP, try http://<JETSON_IP>:5000")

    # 启动 Flask 线程
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Check if DISPLAY is available for cv2.imshow
    has_display = os.environ.get('DISPLAY') is not None
    if not has_display:
        print("Warning: No DISPLAY detected. Running in headless mode with Web Monitor.")

    try:
        frame_count = 0
        prev_time = time.time()
        
        while True:
            # 单目或双目处理逻辑
            frame_l = None
            frame_r = None
            
            if cap0.isOpened():
                ret0, frame_l = cap0.read()
                if not ret0: frame_l = None
            
            if cap1.isOpened():
                ret1, frame_r = cap1.read()
                if not ret1: frame_r = None

            if frame_l is None and frame_r is None:
                print("Error: No frames captured.")
                time.sleep(0.1)
                continue

            if swap_lr and frame_l is not None and frame_r is not None:
                frame_l, frame_r = frame_r, frame_l

            # 处理左图 (ORB 特征演示)
            if frame_l is not None:
                # 转换为灰度
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                # 提取特征点
                kp_l = orb.detect(gray_l, None)
                # 绘制特征点 (绿色)
                frame_l = cv2.drawKeypoints(frame_l, kp_l, None, color=(0, 255, 0), flags=0)
                cv2.putText(frame_l, f"Left: {len(kp_l)} feats", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 处理右图 (ORB 特征演示)
            if frame_r is not None:
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                kp_r = orb.detect(gray_r, None)
                frame_r = cv2.drawKeypoints(frame_r, kp_r, None, color=(0, 255, 0), flags=0)
                cv2.putText(frame_r, f"Right: {len(kp_r)} feats", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 拼接画面
            if frame_l is not None and frame_r is not None:
                combined = np.hstack((frame_l, frame_r))
            elif frame_l is not None:
                combined = frame_l
            elif frame_r is not None:
                combined = frame_r
            
            # 计算帧率
            curr_time = time.time()
            fps_val = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(combined, f"FPS: {fps_val:.1f}", (combined.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 更新全局帧给 Flask
            with frame_lock:
                global_frame = combined.copy()

            if has_display:
                cv2.imshow("ORB-SLAM3 Camera Preview", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Streaming... FPS: {fps_val:.1f}", end='\r')
 
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if cap0.isOpened(): cap0.release()
        if cap1.isOpened(): cap1.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
