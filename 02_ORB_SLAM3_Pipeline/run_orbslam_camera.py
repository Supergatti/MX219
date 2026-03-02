import cv2
import numpy as np
import time

def get_gst_pipeline(sensor_id=0, width=640, height=480, fps=30):
    """
    Generate GStreamer pipeline for IMX219 on Jetson.
    Using rotate=0 and swap_lr=False as per user instructions.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=True"
    )

def main():
    width = 640
    height = 480
    fps = 30

    print("Opening Cameras...")
    cap0 = cv2.VideoCapture(get_gst_pipeline(0, width, height, fps), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(get_gst_pipeline(1, width, height, fps), cv2.CAP_GSTREAMER)

    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both cameras.")
        return

    print("Cameras opened. Press 'q' to exit.")
    
    # Check if DISPLAY is available for cv2.imshow
    has_display = os.environ.get('DISPLAY') is not None
    if not has_display:
        print("Warning: No DISPLAY detected. imshow will be disabled. Running in headless mode.")

    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                print("Error: Failed to capture frames.")
                break

            if has_display:
                # In-memory alignment / concatenation for visualization
                combined = np.hstack((frame0, frame1))
                cv2.imshow("IMX219 Stereo (Left | Right)", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Headless mode: just log capture status or process frames
                # print("Captured frames...", end='\r')
                pass
    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
