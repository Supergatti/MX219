# ORB-SLAM3 Pipeline for IMX219 Dual Camera (Jetson Orin Nano)

## Prerequisites and Installation

To run ORB-SLAM3 on NVIDIA Jetson Orin Nano, you need to install several dependencies.

### 1. Update and Essential Build Tools
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libgoogle-glog-dev libgflags-dev libglew-dev libopencv-dev
```

### 2. Install Eigen3
```bash
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xf eigen-3.4.0.tar.gz
cd eigen-3.4.0 && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### 3. Install Pangolin (for Visualization)
```bash
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### 4. Build ORB-SLAM3
```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

## Running the IMX219 Stereo Pipeline

Ensure your camera configuration matches the `IMX219_Stereo.yaml` provided in this directory. 

1.  Start the GStreamer camera capture script:
    ```bash
    # If running over SSH without X11 forwarding, the script will run in headless mode.
    # To enable visualization, ensure you have a display connected or use X11 forwarding:
    # export DISPLAY=:0
    python3 run_orbslam_camera.py
    ```

2.  Feed the stereo images to ORB-SLAM3 (using the Stereo Example):
    ```bash
    ./Examples/Stereo/stereo_camera Vocabulary/ORBvoc.txt /path/to/IMX219_Stereo.yaml
    ```

**Note:** The physical baseline for the IMX219 Dual is ~60mm. In the configuration file, we have applied a correction factor of 0.55x to compensate for the scale error observed in previous experiments.
The final baseline used is ~33mm (0.033m).
```yaml
Camera.bf = Camera.fx * 0.033
```
