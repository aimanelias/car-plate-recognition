# DeepStream LPR: Multi-Model Vehicle and License Plate Recognition

A comprehensive DeepStream pipeline for vehicle, license plate, and brand logo detection using YOLO-based models. Supports both C++ and Python bindings, with Docker and native setup instructions.

---


## Project Overview
This project provides a DeepStream pipeline for real-time vehicle, license plate, and brand logo detection using multiple YOLO-based models. It supports both C++ and Python bindings and is designed for easy deployment with Docker or on native systems.

## Requirements
- **NVIDIA Driver:** 571.96
- **CUDA:** 12.8
- **TensorRT:** 10.8.0.43
- **DeepStream:** 7.1.0
- **GStreamer:** 1.20.3
- **Docker** (optional, for containerized setup)
- **Python 3.10+** (for Python binding)

## Models & Configurations
- **Models:**
  - `best_16.pt` — Car plate detection
  - `best_29.pt` — Car brand logo detection
  - `yolo11s.pt` — Vehicle detection (car, truck, motorbike, bus)
  - `best.pt` — License plate character recognition (LPR)
- **Config Files:**
  - `config_infer_primary_yolo11.txt` — Vehicle detection
  - `config_infer_yolo_brand.txt` — Brand logo detection
  - `config_infer_yolo.txt` — Car plate detection
  - `config_infer_yoloLP.txt` — LPR (edit to add your model)
- **Label Files:**
  - `labels_default.txt` — Default YOLO labels
  - `labels.txt` — Car plate detection labels
  - `labelsbrand.txt` — Brand logo labels
  - `labelsLP.txt` — LPR labels

## Setup Instructions

### Docker Setup
1. **Open PowerShell as Administrator:**
   ```sh
   wsl
   ```
2. **In WSL shell:**
   ```sh
   export DISPLAY=:0
   xhost +
   docker run --gpus all -it \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v /mnt/c/Users/<your-user>/Documents/Deepstream:/home/<user>/testdeepstream \
     --network host \
     nvcr.io/nvidia/deepstream:7.1-triton-multiarch
   ```
   *Adjust the volume path to your local folder as needed.*
3. **Inside the container:**
   ```sh
   apt update && apt install -y \
     libflac8 libmp3lame0 libavcodec58 libmpg123-0 \
     mjpegtools libpulse0 librivermax1
   ```

---

### C++ Binding
1. **Clone DeepStream-YOLO and build the custom plugin:**
   ```sh
   git clone https://github.com/marcoslucianops/DeepStream-Yolo
   cd DeepStream-Yolo
   export CUDA_VER=12.6
   make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
   ```
2. **(Optional) Convert YOLO (.pt) to ONNX (.onnx):**
   ```sh
   pip3 install onnx torch torchvision ultralytics
   python3 utils/export_yolo11.py -w /path/to/model.pt --dynamic
   # Example for LPR model:
   python3 utils/export_yolo11.py -w /path/to/best.pt -s 256 256 --dynamic
   ```
3. **Edit `deepstream_app_config.txt`:**
   - Set model engine and config file paths under `[secondary-gie?]`.
   - Example:
     ```ini
     model-engine-file=/home/<user>/testdeepstream/models/model_name.pt.onnx_b1_gpu0_fp32.engine
     config-file=/home/<user>/testdeepstream/configs/config_infer_yoloLP.txt
     ```
   - Set video or RTSP source under `[source0]`:
     ```ini
     uri=file:///home/<user>/testdeepstream/hiv00020.mp4
     # or
     uri=rtsp://<user>:<password>@<ip>/axis-media/media.amp
     ```
4. **Edit `config_infer_yoloLP.txt`:**
   - Update `onnx-file` and `model-engine-file` paths to match your model.
5. **Run the C++ pipeline:**
   ```sh
   deepstream-app -c /home/<user>/testdeepstream/deepstream_app_config.txt
   ```

---

### Python Binding
1. **Install dependencies:**
   ```sh
   apt update && apt install -y \
     python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
     python3 python3-pip python3.10-dev cmake g++ build-essential \
     libglib2.0-dev libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf \
     automake libgirepository1.0-dev libcairo2-dev
   pip3 install build
   ```
2. **Clone required repositories:**
   ```sh
   cd /opt/nvidia/deepstream/deepstream-7.1/sources
   git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
   git clone https://github.com/marcoslucianops/DeepStream-Yolo
   cd DeepStream-Yolo
   export CUDA_VER=12.6
   make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
   ```
3. **(Optional) Convert YOLO (.pt) to ONNX (.onnx):**
   ```sh
   pip3 install onnx torch torchvision ultralytics
   python3 utils/export_yolo11.py -w /path/to/best.pt -s 256 256 --dynamic
   ```
4. **Build DeepStream Python Bindings:**
   ```sh
   cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/
   git submodule update --init
   python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse
   cd bindings/3rdparty/gstreamer/subprojects/gst-python/
   meson setup build
   cd build
   ninja
   ninja install
   ```
5. **Build and install pyds (Python bindings):**
   ```sh
   cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/bindings
   export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
   python3 -m build
   cd dist/
   pip3 install ./pyds-1.2.0-*.whl
   pip3 install cuda-python
   ```
6. **Configure and run the Python app:**
   ```sh
   cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/apps/deepstream-test2
   cp deepstream_test_2.py deepstream_test_2_custom.py
   nano deepstream_test_2_custom.py
   # Edit as needed for your use case
   ```
7. **Run the Python pipeline:**
   - For RTSP (live CCTV):
     ```sh
     python3 deepstream_test_2_custom.py 'rtsp://<user>:<password>@<ip>/axis-media/media.amp'
     ```
   - For video file:
     ```sh
     python3 deepstream_test_2_custom.py /home/<user>/testdeepstream/hiv00020.mp4
     ```

---

## Usage Notes
- Engine files are auto-generated on first run if not present. Ensure config files point to the correct engine/model files.
- Detection metadata is logged to `detection_log.jsonl`.
- To check if a file exists:
  ```sh
  ls /home/<user>/testdeepstream/hiv00020.mp4
  ```
- Example config options:
  - `operate-on-class-ids=2;3;5;7` (2=car, 3=motorcycle, 5=bus, 7=truck)
  - `network-mode=0` (0=FP32, 1=INT8, 2=FP16)
  - `cluster-mode=2` (NMS clustering)

## Troubleshooting
- Ensure all paths in config files are correct and accessible inside the container or environment.
- If engine files are missing, they will be generated on first run (ensure write permissions).
- For dependency issues, refer to the official DeepStream and DeepStream-YOLO documentation.
- For custom models, always use the provided export scripts for ONNX conversion.

---

*For more details, refer to the documentation in the `docs/` folder or the official DeepStream and DeepStream-YOLO repositories.*

