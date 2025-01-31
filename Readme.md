# YOLOv5 Deployment and Quantization Guide

## Overview
This comprehensive guide walks you through deploying YOLOv5 object detection models on the NXP i.MX 8M Plus platform, focusing on model conversion, quantization, and NPU acceleration challenges.

## Table of Contents
1. [Project Context](#project-context)
2. [Requirements](#requirements)
3. [Environment Setup](#environment-setup)
4. [Model Preparation](#model-preparation)
5. [Conversion Strategies](#conversion-strategies)
6. [Quantization Workflow](#quantization-workflow)
7. [Deployment Process](#deployment-process)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Known Limitations](#known-limitations)

## Project Context
This project involves deploying a YOLOv5 object detection model on an embedded system with NPU (Neural Processing Unit) acceleration. The primary challenges include model conversion, quantization, and ensuring optimal inference performance across different compute environments.

## Requirements
### Hardware
- Target Platform: NXP i.MX 8M Plus
- Recommended Board Configuration: VX delegate compatible

### Software Environment
- **Operating System**: Ubuntu 20.04 LTS
- **Python Version**: 3.9.5 (Strictly Required)
- **Key Dependencies**:
  ```bash
  ultralytics
  tensorflow
  opencv-python
  pandas
  tflite
  onnx
  onnx-tf
  ```

## Environment Setup
### Python Version Management
Precise Python version management is crucial for dependency compatibility:

1. **Add Python Repository**:
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   ```

2. **Install Python 3.9**:
   ```bash
   sudo apt install python3.9
   ```

3. **Update Alternatives** (Optional but Recommended):
   ```bash
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
   ```

4. **Pip Configuration**:
   ```bash
   python3.9 -m ensurepip --upgrade
   python3.9 -m pip install --upgrade pip
   ```

## Model Preparation
### Conversion Methods
We provide two primary conversion strategies:

#### 1. Ultralytics Conversion Method
```python
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# Load pre-trained model
model = YOLO("yolov5n.pt")

# Clean model state
for m in model.model.model.modules():                     
    if hasattr(m, "num_batches_tracked"): 
        del m.num_batches_tracked

# Save and re-load cleaned model
model.ckpt.update(dict(model=model.model))
if "ema" in model.ckpt: 
    del model.ckpt["ema"]
model.save("model.pt")

# Export to TFLite with int8 quantization
model = YOLO("model.pt")
Detect.postprocess = lambda s,x,y,z: x
model.export(format="tflite", int8=True)
```

#### 2. NXP ONNX Conversion Method
```python
import torch

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Prepare dummy input
image_height = 640
image_width = 640
dummy_input = torch.randn(1, 3, image_height, image_width)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'yolov5s.onnx',
    verbose=False,
    opset_version=12,
    training=torch.onnx.TrainingMode.EVAL,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output']
)
```

## Quantization Workflow
### Quantization Scripts Overview
Our quantization approach involves multiple scripts with distinct behaviors:

1. **`quantization.py`**:
   - Passes CPU benchmarking
   - Does not quantize inputs/outputs
   - Not compatible with NPU
   - Workflow Partner: `run_inference.py`

2. **`quantization1.py`**:
   - Works with NPU
   - Introduces parasite inferences
   - Workflow Partner: `run_inference1.py`

### Current Status
**Latest Achievement**: Successfully achieved inference on the target board, but experiencing parasite detections during NPU execution.

## Deployment Process
### File Transfer Methods
1. **SCP (Secure Copy)**:
   ```bash
   scp -r /path/to/local/folder username@remote_host:/path/to/destination/
   ```

2. **RSync (Synchronization)**:
   ```bash
   rsync -avz /path/to/local/folder/ username@remote_host:/path/to/destination/
   ```

## Performance Optimization
### Benchmarking
Utilize TensorFlow Lite benchmark tool:
```bash
./benchmark_model \
  --graph=/yolov5s/model/yolov5model.tflite \
  --enable_op_profiling=true \
  --external_delegate_path=/usr/lib/libvx_delegate.so
```

## Troubleshooting
- Verify Python version (3.9.5)
- Confirm OpenCV and pandas installation
- Isolate eIQ quantization methods
- Check board-specific path configurations

## Known Limitations
- Potential bounding box detection inconsistencies
- Multiple false-positive detections
- Varied inference results across compute environments

## Important Reminders
1. Python 3.9 is mandatory
2. Install OpenCV and pandas explicitly
3. Isolate quantization method implementations
4. Paths are specific to target board configuration

## References
- [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [NXP Community Forums](https://community.nxp.com)
- [Phytec Documentation](https://www.phytec.de/cdocuments/)

---

**Project Status**: Active development, inference achieved with ongoing optimization
