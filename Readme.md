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
### Dataset Configuration
The quantization process uses a subset of the COCO dataset:
- **Dataset Source**: COCO val2017
- **Sample Size**: First 100 images
- **Class Configuration**: Using COCO128 class definitions
- **Class Categories**: 80 classes including person, bicycle, car, etc. (as defined in COCO128)

### Calibration Dataset Setup
1. Download the COCO val2017 dataset
2. Extract the first 100 images for calibration
3. Ensure annotations follow COCO format
4. Verify class mappings match COCO128 configuration

### Quantization Scripts Overview
Our quantization approach involves two main scripts with distinct behaviors:

1. **`quantize.py`**:
   - Passes CPU benchmarking
   - Does not quantize inputs/outputs
   - Not compatible with NPU
   - Uses calibration dataset for determining quantization parameters
   - Workflow Partner: `run_inference.py`

2. **`quantize1.py`**:
   - Works with NPU
   - Introduces parasite inferences
   - Uses the same calibration dataset
   - Supports full INT8 quantization including inputs/outputs
   - Workflow Partner: `run_inference1.py`

### Quantization Process
For both scripts, the quantization workflow follows these steps:
1. Load the calibration dataset (first 100 images from COCO val2017)
2. Preprocess images according to YOLOv5 requirements
3. Run calibration to determine optimal quantization parameters
4. Apply quantization to the model
5. Validate quantized model performance

### Current Status
**Latest Achievement**: Successfully achieved inference on the target board, but experiencing parasite detections during NPU execution.
      We found on the NXP forum people who encountered this same issue [link to the ticket](https://community.nxp.com/t5/i-MX-Processors/Yolov5-Tflite-CPU-vs-VX-delegate-NPU/td-p/1557873) 

### Dataset References
- COCO128 Class Configuration: [GitHub - ultralytics/yolov5/data/coco128.yaml](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)
- Full COCO Dataset: [Kaggle - COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

## Deployment Process
### Board Directory Structure
The following directory structure should be maintained on the target board:
```
yolov5s/
├── model/
│   └── yolov5s_quant_*.tflite    # Quantized model files
├── img/
│   └── zidane.jpg                # Test images
├── output/                       # Directory for inference results
```

### File Transfer Methods
1. **SCP (Secure Copy)**:
   ```bash
   scp -r /path/to/local/yolov5s username@remote_host:/path/to/destination/
   ```

2. **RSync (Synchronization)**:
   ```bash
   rsync -avz /path/to/local/yolov5s/ username@remote_host:/path/to/destination/yolov5s/
   `
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


