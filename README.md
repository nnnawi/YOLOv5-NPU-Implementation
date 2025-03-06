# 🚀 Optimized YOLOv5 for NPU Deployment

## 📌 Overview
This repository is a **fork of Ultralytics' YOLOv5** with added tools for **pruning** and **quantizing** the model to run efficiently on an **NPU (Neural Processing Unit)**, specifically the **Phytec i.MX8+**. Our optimizations allow for significant performance gains while maintaining accuracy.

## 🔥 Key Features
- **Unstructured Post-Training Pruning**: Reduce model size while preserving accuracy.
- **Static Quantization**: Convert model to **int8** for NPU compatibility.
- **PyTorch & TensorFlow Pipeline**: Pruning is done in **PyTorch**, while **TensorFlow Lite** handles quantization.
- **Massive FPS Boost**: Achieves **2-3x** more FPS compared to CPU execution.
- **Compatible with YOLOv5s and YOLOv5m**: Ensuring efficiency on embedded devices.

## 🏗 Project Workflow
1. **Model Selection**
   - Chose **YOLOv5s** for initial experimentation due to its lightweight nature.
   - The final deployment was done on **YOLOv5m** for a balance between speed and accuracy.

2. **Dataset Selection**
   - Utilized **COCO 2017** dataset (123,287 images, 80 classes, 886,284 labels).
   - Evaluated baseline model performance before applying pruning & quantization.

3. **Performance Baseline Evaluation**
   - Measured precision, recall, and mean Average Precision (mAP50/mAP50-95) on 10000 validation images.
   - Conducted inference time analysis for single images and video streams.

### 🚀 Streamlining your model
- In the **yolov5** folder, use the **PFE_YOLOv5_Dashboard** to find all the tools required to streamline your model as you wish.

### 🚀 Deploying on NPU
- Deployed on **Phytec i.MX8+** with TFLite.
- Used **VX Delegate** for optimized inference.
- Performance: **Real-time inference with 2-3x FPS improvement over CPU**.

## 🚀 Results & Future Improvements
### ✅ Achievements
- Successfully **deployed YOLOv5m on the NPU**.
- Achieved **real-time inference** with minimal accuracy loss.
- Pruning & quantization provided **optimal speed/accuracy tradeoff**.

### 🔧 Challenges & Future Work
- **Detection inconsistencies**: Some noise in NPU predictions.
- **Bounding box accuracy variations**: Potential improvements via fine-tuning.
- **Exploring structured pruning** for better layer-level compression.
- **Investigating YOLOv5L/X models** for potential future optimization.

## 📜 License
This project is licensed under the MIT License.

---

🔥 **Supercharge your object detection with optimized YOLOv5 for NPUs!** 🚀
