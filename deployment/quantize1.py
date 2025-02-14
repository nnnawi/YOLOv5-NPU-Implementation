import tensorflow as tf
import os
import numpy as np
import cv2 as cv

#Path to the calibration dataset image's folder
dataset_dir = "/home/hlymly/Documents/yolov5/pfe/YOLOv5-NPU-Implementation-main/mycoco100/images/val2017"
saved_model_dir = "/home/hlymly/Documents/yolov5/objectdetection/models/yolov5s_saved_model"

def representative_dataset_gen():
    img_files = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir) if i.endswith('.jpg')]
    
    for img_path in img_files:
        img = cv.imread(img_path)
        img_resized = cv.resize(img, (640, 640))
        img_normalized = img_resized.astype(np.float32) / 255.0  # Cast to float32 first
        img_input = np.expand_dims(img_normalized, axis=0)
        yield [img_input]
def convert():
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.int8  # Model expects int8 inputs post-quantization
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    # Save the model

    open('/home/hlymly/Documents/yolov5/objectdetection/models/yolov5s_quant_2.tflite', 'wb').write(tflite_quant_model)
    print("Quantization complete! - model.tflite")

convert()