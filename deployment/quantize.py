import tensorflow as tf
import os
import numpy as np
import cv2 as cv

#Path to the calibration dataset image's folder
dataset_dir = "/home/hlymly/Documents/yolov5/pfe/YOLOv5-NPU-Implementation-main/mycoco100/images/val2017"

def representative_dataset_gen():
    img_file = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir) if i.endswith('.jpg')]

    for img_path in img_file:
        img = cv.imread(img_path)
        # Resize image to the input size expected by the model (e.g., 640x640 for YOLOv5)
        img_resized = cv.resize(img, (640, 640))
        # Normalize image to range [0, 1] if required
        img_normalized = img_resized / 255.0
        # Convert to float32 if the model expects it
        img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)
        # Yield the preprocessed image
        yield [img_input]

def convert():
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/hlymly/Documents/yolov5/objectdetection/models/yolov5s_saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8  
                                           ]
    converter.allow_custom_ops = True   
    converter.inference_input_type = tf.float32 
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen
                                                                                            # converter.target_spec.supported_types = [tf.int8]

    # converter.experimental_new_converter = True
    # converter.experimental_new_quantizer = True
    

    tflite_quant_model = converter.convert()

    open('/home/hlymly/Documents/yolov5/objectdetection/models/yolov5s_quant_1.tflite', 'wb').write(tflite_quant_model)

    print("Quantization complete! - model.tflite")
convert()