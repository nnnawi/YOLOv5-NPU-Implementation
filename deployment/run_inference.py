import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os 

# Load Vx delegate

os.environ['VIV_VX_CACHE_BINARY_GRAPH_DIR'] = os.getcwd()
os.environ['VIV_VX_ENABLE_CACHE_GRAPH_BINARY'] = '1'

ext_delegate= '/usr/lib/libvx_delegate.so'
ext_delegate= [ tflite.load_delegate(ext_delegate)]

delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so')
interpreter = tflite.Interpreter(
    model_path='/yolov5s/model/yolov5s_quant_1.tflite',
    experimental_delegates=[delegate]
)

# Verify delegate
print("Delegate Details:", delegate)
print("Delegate Type:", type(delegate))

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Verify hardware acceleration
print("\nHardware Acceleration:")
delegate_info = input_details[0].get('delegate', 'CPU/Default')
print("Active Delegate:", delegate_info)
print("Vx Delegate Active:", 'vx' in str(delegate).lower())
print("/n")

print(str(delegate).lower())

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.uint8)  # [0, 255]
    return np.expand_dims(image, axis=0).astype(np.int8)  # Cast to int8

def parse_detection_results(output_data, conf_threshold=0.5):
    """Parse and filter detection results"""
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        # ... (rest of class names)
    ]
    
    filtered_detections = []
    
    for detection in output_data[0]:
        if len(detection) >= 6:
            confidence = detection[4]
            class_id = int(detection[5])
            
            if confidence > conf_threshold:
                class_name = class_names[class_id]
                filtered_detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x) for x in detection[:4]]
                })
    
    return filtered_detections

def detect(image_path, conf_threshold=0.5):
    """Run object detection on an image"""
    input_data = preprocess_image(image_path)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return parse_detection_results(output_data, conf_threshold)

# Main execution
image_path = '/yolov5s/input/zidane.jpg'
results = detect(image_path)

# Print results
print("\nDetected Objects:")
for detection in results:
    print(f"- {detection['class']} (Confidence: {detection['confidence']:.2f})")
    print(f"  Bounding Box: {detection['bbox']}")