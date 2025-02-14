import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os 

# Load Vx delegate
os.environ['VIV_VX_CACHE_BINARY_GRAPH_DIR'] = os.getcwd()
os.environ['VIV_VX_ENABLE_CACHE_GRAPH_BINARY'] = '1'

delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so')
interpreter = tflite.Interpreter(
    model_path="/yolov5s/model/yolov5s_quant_2.tflite",  # Updated model path
    experimental_delegates=[delegate]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1] as in representative data
    
    # Get quantization parameters
    input_details = interpreter.get_input_details()[0]
    scale = input_details['quantization'][0]
    zero_point = input_details['quantization'][1]
    
    # Quantize the input
    image = image / scale + zero_point
    image = np.clip(image, np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    return np.expand_dims(image.astype(np.int8), axis=0)    





def parse_detection_results(output_data, conf_threshold=0.5, max_det=300):
    batch_size, num_anchors, _ = output_data.shape
    nc = 80  # Number of classes

    boxes = output_data[..., :4]
    scores = output_data[..., 4:4+nc]

    # Get top anchors based on max class score
    max_scores = np.amax(scores, axis=2)
    topk_indices = np.argpartition(-max_scores, max_det, axis=1)[:, :max_det]

    # Gather top-k entries
    batch_indices = np.arange(batch_size)[:, np.newaxis]
    boxes = boxes[batch_indices, topk_indices]
    scores = scores[batch_indices, topk_indices]

    # Flatten and get top-k across all classes
    scores_flat = scores.reshape(batch_size, -1)
    topk_flat = np.argpartition(-scores_flat, max_det, axis=1)[:, :max_det]

    # Convert indices
    anchor_indices = topk_flat // nc
    class_indices = topk_flat % nc

    # Gather final detections
    selected_boxes = boxes[batch_indices, anchor_indices]
    selected_scores = scores[batch_indices, anchor_indices, class_indices]
    selected_classes = class_indices.astype(np.float32)

    # Combine results
    detections = np.concatenate([
        selected_boxes,
        selected_scores[..., np.newaxis],
        selected_classes[..., np.newaxis]
    ], axis=-1)

    # Filter by confidence
    mask = detections[..., 4] > conf_threshold
    filtered = detections[mask]

    if filtered.size == 0:
        return []

    # Convert boxes to x1, y1, x2, y2 format for NMS
    x_center = filtered[:, 0]
    y_center = filtered[:, 1]
    w = filtered[:, 2]
    h = filtered[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    converted_boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = filtered[:, 4]
    classes = filtered[:, 5]

    # NMS function
    def nms(boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    # Apply NMS per class
    keep_indices = []
    iou_threshold = 0.5  # Adjust as needed

    for cls in np.unique(classes):
        cls_mask = (classes == cls)
        if not np.any(cls_mask):
            continue
        cls_boxes = converted_boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_original_indices = np.where(cls_mask)[0]
        if len(cls_boxes) == 0:
            continue
        nms_keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend(cls_original_indices[nms_keep].tolist())

    # Keep only the best detections after NMS
    filtered = filtered[keep_indices]

    # Format output
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush" ]

    results = []
    for det in filtered:
        x, y, w, h, conf, cls = det
        x = x * 640
        y = y * 640
        w = w * 640
        h = h * 640
        results.append({
            'class': class_names[int(cls)-1],
            'confidence': float(conf),
            'bbox': [float(x), float(y), float(w), float(h)]
        })
    return results

def detect(image_path, conf=0.5):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Dequantize output
    output = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]
    output = (output - output_zero_point) * output_scale  # Dequantize
    return parse_detection_results(output, conf)

# Execute
results = detect('/yolov5s/input/car.jpg')
print("\nDetected Objects:")
for obj in results:
    print(f"- {obj['class']} ({obj['confidence']:.2f}): {obj['bbox']}")

