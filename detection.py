import cv2
import numpy as np
import os
import pyttsx3
from collections import defaultdict

def load_yolo():
    cfg_path = "C:/Users/MK/Desktop/PYTHON/Python/yolov3.cfg"
    weights_path = "C:/Users/MK/Desktop/PYTHON/Python/yolov3.weights"
    names_path = "C:/Users/MK/Desktop/PYTHON/Python/coco.names"
    
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not os.path.isfile(names_path):
        raise FileNotFoundError(f"Names file not found: {names_path}")
    
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    
    if len(out_layers.shape) > 1:
        out_layers = out_layers.flatten()
    
    output_layers = [layer_names[i - 1] for i in out_layers]
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    
    if not isinstance(outs, list):
        outs = [outs]
    
    for out in outs:
        for detection in out:
            for obj in detection:
                if len(obj) > 5:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
    return class_ids, confidences, boxes

def draw_labels(img, class_ids, confidences, boxes, classes, object_count, previously_detected):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    
    if len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []
    
    current_detected = defaultdict(int)
    new_detections = defaultdict(int)

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        text = f"{label}"
        text_info = f"Coords: ({x}, {y}) - ({x + w}, {y + h})"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 30), font, 0.7, color, 2)
        cv2.putText(img, text_info, (x, y - 10), font, 0.7, color, 2)
        
        current_detected[label] += 1
        
        is_new_detection = True
        for (prev_x, prev_y, prev_w, prev_h) in previously_detected.get(label, []):
            if (abs(prev_x - x) < 50 and abs(prev_y - y) < 50):
                is_new_detection = False
                break
        
        if is_new_detection:
            new_detections[label] += 1
            previously_detected.setdefault(label, []).append((x, y, w, h))
    
    for obj, count in new_detections.items():
        object_count[obj] += count
        speak_object(obj, object_count[obj])

    return img

def speak_object(label, count):
    engine = pyttsx3.init()
    engine.say(f"Detected {count} {label}{'s' if count > 1 else ''}.")
    engine.runAndWait()

def main():
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    object_count = defaultdict(int)
    previously_detected = defaultdict(list)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        class_ids, confidences, boxes = detect_objects(frame, net, output_layers)
        frame = draw_labels(frame, class_ids, confidences, boxes, classes, object_count, previously_detected)
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
