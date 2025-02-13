import cv2
import numpy as np
from sort import Sort

def detect_objects(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    print("Detection outputs:")
    for out in outs:
        print("Out shape:", out.shape)  # Print the shape of each output layer
        print("Out contents:", out)  # Print the contents for debugging

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                if obj.ndim == 1 and len(obj) >= 7:  # Ensure obj is a 1D array with enough elements
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    confidence = obj[4]
                    scores = obj[5:]
                    class_id = np.argmax(scores)

                    if confidence > 0.3:  # Lowered confidence threshold
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    return class_ids, confidences, boxes

def draw_labels(frame, tracked_boxes, class_ids, classes, object_count):
    tracked_boxes = [list(map(int, box)) for box in tracked_boxes]
    detected_classes = set()

    for i, box in enumerate(tracked_boxes):
        x, y, w, h = box[:4]
        class_id = int(box[4])
        label = str(classes[class_id])

        if label not in detected_classes:
            detected_classes.add(label)
            if label in object_count:
                object_count[label] += 1
            else:
                object_count[label] = 1
            
            count = object_count[label]
            cv2.putText(frame, f"{label} {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, object_count

def main():
    cfg_path = "C:/Users/MK/Desktop/PYTHON/Python/yolov3.cfg"
    weights_path = "C:/Users/MK/Desktop/PYTHON/Python/yolov3.weights"
    names_path = "C:/Users/MK/Desktop/PYTHON/Python/coco.names"

    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()

    # Convert output_layers_indices to zero-based indexing
    output_layers_indices = output_layers_indices.flatten() - 1
    output_layers = [layer_names[i] for i in output_layers_indices]

    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(0)
    trackers = Sort()
    object_count = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_ids, confidences, boxes = detect_objects(frame, net, output_layers)

        if boxes:
            print("Detected boxes:", boxes)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            if len(indexes) > 0:
                boxes = [boxes[i[0]] for i in indexes.flatten()]
                tracked_boxes = trackers.update(np.array(boxes))
                frame, object_count = draw_labels(frame, tracked_boxes, class_ids, classes, object_count)
        
        cv2.imshow("Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
