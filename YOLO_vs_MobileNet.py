import cv2
import numpy as np
import time

# Load class labels
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set up YOLO model
def load_yolo():
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Set up MobileNet-SSD model
def load_mobilenet_ssd():
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    return net

# Function to process frame with YOLO
def yolo_detect(net, output_layers, frame):
    height, width = frame.shape[:2]
    # Create blob and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Initialize lists
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

# Function to process frame with MobileNet-SSD
def mobilenet_ssd_detect(net, frame):
    height, width = frame.shape[:2]
    # Create blob and perform forward pass
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []
    class_ids = []

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            w = x1 - x
            h = y1 - y
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(idx)
    return boxes, confidences, class_ids

# Initialize models
yolo_net, yolo_output_layers = load_yolo()
mobilenet_ssd_net = load_mobilenet_ssd()

# Define video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deep copy of frame for each model
    frame_yolo = frame.copy()
    frame_mobilenet = frame.copy()

    # YOLO Detection
    start_time = time.time()
    boxes_yolo, confidences_yolo, class_ids_yolo, indexes_yolo = yolo_detect(yolo_net, yolo_output_layers, frame_yolo)
    yolo_time = time.time() - start_time

    # Draw YOLO detections
    if len(indexes_yolo) > 0:
        for i in indexes_yolo.flatten():
            x, y, w, h = boxes_yolo[i]
            label = str(classes[class_ids_yolo[i]])
            confidence = confidences_yolo[i]
            color = (255, 0, 0)
            cv2.rectangle(frame_yolo, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_yolo, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # MobileNet-SSD Detection
    start_time = time.time()
    boxes_mobilenet, confidences_mobilenet, class_ids_mobilenet = mobilenet_ssd_detect(mobilenet_ssd_net, frame_mobilenet)
    mobilenet_time = time.time() - start_time

    # Draw MobileNet-SSD detections
    if len(boxes_mobilenet) > 0:
        for i in range(len(boxes_mobilenet)):
            x, y, w, h = boxes_mobilenet[i]
            label = str(classes[class_ids_mobilenet[i]])
            confidence = confidences_mobilenet[i]
            color = (0, 255, 0)
            cv2.rectangle(frame_mobilenet, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_mobilenet, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display latency
    cv2.putText(frame_yolo, f'YOLO Inference Time: {yolo_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame_mobilenet, f'MobileNet-SSD Inference Time: {mobilenet_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frames
    cv2.imshow('YOLO Detection', frame_yolo)
    cv2.imshow('MobileNet-SSD Detection', frame_mobilenet)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
