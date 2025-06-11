import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO("model/capstone2.5_ncnn_model")

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional, adjust 128-as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform inference with YOLOv11n
    results = model.track(source=frame, conf=0.8, iou=0.45, persist=True)

    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow("YOLOv11n Real-Time Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()