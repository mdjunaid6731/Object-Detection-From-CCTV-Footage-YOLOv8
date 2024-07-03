from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Open the default camera
#capture = cv2.VideoCapture(0)
# Set the width and height of the video frame to display
#capture.set(3, 1280)
#capture.set(4, 720)

capture = cv2.VideoCapture("../Videos/cars.mp4") # For Video

model = YOLO('../YOLO_Weights/yolov8n.pt')  # Load the yolo model with weights

classNames = ["person" , "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "pen", "fan"]

# Start an infinite loop to continuously capture frames from the webcam
while True:
    success, img = capture.read()   # Read a frame from the webcam
    results = model(img, stream = True)  # Pass the captured frames to model for r in results:
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding Box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1  # Calculate the width and height of bounding box
            cvzone.cornerRect(img, (x1, y1, w, h),l=15, t=5, rt=2, colorR=(10,255,10), colorC=(0,0,255 )) # cornerRectangle with custom colours
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            # Display Class Name and Confidence
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1,colorR=(255, 0, 0))

    cv2.imshow("Webcam", img)   # Display the captured frame in a window named "Webcam"
    cv2.waitKey(1)   # Wait for 1 millisecond and check if a key is pressed



