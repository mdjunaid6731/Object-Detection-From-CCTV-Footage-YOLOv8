from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

capture = cv2.VideoCapture("../Videos/Camera_View.mp4") # For Video

# Get the total duration of the video in frames
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the frames per second (FPS) of the video
fps = capture.get(cv2.CAP_PROP_FPS)

model = YOLO('../YOLO_Weights/yolov8l.pt')  # Load the yolo model with weights

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

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [184, 487, 804, 489]
totalCount = []
# Start an infinite loop to continuously capture frames from the webcam
while True:
    success, img = capture.read()   # Read a frame from the webcam
    if not success:
        break  # If the video ends, exit the loop

    # Get the current frame number
    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

    # Calculate the elapsed time and total duration in seconds
    elapsed_time = current_frame / fps
    total_time = total_frames / fps

    # Format the elapsed time and total duration as MM:SS
    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    total_time_str = time.strftime("%M:%S", time.gmtime(total_time))

    # Display the timeline as "elapsed_time / total_time"
    timeline = f"{elapsed_time_str}/{total_time_str}"

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream = True)  # Pass the captured frames to model for r in results:
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding Box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1  # Calculate the width and height of bounding box

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "motorbike" or currentClass == "bus" or currentClass == "truck" or currentClass == "bicycle" and conf >= 0.2:
                #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, offset=3, thickness=1, colorR=(255, 0, 0))
                #cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=5, rt=5, colorR=(10, 255, 10), colorC=(0, 0, 255))  # cornerRectangle with custom colors
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (10,0,10), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=5, rt=2, colorR=(0, 255, 0), colorC=(0, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1, offset=5, thickness=1, colorR=(255, 0, 0))

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Print cx, cy, and limits to debug
        #print(f"ID: {id}, Center: ({cx}, {cy}), Limits: {limits}")

        if limits[0] < cx < limits[2] and limits[1]-5 < cy < limits[1]+5:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

            print(f'final count: {totalCount}')
                # Print cx, cy, and limits to debug
            print(f"ID: {id}, Center: ({cx}, {cy}), Limits: {limits}")

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (967, 141))
    # Add the timeline to the frame
    cv2.putText(img, timeline, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Video", img)   # Display the captured frame in a window named "Webcam"
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)   # Wait for 1 millisecond and check if a key is pressed


