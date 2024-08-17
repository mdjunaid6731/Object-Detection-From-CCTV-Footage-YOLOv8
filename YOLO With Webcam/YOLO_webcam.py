from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Open the default camera
#capture = cv2.VideoCapture(0)
#capture.set(3, 1280) # Set the width and height of the video frame to display
#capture.set(4, 720)

capture = cv2.VideoCapture("../Videos/cars.mp4") # For Video

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

    # Add the timeline to the frame
    cv2.putText(img, timeline, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Webcam", img)   # Display the captured frame in a window named "Webcam"
    cv2.waitKey(1)   # Wait for 1 millisecond and check if a key is pressed



