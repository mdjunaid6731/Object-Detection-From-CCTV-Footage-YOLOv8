from ultralytics import YOLO
import cv2
import cvzone

# Open the default camera
capture = cv2.VideoCapture(0)

# Set the width and height of the video frame to display
capture.set(3, 1280)
capture.set(4, 720)

# Start an infinite loop to continuously capture frames from the webcam
while True:
    success, img = capture.read()   # Read a frame from the webcam
    cv2.imshow("Webcam", img)   # Display the captured frame in a window named "Webcam"
    cv2.waitKey(1)   # Wait for 1 millisecond and check if a key is pressed



