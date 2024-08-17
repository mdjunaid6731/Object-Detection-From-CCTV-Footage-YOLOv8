from ultralytics import YOLO
import cv2

model = YOLO('../YOLO_Weights/yolov8l.pt')
results = model("images/schoolbus.jpg", show=True)
cv2.waitKey(0)