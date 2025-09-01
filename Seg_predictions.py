import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("best (1).pt")

image_path = "test3.jpg"

results = model(image_path, conf=0.5)

original_image = cv2.imread(image_path)

# Draw the boxes manually
for result in results:
    for box in result.boxes:
        xyxy = box.xyxy.numpy()[0]  
        x1, y1, x2, y2 = map(int, xyxy)  
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
result_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(result_image)
plt.axis('off')  
plt.show()
