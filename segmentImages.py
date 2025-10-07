# Importing necessary libraries
from ultralytics import YOLO
import os
import cv2
import numpy as np

# Constants assumed
CONF_THRESHOLD = 0.7
BIRD_CLASS_ID = 14

# Loading YOLOv8 segmentation model
model = YOLO("yolov8x-seg.pt")

def segmentImages():
    filenames = [f"origIm{i}.jpg" for i in range(1, 21)]

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue

        results = model(filename)[0]

        if results.masks is None:
            print(f"No segmentation masks found in {filename}")
            continue

        original = cv2.imread(filename)
        height, width = original.shape[:2]
        final_mask = np.zeros((height, width), dtype=np.uint8)

        for i in range(len(results.boxes.cls)):
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])

            if cls_id == BIRD_CLASS_ID and conf >= CONF_THRESHOLD:
                mask = results.masks.data[i].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (width, height))
                final_mask = cv2.bitwise_or(final_mask, mask)

        if np.count_nonzero(final_mask) > 0:
            output_path = filename.replace(".jpg", "_mask.png")
            cv2.imwrite(output_path, final_mask)
            print(f"Mask saved : {output_path}")
        else:
            print(f"Bird not confidently detected in {filename}")
    
segmentImages()