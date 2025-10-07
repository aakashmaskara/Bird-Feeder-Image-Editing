# Importing necessary libraries 
import os
import cv2
import numpy as np

# Constants assumed
CONF_THRESHOLD = 0.5
BIRD_CLASS_ID = 14
MASK_SUFFIX = "_mask.png"
OUTPUT_SUFFIX = "-birdsRemoved.jpg"

# Loading segmentation model
from ultralytics import YOLO
model = YOLO("yolov8x-seg.pt")

# Main function
def removeBirds():
    # Generating filenames: origIm1.jpg to origIm20.jpg
    all_images = [f"origIm{i}.jpg" for i in range(1, 21)]
    selected = []

    # Detecting images with birds
    for path in all_images:
        if not os.path.exists(path):
            continue
        results = model(path)[0]
        if results.masks is None:
            continue
        for i in range(len(results.boxes.cls)):
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            if cls_id == BIRD_CLASS_ID and conf >= CONF_THRESHOLD:
                selected.append(path)
                break
        if len(selected) == 5:
            break

    if len(selected) < 5:
        print(f"Only found {len(selected)} images with birds")

    # Generating masks and inpaint
    for img_path in selected:
        print(f"\n Processing {img_path}")
        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        mask_final = np.zeros((height, width), dtype=np.uint8)

        results = model(img_path)[0]

        if results.masks is None:
            print(f"No masks found in {img_path}")
            continue

        for i in range(len(results.boxes.cls)):
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            if cls_id == BIRD_CLASS_ID and conf >= CONF_THRESHOLD:
                mask = results.masks.data[i].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (width, height))
                mask_final = cv2.bitwise_or(mask_final, mask)

        if np.count_nonzero(mask_final) == 0:
            print("No confident bird region to inpaint")
            continue

        # Inpaint using Telea algorithm
        inpainted = cv2.inpaint(image, mask_final, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Saving output
        output_path = img_path.replace(".jpg", OUTPUT_SUFFIX)
        cv2.imwrite(output_path, inpainted)
        print(f"Bird removed and saved as: {output_path}")

removeBirds()
