# Importing necessary libraries
from ultralytics import YOLO
import os

# Constants assumed
CONF_THRESHOLD = 0.8
BIRD_CLASS_ID = 14
EXPECTED_COUNT = 20
OUTPUT_COUNT = 5

# Loading YOLOv8 model
model = YOLO("yolov8s.pt")

# Defining function for importing images
def subSelectImages():
    # Generating filenames origIm1.jpg to origIm20.jpg
    image_filenames = [f"origIm{i}.jpg" for i in range(1, EXPECTED_COUNT + 1)]

    # Validating all files exist or not
    missing = [f for f in image_filenames if not os.path.exists(f)]
    if missing:
        print("Missing files : ", ", ".join(missing))
        return []

    selected = []

    # Detecting birds with confidence greater than 80%
    for path in image_filenames:
        results = model(path)
        detections = results[0].boxes

        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == BIRD_CLASS_ID and conf >= CONF_THRESHOLD:
                selected.append(path)
                break

        if len(selected) == OUTPUT_COUNT:
            break

    print("Selected images:")
    for f in selected[:OUTPUT_COUNT]:
        print(" -", os.path.basename(f))

    return selected[:OUTPUT_COUNT]
    
subSelectImages()
