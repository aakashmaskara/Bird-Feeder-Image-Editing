# Importing necessary libraries
import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms

# Constants assumed
PROMPT = "a realistic squirrel or chipmunk sitting on a tree branch or near a bird feeder"
IMAGE_SIZE = 512
OUTPUT_SUFFIX = "-NowWithSquirrels.jpg"

# Loading pipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to("cpu")

# Resizing images
def resize_to_512(image):
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)

# Main function
def substituteSquirrels():
    image_filenames = [f"origIm{i}.jpg" for i in range(1, 21)]
    existing_images = [f for f in image_filenames if os.path.exists(f)]

    if len(existing_images) != 20:
        print(f"Expected 20 images, found {len(existing_images)}")
        return

    # Prepare masks
    mask_filenames = [f.replace(".jpg", "_mask.png") for f in existing_images]
    selected_pairs = [(img, mask) for img, mask in zip(existing_images, mask_filenames)
                      if os.path.exists(img) and os.path.exists(mask)]

    # Taking first 5 valid ones
    selected_pairs = selected_pairs[:5]

    for img_path, mask_path in selected_pairs:
        init_image = resize_to_512(Image.open(img_path).convert("RGB"))
        mask_image = resize_to_512(Image.open(mask_path).convert("L"))

        # Inpaint
        result = pipeline(prompt=PROMPT, image=init_image, mask_image=mask_image).images[0]

        # Save
        output_path = img_path.replace(".jpg", OUTPUT_SUFFIX)
        result.save(output_path)
        print(f"Squirrel inserted: {output_path}")

substituteSquirrels()