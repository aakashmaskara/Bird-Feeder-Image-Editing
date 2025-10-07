# Importing necessary libraries
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Constants assumed
PROMPT = "a realistic backyard scene showing squirrels and birds eating seeds together at a bird feeder, natural lighting, photo taken with a DSLR camera"
NEGATIVE_PROMPT = "blurry, low resolution, distorted, text, watermark, double heads, unnatural colors"

SEED = 42
NUM_IMAGES = 5
OUTPUT_PREFIX = "generatedImage"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Loading pipeline
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo",torch_dtype=DTYPE,safety_checker=None).to(DEVICE)

# Main function
def generateBirdFeederImagesFromText():
    for i in range(NUM_IMAGES):
        print(f"Generating image {i+1}")
        generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)

        image = pipeline(prompt=PROMPT,negative_prompt=NEGATIVE_PROMPT,num_inference_steps=25,guidance_scale=3.5,generator=generator).images[0]

        image = image.resize((768, 768), Image.LANCZOS)

        filename = f"{OUTPUT_PREFIX}_{i+1}.png"
        image.save(filename)
        print(f"Saved: {filename}")

        
generateBirdFeederImagesFromText()