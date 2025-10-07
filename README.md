# Bird-Feeder Image Editing with YOLOv8 & Stable Diffusion

Object-centric computer vision + generative AI pipeline for bird-feeder photos:  
**select** images with birds, **segment** bird masks, **remove** birds (background
inpainting), **replace** birds with different birds, **substitute** birds with
squirrels/chipmunks, and **generate** synthetic feeder scenes from text prompts.

---

## Introduction

This project combines **YOLOv8** (detection/segmentation) with **Stable Diffusion**
(inpainting & text-to-image) to transform natural images of backyard feeders.
The code is modular—each task is a small, runnable script—so you can tweak thresholds,
prompts, or models without touching the rest of the pipeline.

Two common generative-editing risks and how we address them:
1. **Unrealistic composites** (floating subjects, wrong lighting/shadows) → use confident masks,
   conservative prompts, and context-aware inpainting.  
2. **Semantic drift** (adding/removing unintended objects) → steer with **negative prompts**,
   guidance scale, and **mask-only** edits.

---

## Business / Research Objectives

- Build a reproducible pipeline to **curate, segment, and edit** feeder images.  
- Compare classical background **inpainting** vs **generative** inpainting.  
- Produce **synthetic** feeder scenes for augmentation or design exploration.

---

## Dataset

- **Input:** 20 natural JPG images named `origIm1.jpg` … `origIm20.jpg`.  
- **Relevant subset:** first 5 images that confidently contain birds (auto-selected).  
- **Artifacts produced:** binary masks (`_mask.png`) and edited images for each task.

> If you have `mySampleImages.zip`, unzip it into the repository root so the filenames above are present.

---

## Analytical Approach (Scripts)

1) **Image Sub-selection** — `subSelectImages.py`  
   Detect birds with **YOLOv8s** and pick the first 5 images above a confidence threshold.
   Prints selected filenames to the console.

2) **Segmentation** — `segmentImages.py`  
   Segment bird regions with **YOLOv8x-seg**; save per-image binary mask
   `origImX_mask.png` (union of all bird instances).

3) **Remove Birds (Background Inpainting)** — `removeBirds.py`  
   Use **OpenCV Telea** inpainting to fill bird regions from surrounding context.
   Saves `origImX-birdsRemoved.jpg`.

4) **Replace Birds (Generative Inpainting)** — `replaceBirds.py`  
   Use **Stable Diffusion Inpainting** to draw **different birds** in the masked area.
   Saves `origImX-birdsReplaced.jpg`.

5) **Substitute Squirrels / Chipmunks** — `substituteSquirrels.py`  
   Use **Stable Diffusion Inpainting** to place a **squirrel or chipmunk**
   near the feeder in the masked area. Saves `origImX-NowWithSquirrels.jpg`.

6) **Text-to-Image Generation** — `generateBirdFeederImagesFromText.py`  
   Use **Stable Diffusion** (Turbo or base) to synthesize **new** feeder scenes with both
   birds and squirrels. Saves `generatedImage_1.png` … `generatedImage_5.png`.

---

## Tools & Libraries

- **Ultralytics YOLOv8** (detection & segmentation)  
- **Diffusers / Stable Diffusion** (inpainting & text-to-image)  
- **OpenCV** (classical Telea inpainting)  
- **PyTorch**, **Transformers**, **Accelerate**, **Pillow (PIL)**, **NumPy**

---

## How to Run

> A GPU is strongly recommended for the diffusion steps; CPU works with smaller image sizes or fewer steps.

### 1) Install dependencies
    pip install ultralytics torch torchvision diffusers transformers accelerate opencv-python pillow numpy

### 2) Prepare images
Unzip or place `origIm1.jpg` … `origIm20.jpg` in the repo root (or unzip `mySampleImages.zip` there).

### 3) Run the pipeline (task by task)
    # 1) Select 5 images with confident bird detections
    python subSelectImages.py

    # 2) Segment bird masks for all 20 images
    python segmentImages.py

    # 3) Remove birds (background inpaint on the 5 selected images)
    python removeBirds.py

    # 4) Replace birds with different birds (generative inpaint on the 5 selected)
    python replaceBirds.py

    # 5) Substitute squirrels/chipmunks (generative inpaint on the 5 selected)
    python substituteSquirrels.py

    # 6) Generate synthetic feeder scenes from text
    python generateBirdFeederImagesFromText.py

**Outputs (by convention):**
- Masks: `origImX_mask.png`  
- Background inpaint: `origImX-birdsRemoved.jpg`  
- Birds replaced: `origImX-birdsReplaced.jpg`  
- Squirrels inserted: `origImX-NowWithSquirrels.jpg`  
- Text-to-image: `generatedImage_#.png`

---

## Key Notes & Tips

- **Thresholds:** Adjust `CONF_THRESHOLD` in detection/segmentation to balance recall vs precision.  
- **Prompts:** Tune descriptive phrases and **negative prompts** to reduce artifacts.  
- **Size / Speed:** Generative steps often use **512×512** or **768×768**; reduce for CPU.  
- **Reproducibility:** Some scripts set seeds; vary seeds to increase diversity.

---

## Files in this Repository

| File | Purpose |
| --- | --- |
| `subSelectImages.py` | Pick first 5 images with confident bird detections (YOLOv8s). |
| `segmentImages.py` | Segment bird masks using YOLOv8x-seg; save `_mask.png` per image. |
| `removeBirds.py` | Remove birds via OpenCV Telea inpainting; save `-birdsRemoved.jpg`. |
| `replaceBirds.py` | Generative inpainting to add **different birds**; save `-birdsReplaced.jpg`. |
| `substituteSquirrels.py` | Generative inpainting to add **squirrels/chipmunks**; save `-NowWithSquirrels.jpg`. |
| `generateBirdFeederImagesFromText.py` | Text-to-image generation of feeder scenes; save `generatedImage_#.png`. |
| `Bird_Feeder_Image_Editing.pdf` | Project write-up: design, prompts, thresholds, and comparisons. |
| `mySampleImages.zip` | (Optional) Example input set (`origIm1.jpg` … `origIm20.jpg`). |

---

## Results (High-Level)

- Confident bird selection yields clean masks with reduced false positives.  
- Classical inpainting produces plausible backgrounds, while **generative inpainting**
  adds diverse **new birds** or **squirrels** with better context when prompts include feeder/location cues.  
- Text-to-image generation provides realistic feeder scenes suitable for augmentation.


## Author

**Aakash Maskara**  
*M.S. Robotics & Autonomy, Drexel University*  
Computer Vision | Generative AI

[LinkedIn](https://linkedin.com/in/aakashmaskara) • [GitHub](https://github.com/AakashMaskara)
