# Image Modifier

Image Modifier is a small Streamlit web app for practical image workflows:

- AI image combination with feature matching and RANSAC stitching.
- Manual image combining, top-bottom or left-right.
- Image resizing.
- Person extraction into a transparent PNG cutout with Segment Anything.

The app follows the CSE 152A homework-style feature pipeline:

1. Detect corner-like features with Gaussian-smoothed second-moment matrices.
2. Score feature patches with NCC and SSD helpers adapted from the homework notebook.
3. Build candidate correspondences with NCC.
4. Use RANSAC to estimate a homography and reject inconsistent matches.
5. Stitch and blend the images only when enough RANSAC inliers indicate a real overlap.

The extract tool assumes the uploaded photo is centered around a person and uses a local Segment Anything checkpoint to separate the subject from the background.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The requirements file uses the PyTorch CUDA 12.8 wheel index so the local SAM extractor can use an NVIDIA GPU when available.

## Run

```powershell
streamlit run app.py
```

Streamlit opens a local browser interface with a home page for choosing the available tools.

## Notes

- This project intentionally does not use the final naive matching function from the homework notebook.
- It uses RANSAC to decide whether the images overlap strongly enough to combine.
- Homography stitching works best when the two images share a mostly planar object/scene or when viewpoint changes are moderate.
- The extract tool is local. It checks CUDA first, uses CPU only as a fallback, expects `models/sam_vit_b_01ec64.pth` to exist, and may need prompt-margin tuning on difficult backgrounds.
- The znelchar editor has been split into its own standalone project at `E:\Znelchar File Editor`.
