# Image Modifier

Image Modifier is a small Streamlit web app that combines two uploaded images when they appear to be overlapping views of the same object or scene.

The app follows the CSE 152A homework-style feature pipeline:

1. Detect corner-like features with Gaussian-smoothed second-moment matrices.
2. Score feature patches with NCC and SSD helpers adapted from the homework notebook.
3. Build candidate correspondences with NCC.
4. Use RANSAC to estimate a homography and reject inconsistent matches.
5. Stitch and blend the images only when enough RANSAC inliers indicate a real overlap.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Streamlit opens a local browser interface with two image upload boxes, matching controls, diagnostics, and a stitched output preview.

## Notes

- This project intentionally does not use the final naive matching function from the homework notebook.
- It uses RANSAC to decide whether the images overlap strongly enough to combine.
- Homography stitching works best when the two images share a mostly planar object/scene or when viewpoint changes are moderate.
