from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from extract_person import SamSetupError, extract_person
from image_combiner import combine_images
from image_resizer import RESIZE_MODES, RESIZE_SCALES, resize_image, scaled_dimensions


IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]
PREVIEW_WIDTH = 280


def uploaded_image_to_array(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.asarray(image)


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(image_to_uint8(image)).save(output, format="PNG")
    return output.getvalue()


def show_image(image: np.ndarray, caption: str, width: int | None = None) -> None:
    if width is not None:
        st.image(image, caption=caption, width=width)
        return

    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def combine_manual_images(image1: np.ndarray, image2: np.ndarray, direction: str) -> np.ndarray:
    img1 = image_to_uint8(image1)
    img2 = image_to_uint8(image2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if direction == "top_bottom":
        canvas = np.full((h1 + h2, max(w1, w2), 3), 255, dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[h1 : h1 + h2, :w2] = img2
        return canvas

    if direction == "left_right":
        canvas = np.full((max(h1, h2), w1 + w2, 3), 255, dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1 : w1 + w2] = img2
        return canvas

    raise ValueError(f"Unknown combine direction: {direction}")


def go_to_page(page: str) -> None:
    st.session_state["page"] = page
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_home_page() -> None:
    st.title("Image Modifier")
    st.caption("Choose a tool to start.")
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            min-height: 4.5rem;
            width: 100%;
            font-size: 1.15rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("AI image combination", type="primary", key="home_ai_image_combination"):
            go_to_page("ai_image_combination")
    with col2:
        if st.button("image combiner", type="primary", key="home_image_combiner"):
            go_to_page("image_combiner")
    with col3:
        if st.button("resize", type="primary", key="home_resize"):
            go_to_page("resize")
    with col4:
        if st.button("extract", type="primary", key="home_extract"):
            go_to_page("extract")


def render_ai_image_combination_page() -> None:
    if st.button("Back to home", key="ai_back_home"):
        go_to_page("home")

    st.title("AI Image Combination")
    st.caption("Upload two images. The app detects feature matches, verifies overlap with RANSAC, and stitches them when the geometry is reliable.")

    with st.sidebar:
        st.header("Matching Controls")
        n_corners = st.slider("Corners per image", 80, 800, 300, 20)
        ncc_radius = st.slider("NCC patch radius", 3, 20, 6, 1)
        ncc_threshold = st.slider("NCC threshold", 0.1, 0.95, 0.65, 0.05)
        ransac_threshold = st.slider("RANSAC threshold (px)", 1.0, 12.0, 4.0, 0.5)
        min_inliers = st.slider("Minimum RANSAC inliers", 4, 60, 12, 1)
        min_inlier_ratio = st.slider("Minimum inlier ratio", 0.05, 0.8, 0.25, 0.05)

    col1, col2 = st.columns(2)
    image1 = None
    image2 = None

    with col1:
        upload1 = st.file_uploader("First image", type=IMAGE_TYPES, key="ai_image1")
        if upload1:
            image1 = uploaded_image_to_array(upload1)
            show_image(image1, "First image preview", width=PREVIEW_WIDTH)

    with col2:
        upload2 = st.file_uploader("Second image", type=IMAGE_TYPES, key="ai_image2")
        if upload2:
            image2 = uploaded_image_to_array(upload2)
            show_image(image2, "Second image preview", width=PREVIEW_WIDTH)

    if image1 is not None and image2 is not None:
        if st.button("Combine Images", type="primary", key="ai_combine"):
            with st.spinner("Detecting corners, matching features, and running RANSAC..."):
                result = combine_images(
                    image1,
                    image2,
                    n_corners=n_corners,
                    ncc_radius=ncc_radius,
                    ncc_threshold=ncc_threshold,
                    ransac_threshold=ransac_threshold,
                    min_inliers=min_inliers,
                    min_inlier_ratio=min_inlier_ratio,
                )

            if result.accepted:
                st.success(result.message)
                show_image(result.stitched, "Combined image")
                st.download_button(
                    "Download combined image",
                    data=image_to_png_bytes(result.stitched),
                    file_name="ai_combined_image.png",
                    mime="image/png",
                    key="ai_download",
                )
            else:
                st.warning(result.message)

            diag = result.diagnostics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidate matches", int(diag["candidate_matches"]))
            m2.metric("RANSAC inliers", int(diag["inliers"]))
            m3.metric("Inlier ratio", f"{diag['inlier_ratio']:.2f}")
            m4.metric("RANSAC threshold", f"{diag['ransac_threshold']:.1f}px")

            with st.expander("Detailed diagnostics"):
                st.json(diag)
                if result.homography is not None:
                    st.write("Homography mapping image 2 into image 1:")
                    matrix_text = np.array2string(result.homography, precision=4, suppress_small=True)
                    st.code(matrix_text, language="text")
    else:
        st.info("Choose two images to enable combining. Each uploaded image will preview immediately.")


def render_image_combiner_page() -> None:
    if st.button("Back to home", key="manual_back_home"):
        go_to_page("home")

    st.title("Image Combiner")
    st.caption("Upload two images and combine them directly without AI matching or alignment.")

    col1, col2 = st.columns(2)
    image1 = None
    image2 = None

    with col1:
        upload1 = st.file_uploader("Image 1", type=IMAGE_TYPES, key="manual_image1")
        if upload1:
            image1 = uploaded_image_to_array(upload1)
            show_image(image1, "Image 1 preview", width=PREVIEW_WIDTH)

    with col2:
        upload2 = st.file_uploader("Image 2", type=IMAGE_TYPES, key="manual_image2")
        if upload2:
            image2 = uploaded_image_to_array(upload2)
            show_image(image2, "Image 2 preview", width=PREVIEW_WIDTH)

    if image1 is None or image2 is None:
        st.info("Choose two images to enable top-bottom or left-right combining. Each uploaded image will preview immediately.")
        return

    top_bottom_col, left_right_col = st.columns(2)
    with top_bottom_col:
        combine_top_bottom = st.button("Combine top-bottom", type="primary", key="combine_top_bottom")
    with left_right_col:
        combine_left_right = st.button("Combine left-right", type="primary", key="combine_left_right")

    if combine_top_bottom or combine_left_right:
        direction = "top_bottom" if combine_top_bottom else "left_right"
        output = combine_manual_images(image1, image2, direction)
        caption = "Top-bottom combined image" if direction == "top_bottom" else "Left-right combined image"
        file_name = "combined_top_bottom.png" if direction == "top_bottom" else "combined_left_right.png"

        show_image(output, caption)
        st.download_button(
            "Download combined image",
            data=image_to_png_bytes(output),
            file_name=file_name,
            mime="image/png",
            key=f"manual_download_{direction}",
        )


def render_resize_page() -> None:
    if st.button("Back to home", key="resize_back_home"):
        go_to_page("home")

    st.title("Resize")
    st.caption("Upload one image, choose target dimensions, and preview the resized output.")

    upload = st.file_uploader("Input image", type=IMAGE_TYPES, key="resize_image")
    if not upload:
        st.info("Choose an image to enable resizing. The uploaded image will preview immediately.")
        return

    image = uploaded_image_to_array(upload)
    original_height, original_width = image.shape[:2]
    image_signature = (upload.name, original_width, original_height, getattr(upload, "size", None))
    if st.session_state.get("resize_image_signature") != image_signature:
        st.session_state["resize_image_signature"] = image_signature
        st.session_state["resize_width"] = original_width
        st.session_state["resize_height"] = original_height
        st.session_state["resize_mode"] = RESIZE_MODES[0]

    col1, col2 = st.columns(2)
    with col1:
        show_image(image, "Original image preview", width=PREVIEW_WIDTH)
    with col2:
        st.metric("Original width", f"{original_width}px")
        st.metric("Original height", f"{original_height}px")

    st.radio("Resize mode", RESIZE_MODES, horizontal=True, key="resize_mode")

    preset_cols = st.columns(len(RESIZE_SCALES))
    for preset_col, scale in zip(preset_cols, RESIZE_SCALES):
        label = f"x{scale:g}"
        with preset_col:
            if st.button(label, key=f"resize_scale_{scale:g}"):
                width, height = scaled_dimensions(original_width, original_height, scale)
                st.session_state["resize_width"] = width
                st.session_state["resize_height"] = height

    max_width, max_height = scaled_dimensions(original_width, original_height, 4.0)
    width_col, height_col = st.columns(2)
    with width_col:
        st.slider("Width", 1, max_width, key="resize_width")
    with height_col:
        st.slider("Height", 1, max_height, key="resize_height")

    target_width = int(st.session_state["resize_width"])
    target_height = int(st.session_state["resize_height"])
    output = resize_image(image, target_width, target_height, st.session_state["resize_mode"])
    output_height, output_width = output.shape[:2]

    st.write(f"Output size: {output_width}px x {output_height}px")
    show_image(output, "Resized image")
    st.download_button(
        "Download resized image",
        data=image_to_png_bytes(output),
        file_name="resized_image.png",
        mime="image/png",
        key="resize_download",
    )


def render_extract_page() -> None:
    if st.button("Back to home", key="extract_back_home"):
        go_to_page("home")

    st.title("Extract")
    st.caption("Upload a photo of a person and export a transparent PNG cutout.")

    with st.sidebar:
        st.header("Extraction Controls")
        rect_margin_percent = st.slider("SAM prompt border margin (%)", 1, 30, 8, 1, key="extract_margin")
        max_side = st.slider("Max SAM processing side", 512, 1536, 1024, 128, key="extract_max_side")
        feather_radius = st.slider("Edge feather", 0, 15, 3, 1, key="extract_feather")

    upload = st.file_uploader("Person photo", type=IMAGE_TYPES, key="extract_image")
    if not upload:
        st.info("Choose a photo to extract the person. The uploaded image will preview immediately.")
        return

    image = uploaded_image_to_array(upload)
    show_image(image, "Original image preview", width=PREVIEW_WIDTH)

    if st.button("Extract person", type="primary", key="extract_person"):
        with st.spinner("Segmenting the person and creating a transparent cutout..."):
            try:
                result = extract_person(
                    image,
                    rect_margin=rect_margin_percent / 100,
                    feather_radius=feather_radius,
                    max_side=max_side,
                )
            except SamSetupError as exc:
                st.error(str(exc))
                return

        st.success("Person extracted as a transparent PNG cutout.")
        col1, col2 = st.columns(2)
        with col1:
            show_image(result.rgba, "Transparent cutout")
        with col2:
            show_image(result.mask, "Alpha mask")

        st.download_button(
            "Download transparent PNG",
            data=image_to_png_bytes(result.rgba),
            file_name="person_cutout.png",
            mime="image/png",
            key="extract_download",
        )

        with st.expander("Extraction diagnostics"):
            st.json(result.diagnostics)


st.set_page_config(page_title="Image Modifier", layout="wide")

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "ai_image_combination":
    render_ai_image_combination_page()
elif st.session_state["page"] == "image_combiner":
    render_image_combiner_page()
elif st.session_state["page"] == "resize":
    render_resize_page()
elif st.session_state["page"] == "extract":
    render_extract_page()
else:
    render_home_page()
