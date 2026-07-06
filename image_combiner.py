from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class CombineResult:
    accepted: bool
    message: str
    stitched: np.ndarray | None
    homography: np.ndarray | None
    candidate_matches: list[tuple[np.ndarray, np.ndarray, float]]
    inlier_matches: list[tuple[np.ndarray, np.ndarray, float]]
    inlier_mask: np.ndarray
    diagnostics: dict[str, float | int]


def gaussian2d(filter_size: int = 1, sig: float = 1.0) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel."""
    ax = np.arange(-filter_size // 2 + 1.0, filter_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)


def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient magnitude and direction, matching the notebook kernels."""
    kx = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]], dtype=float)
    ky = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]], dtype=float)

    gx = convolve2d_same(image, kx)
    gy = convolve2d_same(image, ky)
    g_mag = np.sqrt(np.square(gx) + np.square(gy))
    g_theta = np.degrees(np.arctan2(gy, gx))
    return g_mag, g_theta


def corner_detect(
    image: np.ndarray,
    n_corners: int,
    smooth_std: float = 2.0,
    window_size: int = 13,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect corner-like features using the notebook minor-eigenvalue method."""
    gray = to_gray_float(image)
    kx = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]], dtype=float)
    ky = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]], dtype=float)

    gx = convolve2d_same(gray, kx)
    gy = convolve2d_same(gray, ky)

    kernel = gaussian2d(window_size, smooth_std)
    c_ix2 = convolve2d_same(np.square(gx), kernel)
    c_iy2 = convolve2d_same(np.square(gy), kernel)
    c_ixiy = convolve2d_same(gx * gy, kernel)

    trace = c_ix2 + c_iy2
    determinant = c_ix2 * c_iy2 - np.square(c_ixiy)
    discriminant = np.maximum(np.square(trace) - 4 * determinant, 0)
    minor_eig_image = (trace - np.sqrt(discriminant)) / 2

    border = window_size // 2
    if border > 0:
        minor_eig_image[:border, :] = 0
        minor_eig_image[-border:, :] = 0
        minor_eig_image[:, :border] = 0
        minor_eig_image[:, -border:] = 0

    local_max = maximum_filter_same(minor_eig_image, size=window_size)
    nms = np.where((minor_eig_image == local_max) & (minor_eig_image > 0), minor_eig_image, 0)

    nonzero = np.flatnonzero(nms)
    if nonzero.size == 0:
        return minor_eig_image, np.empty((0, 2), dtype=int), 0.0

    take = min(n_corners, nonzero.size)
    ranked = nonzero[np.argpartition(nms.ravel()[nonzero], -take)[-take:]]
    ranked = ranked[np.argsort(nms.ravel()[ranked])[::-1]]
    rows, cols = np.unravel_index(ranked, nms.shape)
    corners = np.stack((cols, rows), axis=1).astype(int)
    threshold = float(nms.ravel()[ranked[-1]])
    return minor_eig_image, corners, threshold


def ssd_match(img1: np.ndarray, img2: np.ndarray, c1: np.ndarray, c2: np.ndarray, r: int) -> float:
    """Compute the SSD score for two square patches centered on x-y coordinates."""
    w1 = _patch(to_gray_array(img1), c1, r)
    w2 = _patch(to_gray_array(img2), c2, r)
    if w1 is None or w2 is None:
        return float("inf")
    return float(np.sum(np.square(w1 - w2)))


def ncc_match(img1: np.ndarray, img2: np.ndarray, c1: np.ndarray, c2: np.ndarray, r: int) -> float:
    """Compute NCC for two square patches centered on x-y coordinates."""
    w1 = _patch(to_gray_array(img1), c1, r)
    w2 = _patch(to_gray_array(img2), c2, r)
    if w1 is None or w2 is None:
        return -1.0

    w1_diff = w1 - np.mean(w1)
    w2_diff = w2 - np.mean(w2)
    denom = np.sqrt(np.sum(np.square(w1_diff))) * np.sqrt(np.sum(np.square(w2_diff)))
    if denom <= 1e-12:
        return -1.0
    return float(np.sum(w1_diff * w2_diff) / denom)


def combine_images(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    n_corners: int = 300,
    corner_window_size: int = 13,
    smooth_std: float = 2.0,
    ncc_radius: int = 6,
    ncc_threshold: float = 0.65,
    max_candidates: int = 250,
    ransac_threshold: float = 4.0,
    ransac_iterations: int = 1500,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.25,
    random_seed: int = 7,
) -> CombineResult:
    """Detect overlap with NCC candidates plus RANSAC, then stitch if accepted."""
    rgb1 = to_rgb_float(image1)
    rgb2 = to_rgb_float(image2)
    gray1 = to_gray_float(rgb1)
    gray2 = to_gray_float(rgb2)

    _, corners1, corner_threshold1 = corner_detect(gray1, n_corners, smooth_std, corner_window_size)
    _, corners2, corner_threshold2 = corner_detect(gray2, n_corners, smooth_std, corner_window_size)

    matches = build_candidate_matches(
        gray1,
        gray2,
        corners1,
        corners2,
        ncc_radius,
        ncc_threshold,
        max_candidates,
    )

    base_diag: dict[str, float | int] = {
        "corners_image_1": int(len(corners1)),
        "corners_image_2": int(len(corners2)),
        "corner_threshold_image_1": float(corner_threshold1),
        "corner_threshold_image_2": float(corner_threshold2),
        "candidate_matches": int(len(matches)),
        "ransac_threshold": float(ransac_threshold),
        "min_inliers": int(min_inliers),
        "min_inlier_ratio": float(min_inlier_ratio),
    }

    if len(matches) < 4:
        return CombineResult(
            accepted=False,
            message="Not enough NCC candidate matches to estimate overlap.",
            stitched=None,
            homography=None,
            candidate_matches=matches,
            inlier_matches=[],
            inlier_mask=np.zeros(len(matches), dtype=bool),
            diagnostics={**base_diag, "inliers": 0, "inlier_ratio": 0.0},
        )

    pts1 = np.array([m[0] for m in matches], dtype=float)
    pts2 = np.array([m[1] for m in matches], dtype=float)
    homography, inlier_mask, errors = ransac_homography(
        src_points=pts2,
        dst_points=pts1,
        threshold=ransac_threshold,
        iterations=ransac_iterations,
        random_seed=random_seed,
    )

    inlier_matches = [match for match, is_inlier in zip(matches, inlier_mask) if is_inlier]
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = float(inlier_count / len(matches)) if matches else 0.0
    diagnostics = {
        **base_diag,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error": float(np.median(errors[inlier_mask])) if inlier_count else float("inf"),
    }

    if homography is None or inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
        return CombineResult(
            accepted=False,
            message="No reliable overlap found. RANSAC did not find enough consistent matches.",
            stitched=None,
            homography=homography,
            candidate_matches=matches,
            inlier_matches=inlier_matches,
            inlier_mask=inlier_mask,
            diagnostics=diagnostics,
        )

    stitched = stitch_images(rgb1, rgb2, homography)
    return CombineResult(
        accepted=True,
        message="Reliable overlap found. The images were stitched into a combined output.",
        stitched=stitched,
        homography=homography,
        candidate_matches=matches,
        inlier_matches=inlier_matches,
        inlier_mask=inlier_mask,
        diagnostics=diagnostics,
    )


def build_candidate_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    corners1: np.ndarray,
    corners2: np.ndarray,
    r: int,
    ncc_threshold: float,
    max_candidates: int,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Use mutual best NCC matches as RANSAC candidates."""
    valid1 = _filter_patchable(corners1, img1.shape, r)
    valid2 = _filter_patchable(corners2, img2.shape, r)
    if len(valid1) == 0 or len(valid2) == 0:
        return []

    scores = np.full((len(valid1), len(valid2)), -1.0, dtype=float)
    for i, c1 in enumerate(valid1):
        for j, c2 in enumerate(valid2):
            scores[i, j] = ncc_match(img1, img2, c1, c2, r)

    row_best = np.argmax(scores, axis=1)
    col_best = np.argmax(scores, axis=0)
    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i, j in enumerate(row_best):
        score = scores[i, j]
        if score >= ncc_threshold and col_best[j] == i:
            candidates.append((valid1[i].copy(), valid2[j].copy(), float(score)))

    candidates.sort(key=lambda match: match[2], reverse=True)
    return candidates[:max_candidates]


def ransac_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    random_seed: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Estimate a homography from src to dst with RANSAC."""
    n_points = len(src_points)
    rng = np.random.default_rng(random_seed)
    best_mask = np.zeros(n_points, dtype=bool)
    best_h: np.ndarray | None = None
    best_errors = np.full(n_points, float("inf"), dtype=float)

    if n_points < 4:
        return None, best_mask, best_errors

    for _ in range(iterations):
        sample_indices = rng.choice(n_points, size=4, replace=False)
        h = estimate_homography(src_points[sample_indices], dst_points[sample_indices])
        if h is None:
            continue
        errors = reprojection_errors(h, src_points, dst_points)
        mask = errors <= threshold
        if np.sum(mask) > np.sum(best_mask):
            best_mask = mask
            best_h = h
            best_errors = errors

    if best_h is not None and np.sum(best_mask) >= 4:
        refined = estimate_homography(src_points[best_mask], dst_points[best_mask])
        if refined is not None:
            best_h = refined
            best_errors = reprojection_errors(best_h, src_points, dst_points)
            best_mask = best_errors <= threshold

    return best_h, best_mask, best_errors


def estimate_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray | None:
    """Estimate homography H where dst ~= H * src using normalized DLT."""
    if len(src_points) < 4 or len(dst_points) < 4:
        return None
    src_norm, t_src = _normalize_points(src_points)
    dst_norm, t_dst = _normalize_points(dst_points)

    rows = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)

    try:
        _, _, vt = np.linalg.svd(a)
        h_norm = vt[-1].reshape(3, 3)
        h = np.linalg.inv(t_dst) @ h_norm @ t_src
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        return None
    return h / h[2, 2]


def reprojection_errors(h: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    projected = project_points(h, src_points)
    return np.linalg.norm(projected - dst_points, axis=1)


def project_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack((points, np.ones(len(points))))
    projected = (h @ pts_h.T).T
    w = projected[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return projected[:, :2] / w


def stitch_images(image1: np.ndarray, image2: np.ndarray, h_image2_to_image1: np.ndarray) -> np.ndarray:
    """Warp image2 into image1 coordinates and average overlapping pixels."""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=float)
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=float)
    warped_corners2 = project_points(h_image2_to_image1, corners2)
    all_corners = np.vstack((corners1, warped_corners2))

    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    offset_x = -min(0, min_xy[0])
    offset_y = -min(0, min_xy[1])
    canvas_w = int(max_xy[0] + offset_x)
    canvas_h = int(max_xy[1] + offset_y)

    translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=float)
    h2_canvas = translation @ h_image2_to_image1

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=float)
    weights = np.zeros((canvas_h, canvas_w, 1), dtype=float)

    y0, x0 = offset_y, offset_x
    canvas[y0 : y0 + h1, x0 : x0 + w1] += image1
    weights[y0 : y0 + h1, x0 : x0 + w1] += 1

    warped2, mask2 = warp_rgb(image2, h2_canvas, canvas_h, canvas_w)

    canvas += warped2 * mask2
    weights += mask2
    combined = np.divide(canvas, np.maximum(weights, 1), out=np.zeros_like(canvas), where=weights > 0)
    return np.clip(combined, 0, 1)


def to_gray_float(image: np.ndarray) -> np.ndarray:
    arr = to_gray_array(image)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0, 1)


def to_gray_array(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale float while preserving the input intensity scale."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return arr


def convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small zero-padded 2D convolution helper used instead of SciPy."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    windows = sliding_window_view(padded, (kh, kw))
    flipped = kernel[::-1, ::-1]
    return np.einsum("ijxy,xy->ij", windows, flipped)


def maximum_filter_same(image: np.ndarray, size: int) -> np.ndarray:
    """Small zero-padded maximum filter helper used instead of SciPy."""
    radius = size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="constant")
    windows = sliding_window_view(padded, (size, size))
    return np.max(windows, axis=(-1, -2))


def warp_rgb(image: np.ndarray, h_to_canvas: np.ndarray, canvas_h: int, canvas_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-map an RGB image into a canvas with bilinear sampling."""
    inv_h = np.linalg.inv(h_to_canvas)
    yy, xx = np.indices((canvas_h, canvas_w), dtype=float)
    canvas_points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(canvas_h * canvas_w)))
    source = (inv_h @ canvas_points.T).T
    source_xy = source[:, :2] / np.maximum(np.abs(source[:, 2:3]), 1e-12) * np.sign(source[:, 2:3])
    xs = source_xy[:, 0]
    ys = source_xy[:, 1]

    h, w = image.shape[:2]
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = (xs - x0)[:, None]
    wy = (ys - y0)[:, None]
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    sampled = top * (1 - wy) + bottom * wy
    sampled[~valid] = 0

    warped = sampled.reshape(canvas_h, canvas_w, 3)
    mask = valid.reshape(canvas_h, canvas_w, 1).astype(float)
    return warped, mask


def to_rgb_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 4:
        alpha = arr[..., 3:4]
        if alpha.max(initial=0) > 1.0:
            alpha = alpha / 255.0
        rgb = arr[..., :3]
        if rgb.max(initial=0) > 1.0:
            rgb = rgb / 255.0
        arr = rgb * alpha + (1 - alpha)
    else:
        arr = arr[..., :3]
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0, 1)


def _patch(image: np.ndarray, center: Iterable[int], r: int) -> np.ndarray | None:
    x, y = np.asarray(center, dtype=int)
    if y - r < 0 or y + r + 1 > image.shape[0] or x - r < 0 or x + r + 1 > image.shape[1]:
        return None
    return image[y - r : y + r + 1, x - r : x + r + 1]


def _filter_patchable(corners: np.ndarray, image_shape: tuple[int, ...], r: int) -> np.ndarray:
    if len(corners) == 0:
        return np.empty((0, 2), dtype=int)
    corners = np.asarray(corners, dtype=int)
    h, w = image_shape[:2]
    mask = (
        (corners[:, 0] - r >= 0)
        & (corners[:, 0] + r + 1 <= w)
        & (corners[:, 1] - r >= 0)
        & (corners[:, 1] + r + 1 <= h)
    )
    return corners[mask]


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(points, axis=0)
    centered = points - mean
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))
    if mean_dist <= 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2) / mean_dist
    t = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]], dtype=float)
    points_h = np.column_stack((points, np.ones(len(points))))
    normalized = (t @ points_h.T).T
    return normalized[:, :2], t
