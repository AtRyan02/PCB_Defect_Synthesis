import cv2
import numpy as np
import os
import argparse
import random
import json
from datetime import datetime
from typing import Tuple, Optional, Dict, Any


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rule-based spur defect on PCB trace."
    )

    parser.add_argument("--image", type=str, default=os.path.join("PCB_Dataset", "04.JPG"),
                        help="Path to original PCB image.")
    parser.add_argument("--topology_dir", type=str, default=os.path.join("outputs", "topology"),
                        help="Directory containing trace_mask and attack_candidate_mask.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "spur"),
                        help="Directory to save generated spur results.")
    parser.add_argument("--class_id", type=int, default=2,
                        help="YOLO class id for spur. Adjust according to your class mapping.")

    parser.add_argument("--min_spur_length", type=int, default=14,
                        help="Minimum spur length in pixels. Larger default makes spur more distinguishable from small vias/noise.")
    parser.add_argument("--max_spur_length", type=int, default=32,
                        help="Maximum spur length in pixels.")
    parser.add_argument("--spur_width_multiplier", type=float, default=0.65,
                        help="Spur width multiplier based on local trace width.")
    parser.add_argument("--min_spur_width", type=int, default=4,
                        help="Minimum spur width in pixels.")
    parser.add_argument("--max_spur_width", type=int, default=10,
                        help="Maximum spur width in pixels.")
    parser.add_argument("--angle_jitter_deg", type=float, default=18.0,
                        help="Random angular jitter from outward normal, in degrees.")

    parser.add_argument("--max_trace_half_width", type=float, default=12.0,
                        help="Maximum distance-transform value for eligible trace pixels.")
    parser.add_argument("--max_attempts", type=int, default=15000,
                        help="Maximum random attempts to find a valid spur location.")
    parser.add_argument("--component_min_area", type=int, default=5000,
                        help="Only sample from connected components larger than this area.")
    parser.add_argument("--border_margin", type=int, default=40,
                        help="Avoid sampling too close to image border.")

    parser.add_argument("--pad_v_min", type=int, default=120,
                        help="Minimum grayscale value for detecting bright solder pads / joints.")
    parser.add_argument("--pad_s_max", type=int, default=90,
                        help="Maximum HSV saturation for detecting silver/bright solder pads.")
    parser.add_argument("--pad_dilate", type=int, default=30,
                        help="Dilation radius for pad avoidance mask, in pixels.")

    parser.add_argument("--pca_radius", type=int, default=22,
                        help="Local window radius for PCA-based trace validation.")
    parser.add_argument("--min_elongation", type=float, default=3.0,
                        help="Minimum PCA elongation ratio for accepting clean trace edges.")

    parser.add_argument("--min_defect_area", type=int, default=45,
                        help="Minimum added spur area in pixels.")
    parser.add_argument("--min_bbox_width", type=int, default=4,
                        help="Minimum defect bbox width in pixels.")
    parser.add_argument("--min_bbox_height", type=int, default=4,
                        help="Minimum defect bbox height in pixels.")
    parser.add_argument("--min_bbox_long_side", type=int, default=14,
                        help="Minimum long side of defect bbox in pixels. Rejects tiny spur-like noise.")
    parser.add_argument("--min_mask_elongation", type=float, default=2.0,
                        help="Minimum PCA elongation ratio of the final defect mask.")
    parser.add_argument("--dark_v_max", type=int, default=70,
                        help="Maximum grayscale value for detecting dark holes/vias to avoid.")
    parser.add_argument("--dark_dilate", type=int, default=10,
                        help="Dilation radius for avoiding dark holes/vias and via arrays.")
    parser.add_argument("--max_existing_trace_overlap_ratio", type=float, default=0.45,
                        help="Maximum allowed ratio of spur geometry overlapping existing trace.")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of spur samples to generate from the same image.")
    parser.add_argument("--crop_size", type=int, default=128,
                        help="Defect-centered crop size. Use 0 to disable crop output.")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_file_stem(path: str) -> str:
    filename = os.path.basename(path)
    stem, _ = os.path.splitext(filename)
    return stem


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read grayscale image: {path}")
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(path: str, image: np.ndarray):
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to save image: {path}")


def count_connected_components(binary_mask: np.ndarray) -> int:
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    return num_labels - 1


def keep_large_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    kept = np.zeros_like(binary_mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept[labels == label_id] = 255
    return kept


def remove_border_area(mask: np.ndarray, margin: int) -> np.ndarray:
    cleaned = mask.copy()
    h, w = cleaned.shape[:2]
    cleaned[:margin, :] = 0
    cleaned[h - margin:, :] = 0
    cleaned[:, :margin] = 0
    cleaned[:, w - margin:] = 0
    return cleaned


def create_pad_avoid_mask(image_bgr: np.ndarray, v_min: int = 120, s_max: int = 90,
                          dilate_radius: int = 22) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]

    pad_mask = np.zeros(gray.shape, dtype=np.uint8)
    pad_mask[(gray >= v_min) & (saturation <= s_max)] = 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_OPEN, kernel_small)

    if dilate_radius > 0:
        k = int(dilate_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        pad_mask = cv2.dilate(pad_mask, kernel, iterations=1)

    return pad_mask


def create_dark_feature_avoid_mask(image_bgr: np.ndarray, trace_mask: np.ndarray,
                                   v_max: int = 70, dilate_radius: int = 10) -> np.ndarray:
    """
    Detect and avoid small dark holes / vias / via arrays.

    Spur false positives were mainly triggered by small dark dots and via-like
    structures. This mask is only used as an exclusion area for sampling spur
    roots and generated spur bodies; it is not a label mask.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    dark = np.zeros_like(gray, dtype=np.uint8)
    dark[gray <= int(v_max)] = 255

    # Keep dark features near board background, but avoid classifying the trace
    # itself as a dark feature.
    trace_dilated = cv2.dilate(
        trace_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    dark[trace_dilated > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    filtered = np.zeros_like(dark)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # Keep small-to-medium dot/hole structures. Very large dark regions are
        # usually shadows or background and are less useful for exclusion.
        if 2 <= area <= 350 and bw <= 35 and bh <= 35:
            filtered[labels == label_id] = 255

    if dilate_radius > 0:
        k = int(dilate_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        filtered = cv2.dilate(filtered, kernel, iterations=1)

    return filtered


def compute_mask_shape_metrics(binary_mask: np.ndarray) -> Optional[Dict[str, float]]:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) < 3:
        return None

    bbox_width = float(xs.max() - xs.min() + 1)
    bbox_height = float(ys.max() - ys.min() + 1)
    bbox_long_side = max(bbox_width, bbox_height)
    bbox_short_side = max(1.0, min(bbox_width, bbox_height))

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts_centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(pts_centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        mask_elongation = 1.0
    else:
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        mask_elongation = float(eigenvalues[0] / max(eigenvalues[1], 1e-6))

    return {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_long_side": bbox_long_side,
        "bbox_short_side": bbox_short_side,
        "bbox_aspect": bbox_long_side / bbox_short_side,
        "mask_elongation": mask_elongation,
        "area": float(len(xs)),
    }


# -----------------------------------------------------------------------------
# Spur geometry generation
# -----------------------------------------------------------------------------

def get_inner_edge_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    return cv2.subtract(binary_mask, eroded)


def estimate_local_trace_direction(trace_mask: np.ndarray, x: int, y: int,
                                   window_radius: int = 22) -> Optional[Tuple[Tuple[float, float], float]]:
    h, w = trace_mask.shape[:2]
    x1 = max(0, x - window_radius)
    x2 = min(w, x + window_radius + 1)
    y1 = max(0, y - window_radius)
    y2 = min(h, y + window_radius + 1)

    patch = trace_mask[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) < 20:
        return None

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= np.mean(pts, axis=0, keepdims=True)

    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    lambda1 = float(eigenvalues[order[0]])
    lambda2 = float(eigenvalues[order[1]])
    if lambda1 <= 1e-6:
        return None

    elongation = lambda1 / max(lambda2, 1e-6)

    direction = eigenvectors[:, order[0]]
    tx = float(direction[0])
    ty = float(direction[1])
    norm = np.sqrt(tx * tx + ty * ty)
    if norm < 1e-6:
        return None

    return (tx / norm, ty / norm), elongation


def estimate_outward_normal(trace_mask: np.ndarray, x: int, y: int,
                            search_radius: int = 14) -> Optional[Tuple[float, float]]:
    h, w = trace_mask.shape[:2]
    x1 = max(0, x - search_radius)
    x2 = min(w, x + search_radius + 1)
    y1 = max(0, y - search_radius)
    y2 = min(h, y + search_radius + 1)

    patch = trace_mask[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) < 5:
        return None

    xs_global = xs + x1
    ys_global = ys + y1

    cx = float(np.mean(xs_global))
    cy = float(np.mean(ys_global))

    vx = float(x - cx)
    vy = float(y - cy)
    norm = np.sqrt(vx * vx + vy * vy)
    if norm < 1e-6:
        return None

    return vx / norm, vy / norm


def rotate_vector(vx: float, vy: float, degrees: float) -> Tuple[float, float]:
    theta = np.deg2rad(degrees)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return c * vx - s * vy, s * vx + c * vy


def make_spur_mask(shape: Tuple[int, int], start: Tuple[int, int],
                   direction: Tuple[float, float], length: int, width: int) -> np.ndarray:
    x0, y0 = start
    dx, dy = direction
    x1 = int(round(x0 + dx * length))
    y1 = int(round(y0 + dy * length))

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, (int(x0), int(y0)), (x1, y1), 255, thickness=int(width), lineType=cv2.LINE_AA)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def is_valid_spur(trace_mask: np.ndarray, spur_full_mask: np.ndarray, avoid_mask: np.ndarray,
                  start: Tuple[int, int], end: Tuple[int, int],
                  min_defect_area: int = 45,
                  min_bbox_width: int = 4, min_bbox_height: int = 4,
                  min_bbox_long_side: int = 14,
                  min_mask_elongation: float = 2.0,
                  max_existing_trace_overlap_ratio: float = 0.45) -> bool:
    # Reject if the proposed spur touches solder pads, vias, dark holes, or
    # their dilated neighborhoods.
    if np.any((spur_full_mask > 0) & (avoid_mask > 0)):
        return False

    defect_mask = spur_full_mask.copy()
    defect_mask[trace_mask > 0] = 0

    defect_area = int(np.count_nonzero(defect_mask))
    if defect_area < min_defect_area:
        return False

    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False

    metrics = compute_mask_shape_metrics(defect_mask)
    if metrics is None:
        return False

    bbox_width = int(metrics["bbox_width"])
    bbox_height = int(metrics["bbox_height"])

    if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
        return False
    if metrics["bbox_long_side"] < min_bbox_long_side:
        return False
    if metrics["mask_elongation"] < min_mask_elongation:
        return False

    full_area = int(np.count_nonzero(spur_full_mask))
    trace_overlap_area = int(np.count_nonzero((spur_full_mask > 0) & (trace_mask > 0)))
    overlap_ratio = trace_overlap_area / max(full_area, 1)
    if overlap_ratio > max_existing_trace_overlap_ratio:
        return False

    sx, sy = start
    ex, ey = end
    h, w = trace_mask.shape[:2]

    # Root must touch existing trace.
    local_radius = 3
    x1 = max(0, sx - local_radius)
    x2 = min(w, sx + local_radius + 1)
    y1 = max(0, sy - local_radius)
    y2 = min(h, sy + local_radius + 1)
    if np.count_nonzero(trace_mask[y1:y2, x1:x2]) == 0:
        return False

    # Tip should be outside existing trace. This prevents short-circuit-like
    # bridges and discourages drawing along the trace body.
    tip_radius = 3
    x1 = max(0, ex - tip_radius)
    x2 = min(w, ex + tip_radius + 1)
    y1 = max(0, ey - tip_radius)
    y2 = min(h, ey + tip_radius + 1)
    if np.count_nonzero(trace_mask[y1:y2, x1:x2]) > 0:
        return False

    # If adding the spur merges components, it is closer to short circuit.
    before_cc = count_connected_components(trace_mask)
    after_mask = trace_mask.copy()
    after_mask[spur_full_mask > 0] = 255
    after_cc = count_connected_components(after_mask)
    if after_cc < before_cc:
        return False

    return True

def sample_valid_spur(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    min_spur_length: int,
    max_spur_length: int,
    spur_width_multiplier: float,
    min_spur_width: int,
    max_spur_width: int,
    angle_jitter_deg: float,
    max_trace_half_width: float,
    max_attempts: int,
    component_min_area: int,
    border_margin: int,
    pad_v_min: int,
    pad_s_max: int,
    pad_dilate: int,
    pca_radius: int,
    min_elongation: float,
    min_defect_area: int,
    min_bbox_width: int,
    min_bbox_height: int,
    min_bbox_long_side: int,
    min_mask_elongation: float,
    max_existing_trace_overlap_ratio: float,
    dark_v_max: int,
    dark_dilate: int
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    filtered_candidates = keep_large_components(attack_candidate_mask, min_area=component_min_area)
    filtered_candidates = remove_border_area(filtered_candidates, margin=border_margin)

    pad_avoid_mask = create_pad_avoid_mask(
        image_bgr=image_bgr,
        v_min=pad_v_min,
        s_max=pad_s_max,
        dilate_radius=pad_dilate
    )
    dark_avoid_mask = create_dark_feature_avoid_mask(
        image_bgr=image_bgr,
        trace_mask=trace_mask,
        v_max=dark_v_max,
        dilate_radius=dark_dilate
    )
    avoid_mask = np.zeros_like(trace_mask)
    avoid_mask[(pad_avoid_mask > 0) | (dark_avoid_mask > 0)] = 255

    distance_map = cv2.distanceTransform(trace_mask, cv2.DIST_L2, 5)

    eligible = np.zeros_like(trace_mask)
    eligible[(filtered_candidates > 0) & (distance_map <= max_trace_half_width)] = 255
    eligible[avoid_mask > 0] = 0

    edge_mask = get_inner_edge_mask(eligible, kernel_size=3)
    edge_mask = remove_border_area(edge_mask, margin=border_margin)

    ys, xs = np.where(edge_mask > 0)
    if len(xs) == 0:
        raise RuntimeError(
            "No eligible trace-edge pixels found. Try relaxing pad_dilate, component_min_area, or border_margin."
        )

    h, w = trace_mask.shape[:2]

    for attempt in range(1, max_attempts + 1):
        idx = random.randint(0, len(xs) - 1)
        x = int(xs[idx])
        y = int(ys[idx])

        if x < border_margin or x >= w - border_margin:
            continue
        if y < border_margin or y >= h - border_margin:
            continue

        local_half_width = float(distance_map[y, x])
        if local_half_width > max_trace_half_width:
            continue

        dir_info = estimate_local_trace_direction(trace_mask, x=x, y=y, window_radius=pca_radius)
        if dir_info is None:
            continue

        tangent, elongation = dir_info
        if elongation < min_elongation:
            continue

        normal = estimate_outward_normal(
            trace_mask=trace_mask,
            x=x,
            y=y,
            search_radius=max(10, int(max_spur_width * 2))
        )
        if normal is None:
            continue

        nx, ny = normal
        jitter = random.uniform(-angle_jitter_deg, angle_jitter_deg)
        dx, dy = rotate_vector(nx, ny, jitter)

        norm = np.sqrt(dx * dx + dy * dy)
        if norm < 1e-6:
            continue
        dx /= norm
        dy /= norm

        length = random.randint(min_spur_length, max_spur_length)
        trace_width = max(2.0 * local_half_width, 3.0)
        spur_width = int(round(trace_width * spur_width_multiplier))
        spur_width = int(np.clip(spur_width, min_spur_width, max_spur_width))

        end_x = int(round(x + dx * length))
        end_y = int(round(y + dy * length))
        if end_x < border_margin or end_x >= w - border_margin:
            continue
        if end_y < border_margin or end_y >= h - border_margin:
            continue

        spur_full_mask = make_spur_mask(
            shape=trace_mask.shape,
            start=(x, y),
            direction=(dx, dy),
            length=length,
            width=spur_width
        )

        if not is_valid_spur(
            trace_mask=trace_mask,
            spur_full_mask=spur_full_mask,
            avoid_mask=avoid_mask,
            start=(x, y),
            end=(end_x, end_y),
            min_defect_area=min_defect_area,
            min_bbox_width=min_bbox_width,
            min_bbox_height=min_bbox_height,
            min_bbox_long_side=min_bbox_long_side,
            min_mask_elongation=min_mask_elongation,
            max_existing_trace_overlap_ratio=max_existing_trace_overlap_ratio
        ):
            continue

        defect_mask = spur_full_mask.copy()
        defect_mask[trace_mask > 0] = 0

        ys_def, xs_def = np.where(defect_mask > 0)
        cx = int(round((xs_def.min() + xs_def.max()) / 2.0))
        cy = int(round((ys_def.min() + ys_def.max()) / 2.0))

        bbox_w = int(xs_def.max() - xs_def.min() + 1)
        bbox_h = int(ys_def.max() - ys_def.min() + 1)
        defect_area = int(np.count_nonzero(defect_mask))
        shape_metrics = compute_mask_shape_metrics(defect_mask) or {}

        geometry_info = {
            "start_xy": [int(x), int(y)],
            "end_xy": [int(end_x), int(end_y)],
            "length": int(length),
            "spur_width": int(spur_width),
            "direction": [float(dx), float(dy)],
            "angle_jitter_deg": float(jitter),
            "local_trace_half_width": float(local_half_width),
            "tangent": [float(tangent[0]), float(tangent[1])],
            "elongation": float(elongation),
            "defect_area_px": int(defect_area),
            "defect_bbox_size_px": [int(bbox_w), int(bbox_h)],
            "defect_bbox_long_side_px": int(max(bbox_w, bbox_h)),
            "defect_mask_elongation": float(shape_metrics.get("mask_elongation", 0.0)),
            "avoid_dark_features": True,
            "attempts_used": int(attempt)
        }

        print(f"[INFO] Valid spur found after {attempt} attempts.")
        print(f"[INFO] Start: ({x}, {y})")
        print(f"[INFO] End: ({end_x}, {end_y})")
        print(f"[INFO] Length: {length} px")
        print(f"[INFO] Spur width: {spur_width} px")
        print(f"[INFO] PCA elongation: {elongation:.2f}")
        print(f"[INFO] Defect area: {defect_area} px")
        print(f"[INFO] Defect bbox size: {bbox_w} x {bbox_h} px")
        print(f"[INFO] Defect mask elongation: {shape_metrics.get('mask_elongation', 0.0):.2f}")

        return defect_mask, (cx, cy), geometry_info

    raise RuntimeError(
        "Failed to find valid spur location. Try increasing max_attempts or relaxing pad / elongation constraints."
    )


# -----------------------------------------------------------------------------
# Rendering and annotation utilities
# -----------------------------------------------------------------------------

def get_local_trace_color(image_bgr: np.ndarray, trace_mask: np.ndarray,
                          defect_mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(3, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

    local = cv2.dilate(defect_mask, kernel, iterations=1)
    local[defect_mask > 0] = 0

    trace_region = local.copy()
    trace_region[trace_mask == 0] = 0

    ys, xs = np.where(trace_region > 0)
    if len(xs) < 10:
        ys, xs = np.where(trace_mask > 0)

    pixels = image_bgr[ys, xs]
    median_color = np.median(pixels, axis=0).astype(np.uint8)
    return median_color


def render_spur(image_bgr: np.ndarray, trace_mask: np.ndarray,
                defect_mask: np.ndarray, blend_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    output = image_bgr.copy()

    trace_color = get_local_trace_color(
        image_bgr=image_bgr,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        radius=blend_radius
    )

    output[defect_mask > 0] = trace_color

    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.dilate(defect_mask, boundary_kernel, iterations=1)
    boundary = cv2.subtract(boundary, cv2.erode(defect_mask, boundary_kernel, iterations=1))

    blurred = cv2.GaussianBlur(output, (3, 3), 0)
    output[boundary > 0] = blurred[boundary > 0]

    return output, trace_color


def mask_to_yolo_bbox(defect_mask: np.ndarray, class_id: int) -> Optional[str]:
    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = defect_mask.shape[:2]

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    x_center = ((x_min + x_max) / 2.0) / w
    y_center = ((y_min + y_max) / 2.0) / h
    bbox_width = (x_max - x_min + 1) / w
    bbox_height = (y_max - y_min + 1) / h

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{bbox_width:.6f} "
        f"{bbox_height:.6f}"
    )


def get_bbox_pixels(defect_mask: np.ndarray) -> Optional[Dict[str, int]]:
    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    return {
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
        "area_px": int(np.count_nonzero(defect_mask))
    }


def create_debug_overlay(image_bgr: np.ndarray, defect_mask: np.ndarray,
                         bbox_line: Optional[str]) -> np.ndarray:
    debug = image_bgr.copy()

    color_layer = np.zeros_like(debug)
    color_layer[:, :, 2] = 255

    mask_bool = defect_mask > 0
    debug[mask_bool] = cv2.addWeighted(
        debug[mask_bool],
        0.4,
        color_layer[mask_bool],
        0.6,
        0
    )

    if bbox_line is not None:
        parts = bbox_line.strip().split()
        _, xc, yc, bw, bh = parts

        h, w = defect_mask.shape[:2]

        xc = float(xc) * w
        yc = float(yc) * h
        bw = float(bw) * w
        bh = float(bh) * h

        x1 = int(round(xc - bw / 2))
        y1 = int(round(yc - bh / 2))
        x2 = int(round(xc + bw / 2))
        y2 = int(round(yc + bh / 2))

        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return debug


def crop_around_mask(image_bgr: np.ndarray, defect_mask: np.ndarray,
                     crop_size: int = 128) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot crop around empty defect mask.")

    h, w = defect_mask.shape[:2]
    cx = int(round((int(xs.min()) + int(xs.max())) / 2.0))
    cy = int(round((int(ys.min()) + int(ys.max())) / 2.0))

    half = crop_size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped_image = image_bgr[y1:y2, x1:x2]
    cropped_mask = defect_mask[y1:y2, x1:x2]
    return cropped_image, cropped_mask, (x1, y1, x2, y2)


def create_crop_debug_overlay(crop_bgr: np.ndarray, crop_defect_mask: np.ndarray,
                              crop_yolo_line: Optional[str]) -> np.ndarray:
    return create_debug_overlay(
        image_bgr=crop_bgr,
        defect_mask=crop_defect_mask,
        bbox_line=crop_yolo_line
    )


def write_label(path: str, yolo_line: Optional[str]):
    if yolo_line is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(yolo_line + "\n")


def write_metadata(path: str, metadata: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Main generation routine
# -----------------------------------------------------------------------------

def generate_one_sample(args, image: np.ndarray, trace_mask: np.ndarray,
                        attack_candidate_mask: np.ndarray, stem: str,
                        sample_index: int) -> Dict[str, Any]:
    defect_mask, center, geometry_info = sample_valid_spur(
        image_bgr=image,
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        min_spur_length=args.min_spur_length,
        max_spur_length=args.max_spur_length,
        spur_width_multiplier=args.spur_width_multiplier,
        min_spur_width=args.min_spur_width,
        max_spur_width=args.max_spur_width,
        angle_jitter_deg=args.angle_jitter_deg,
        max_trace_half_width=args.max_trace_half_width,
        max_attempts=args.max_attempts,
        component_min_area=args.component_min_area,
        border_margin=args.border_margin,
        pad_v_min=args.pad_v_min,
        pad_s_max=args.pad_s_max,
        pad_dilate=args.pad_dilate,
        pca_radius=args.pca_radius,
        min_elongation=args.min_elongation,
        min_defect_area=args.min_defect_area,
        min_bbox_width=args.min_bbox_width,
        min_bbox_height=args.min_bbox_height,
        min_bbox_long_side=args.min_bbox_long_side,
        min_mask_elongation=args.min_mask_elongation,
        max_existing_trace_overlap_ratio=args.max_existing_trace_overlap_ratio,
        dark_v_max=args.dark_v_max,
        dark_dilate=args.dark_dilate
    )

    blend_radius = max(geometry_info["spur_width"], 5)

    synthetic_image, trace_color = render_spur(
        image_bgr=image,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        blend_radius=blend_radius
    )

    yolo_line = mask_to_yolo_bbox(defect_mask=defect_mask, class_id=args.class_id)
    debug_overlay = create_debug_overlay(
        image_bgr=synthetic_image,
        defect_mask=defect_mask,
        bbox_line=yolo_line
    )

    if args.num_samples > 1:
        suffix = f"spur_{sample_index:04d}"
    else:
        suffix = "spur"

    base_name = f"{stem}_{suffix}"

    synthetic_path = os.path.join(args.output_dir, f"{base_name}.png")
    defect_mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
    debug_path = os.path.join(args.output_dir, f"{base_name}_debug.png")
    label_path = os.path.join(args.output_dir, f"{base_name}.txt")
    metadata_path = os.path.join(args.output_dir, f"{base_name}_metadata.json")

    save_image(synthetic_path, synthetic_image)
    save_image(defect_mask_path, defect_mask)
    save_image(debug_path, debug_overlay)
    write_label(label_path, yolo_line)

    crop_info = None
    crop_path = None
    crop_debug_path = None

    if args.crop_size > 0:
        crop_image, crop_mask, crop_box = crop_around_mask(
            image_bgr=synthetic_image,
            defect_mask=defect_mask,
            crop_size=args.crop_size
        )

        crop_yolo_line = mask_to_yolo_bbox(defect_mask=crop_mask, class_id=args.class_id)
        crop_debug = create_crop_debug_overlay(
            crop_bgr=crop_image,
            crop_defect_mask=crop_mask,
            crop_yolo_line=crop_yolo_line
        )

        crop_path = os.path.join(args.output_dir, f"{base_name}_crop.png")
        crop_mask_path = os.path.join(args.output_dir, f"{base_name}_crop_mask.png")
        crop_debug_path = os.path.join(args.output_dir, f"{base_name}_crop_debug.png")
        crop_label_path = os.path.join(args.output_dir, f"{base_name}_crop.txt")

        save_image(crop_path, crop_image)
        save_image(crop_mask_path, crop_mask)
        save_image(crop_debug_path, crop_debug)
        write_label(crop_label_path, crop_yolo_line)

        crop_info = {
            "crop_box_xyxy": [int(v) for v in crop_box],
            "crop_size": int(args.crop_size),
            "crop_yolo_label": crop_yolo_line,
            "crop_bbox_px": get_bbox_pixels(crop_mask),
            "crop_image_path": crop_path,
            "crop_mask_path": crop_mask_path,
            "crop_debug_path": crop_debug_path,
            "crop_label_path": crop_label_path
        }

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": args.image,
        "defect_type": "spur",
        "class_id": int(args.class_id),
        "sample_index": int(sample_index),
        "center_xy": [int(center[0]), int(center[1])],
        "geometry": geometry_info,
        "trace_color_bgr": trace_color.astype(int).tolist(),
        "full_yolo_label": yolo_line,
        "full_bbox_px": get_bbox_pixels(defect_mask),
        "generation_parameters": {
            "min_spur_length": int(args.min_spur_length),
            "max_spur_length": int(args.max_spur_length),
            "spur_width_multiplier": float(args.spur_width_multiplier),
            "min_spur_width": int(args.min_spur_width),
            "max_spur_width": int(args.max_spur_width),
            "angle_jitter_deg": float(args.angle_jitter_deg),
            "max_trace_half_width": float(args.max_trace_half_width),
            "max_attempts": int(args.max_attempts),
            "component_min_area": int(args.component_min_area),
            "border_margin": int(args.border_margin),
            "pad_v_min": int(args.pad_v_min),
            "pad_s_max": int(args.pad_s_max),
            "pad_dilate": int(args.pad_dilate),
            "pca_radius": int(args.pca_radius),
            "min_elongation": float(args.min_elongation),
            "min_defect_area": int(args.min_defect_area),
            "min_bbox_width": int(args.min_bbox_width),
            "min_bbox_height": int(args.min_bbox_height),
            "min_bbox_long_side": int(args.min_bbox_long_side),
            "min_mask_elongation": float(args.min_mask_elongation),
            "max_existing_trace_overlap_ratio": float(args.max_existing_trace_overlap_ratio),
            "dark_v_max": int(args.dark_v_max),
            "dark_dilate": int(args.dark_dilate),
            "seed": args.seed,
            "crop_size": int(args.crop_size)
        },
        "outputs": {
            "synthetic_image_path": synthetic_path,
            "defect_mask_path": defect_mask_path,
            "debug_path": debug_path,
            "label_path": label_path,
            "metadata_path": metadata_path,
            "crop": crop_info
        }
    }

    write_metadata(metadata_path, metadata)

    print("[SUCCESS] Spur generation completed.")
    print(f"  Synthetic image: {synthetic_path}")
    print(f"  Defect mask:     {defect_mask_path}")
    print(f"  Debug overlay:   {debug_path}")
    print(f"  YOLO label:      {label_path}")
    if crop_path is not None:
        print(f"  Crop image:      {crop_path}")
        print(f"  Crop debug:      {crop_debug_path}")
    print(f"  Metadata:        {metadata_path}")
    print(f"  YOLO line:       {yolo_line}")

    return metadata


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ensure_dir(args.output_dir)

    stem = get_file_stem(args.image)

    trace_mask_path = os.path.join(args.topology_dir, f"{stem}_trace_mask.png")
    attack_candidate_path = os.path.join(args.topology_dir, f"{stem}_attack_candidate_mask.png")

    image = load_bgr(args.image)
    trace_mask = load_gray(trace_mask_path)
    attack_candidate_mask = load_gray(attack_candidate_path)

    print(f"[INFO] Loaded image: {args.image}")
    print(f"[INFO] Loaded trace mask: {trace_mask_path}")
    print(f"[INFO] Loaded attack candidate mask: {attack_candidate_path}")
    print(f"[INFO] Number of samples: {args.num_samples}")

    all_metadata = []

    for i in range(1, args.num_samples + 1):
        print(f"\n[INFO] Generating spur sample {i}/{args.num_samples}...")
        try:
            metadata = generate_one_sample(
                args=args,
                image=image,
                trace_mask=trace_mask,
                attack_candidate_mask=attack_candidate_mask,
                stem=stem,
                sample_index=i
            )
            all_metadata.append(metadata)
        except RuntimeError as exc:
            print(f"[WARNING] Sample {i} skipped: {exc}")

    if args.num_samples > 1:
        summary_path = os.path.join(args.output_dir, f"{stem}_spur_summary.json")
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_image": args.image,
            "requested_samples": int(args.num_samples),
            "successful_samples": int(len(all_metadata)),
            "samples": all_metadata
        }
        write_metadata(summary_path, summary)
        print(f"\n[SUCCESS] Batch summary saved to: {summary_path}")
        print(f"[SUCCESS] Successful samples: {len(all_metadata)} / {args.num_samples}")


if __name__ == "__main__":
    main()
