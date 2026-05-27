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
        description="Generate rule-based open circuit defect on PCB trace."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join("PCB_Dataset", "04.JPG"),
        help="Path to original PCB image."
    )

    parser.add_argument(
        "--topology_dir",
        type=str,
        default=os.path.join("outputs", "topology"),
        help="Directory containing trace_mask and attack_candidate_mask."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "open_circuit"),
        help="Directory to save generated open circuit results."
    )

    parser.add_argument(
        "--class_id",
        type=int,
        default=1,
        help="YOLO class id for open circuit. Adjust according to your class mapping."
    )

    parser.add_argument(
        "--min_gap_length",
        type=int,
        default=6,
        help="Minimum gap length along the trace direction, in pixels."
    )

    parser.add_argument(
        "--max_gap_length",
        type=int,
        default=14,
        help="Maximum gap length along the trace direction, in pixels."
    )

    parser.add_argument(
        "--cut_width_multiplier",
        type=float,
        default=2.6,
        help="Cut width multiplier based on local trace half-width. Larger values cut through thicker traces."
    )

    parser.add_argument(
        "--min_cut_width",
        type=int,
        default=10,
        help="Minimum cut width across the trace, in pixels."
    )

    parser.add_argument(
        "--max_cut_width",
        type=int,
        default=35,
        help="Maximum cut width across the trace, in pixels."
    )

    parser.add_argument(
        "--min_trace_half_width",
        type=float,
        default=2.0,
        help="Minimum distance-transform value for eligible trace center pixels."
    )

    parser.add_argument(
        "--max_trace_half_width",
        type=float,
        default=12.0,
        help="Maximum distance-transform value for eligible trace center pixels. Helps avoid pads/large copper blocks."
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5000,
        help="Maximum random attempts to find a valid open circuit location."
    )

    parser.add_argument(
        "--component_min_area",
        type=int,
        default=5000,
        help="Only sample from connected components larger than this area."
    )

    parser.add_argument(
        "--border_margin",
        type=int,
        default=40,
        help="Avoid sampling too close to image border."
    )

    parser.add_argument(
        "--pad_v_min",
        type=int,
        default=120,
        help="Minimum grayscale value for detecting bright solder pads / joints."
    )

    parser.add_argument(
        "--pad_s_max",
        type=int,
        default=90,
        help="Maximum HSV saturation for detecting silver/bright solder pads."
    )

    parser.add_argument(
        "--pad_dilate",
        type=int,
        default=28,
        help="Dilation radius for pad avoidance mask, in pixels."
    )

    parser.add_argument(
        "--pca_radius",
        type=int,
        default=24,
        help="Local window radius for PCA-based straight trace validation."
    )

    parser.add_argument(
        "--min_elongation",
        type=float,
        default=4.0,
        help="Minimum PCA elongation ratio for accepting a straight trace segment."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of open circuit samples to generate from the same image."
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="Defect-centered crop size. Use 0 to disable crop output."
    )

    parser.add_argument(
        "--min_defect_area",
        type=int,
        default=35,
        help="Minimum open circuit defect area in pixels."
    )

    parser.add_argument(
        "--min_bbox_width",
        type=int,
        default=4,
        help="Minimum defect bbox width in pixels."
    )

    parser.add_argument(
        "--min_bbox_height",
        type=int,
        default=4,
        help="Minimum defect bbox height in pixels."
    )

    parser.add_argument(
        "--local_check_padding",
        type=int,
        default=24,
        help="Padding around defect for local connectivity validation."
    )

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
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )
    return num_labels - 1


def keep_large_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Keep only large connected components.

    This helps avoid sampling from isolated text such as P0 / P1 / P2 / P3.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

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


def create_pad_avoid_mask(
    image_bgr: np.ndarray,
    v_min: int = 120,
    s_max: int = 90,
    dilate_radius: int = 28
) -> np.ndarray:
    """
    Detect bright low-saturation solder pads / joints and dilate them.

    Open-circuit defects should be generated on normal trace segments,
    not in pad-near or solder-joint regions.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    saturation = hsv[:, :, 1]

    pad_mask = np.zeros(gray.shape, dtype=np.uint8)
    pad_mask[(gray >= v_min) & (saturation <= s_max)] = 255

    # Clean tiny bright noise.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_OPEN, kernel_small)

    if dilate_radius > 0:
        k = int(dilate_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        pad_mask = cv2.dilate(pad_mask, kernel, iterations=1)

    return pad_mask


# -----------------------------------------------------------------------------
# Open-circuit geometry generation
# -----------------------------------------------------------------------------

def estimate_local_trace_direction(
    trace_mask: np.ndarray,
    x: int,
    y: int,
    window_radius: int = 24
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Estimate local trace tangent direction using PCA on local trace pixels.

    Returns:
        ((tx, ty), elongation_ratio), or None if direction is unreliable.

    The elongation ratio is used to reject corners, junctions, pads, and
    other non-straight local structures. Larger means more line-like.
    """
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


def make_rotated_rect_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    tangent: Tuple[float, float],
    gap_length: int,
    cut_width: int
) -> np.ndarray:
    """
    Create a rotated rectangular cut mask.

    gap_length is along the trace tangent direction.
    cut_width is across the trace normal direction.
    """
    tx, ty = tangent
    nx, ny = -ty, tx
    cx, cy = center

    half_l = gap_length / 2.0
    half_w = cut_width / 2.0

    corners = []
    for sign_l, sign_w in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        px = cx + sign_l * half_l * tx + sign_w * half_w * nx
        py = cy + sign_l * half_l * ty + sign_w * half_w * ny
        corners.append([px, py])

    box = np.array(corners, dtype=np.int32)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    return mask


def local_connectivity_increased(
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    padding: int = 24
) -> bool:
    """
    Check whether removing defect_mask increases local connected components.

    For open circuit, this should usually be True.
    """
    ys, xs = np.where(defect_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return False

    h, w = trace_mask.shape[:2]

    x1 = max(0, int(xs.min()) - padding)
    x2 = min(w, int(xs.max()) + padding + 1)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(h, int(ys.max()) + padding + 1)

    local_before = trace_mask[y1:y2, x1:x2].copy()
    local_defect = defect_mask[y1:y2, x1:x2].copy()

    local_after = local_before.copy()
    local_after[local_defect > 0] = 0

    before_cc = count_connected_components(local_before)
    after_cc = count_connected_components(local_after)

    return after_cc > before_cc


def is_valid_open_circuit(
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    min_defect_area: int = 35,
    min_bbox_width: int = 4,
    min_bbox_height: int = 4,
    local_check_padding: int = 24
) -> bool:
    """
    Validate open circuit geometry.

    Requirements:
    1. defect area and bbox are visible enough;
    2. the defect should locally break a trace;
    3. it should not remove an extremely large region.
    """
    defect_area = int(np.count_nonzero(defect_mask))

    if defect_area < min_defect_area:
        return False

    ys, xs = np.where(defect_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return False

    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)

    if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
        return False

    # Avoid huge cuts that no longer look like a local open circuit.
    if max(bbox_width, bbox_height) > 45:
        return False

    if not local_connectivity_increased(
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        padding=local_check_padding
    ):
        return False

    return True


def sample_valid_open_circuit(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    min_gap_length: int,
    max_gap_length: int,
    cut_width_multiplier: float,
    min_cut_width: int,
    max_cut_width: int,
    min_trace_half_width: float,
    max_trace_half_width: float,
    max_attempts: int,
    component_min_area: int,
    border_margin: int,
    min_defect_area: int,
    min_bbox_width: int,
    min_bbox_height: int,
    local_check_padding: int,
    pad_v_min: int,
    pad_s_max: int,
    pad_dilate: int,
    pca_radius: int,
    min_elongation: float
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    """
    Randomly sample a valid open circuit defect.

    Returns:
        defect_mask, center, geometry_info
    """
    eligible = keep_large_components(
        attack_candidate_mask,
        min_area=component_min_area
    )
    eligible = remove_border_area(eligible, margin=border_margin)

    # Avoid pads / solder joints and their neighborhood.
    pad_avoid_mask = create_pad_avoid_mask(
        image_bgr=image_bgr,
        v_min=pad_v_min,
        s_max=pad_s_max,
        dilate_radius=pad_dilate
    )
    eligible[pad_avoid_mask > 0] = 0

    # Distance transform gives approximate local trace half-width.
    dist = cv2.distanceTransform(eligible, cv2.DIST_L2, 5)

    candidate_mask = np.zeros_like(eligible)
    candidate_mask[
        (eligible > 0) &
        (dist >= min_trace_half_width) &
        (dist <= max_trace_half_width)
    ] = 255

    ys, xs = np.where(candidate_mask > 0)

    if len(xs) == 0:
        raise RuntimeError(
            "No eligible trace center pixels found. Try reducing min_trace_half_width, "
            "increasing max_trace_half_width, or reducing component_min_area."
        )

    h, w = trace_mask.shape[:2]

    for attempt in range(max_attempts):
        idx = random.randint(0, len(xs) - 1)
        x = int(xs[idx])
        y = int(ys[idx])

        local_half_width = float(dist[y, x])
        gap_length = random.randint(min_gap_length, max_gap_length)
        cut_width = int(round(local_half_width * cut_width_multiplier))
        cut_width = max(min_cut_width, min(cut_width, max_cut_width))

        if x - cut_width < border_margin or x + cut_width >= w - border_margin:
            continue
        if y - cut_width < border_margin or y + cut_width >= h - border_margin:
            continue

        direction_info = estimate_local_trace_direction(
            trace_mask=trace_mask,
            x=x,
            y=y,
            window_radius=max(pca_radius, cut_width)
        )

        if direction_info is None:
            continue

        tangent, elongation = direction_info

        # Reject corners, junctions, pad-near structures, and non-straight traces.
        if elongation < min_elongation:
            continue

        cut_shape = make_rotated_rect_mask(
            shape=trace_mask.shape,
            center=(x, y),
            tangent=tangent,
            gap_length=gap_length,
            cut_width=cut_width
        )

        defect_mask = cv2.bitwise_and(cut_shape, trace_mask)

        if is_valid_open_circuit(
            trace_mask=trace_mask,
            defect_mask=defect_mask,
            min_defect_area=min_defect_area,
            min_bbox_width=min_bbox_width,
            min_bbox_height=min_bbox_height,
            local_check_padding=local_check_padding
        ):
            ys_def, xs_def = np.where(defect_mask > 0)
            defect_area = int(np.count_nonzero(defect_mask))
            bbox_w = int(xs_def.max() - xs_def.min() + 1)
            bbox_h = int(ys_def.max() - ys_def.min() + 1)

            geometry_info = {
                "gap_length": int(gap_length),
                "cut_width": int(cut_width),
                "local_trace_half_width": float(local_half_width),
                "tangent": [float(tangent[0]), float(tangent[1])],
                "pca_elongation": float(elongation),
                "pad_v_min": int(pad_v_min),
                "pad_s_max": int(pad_s_max),
                "pad_dilate": int(pad_dilate),
                "pca_radius": int(pca_radius),
                "min_elongation": float(min_elongation),
                "defect_area_px": int(defect_area),
                "defect_bbox_size_px": [int(bbox_w), int(bbox_h)],
                "attempts_used": int(attempt + 1)
            }

            print(f"[INFO] Valid open circuit found after {attempt + 1} attempts.")
            print(f"[INFO] Center: ({x}, {y})")
            print(f"[INFO] Gap length: {gap_length} px")
            print(f"[INFO] Cut width: {cut_width} px")
            print(f"[INFO] Local trace half-width: {local_half_width:.2f} px")
            print(f"[INFO] PCA elongation: {elongation:.2f}")
            print(f"[INFO] Defect area: {defect_area} px")
            print(f"[INFO] Defect bbox size: {bbox_w} x {bbox_h} px")

            return defect_mask, (x, y), geometry_info

    raise RuntimeError(
        "Failed to find valid open circuit location. Try increasing max_attempts, "
        "reducing min_defect_area, or adjusting gap/cut parameters."
    )


# -----------------------------------------------------------------------------
# Rendering and annotation utilities
# -----------------------------------------------------------------------------

def get_local_substrate_color(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    radius: int
) -> np.ndarray:
    """
    Estimate local substrate color around the defect.

    We sample from an annulus around the defect, excluding trace pixels.
    """
    radius = max(3, int(radius))

    kernel_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 4 + 1, radius * 4 + 1)
    )

    kernel_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1)
    )

    outer = cv2.dilate(defect_mask, kernel_outer, iterations=1)
    inner = cv2.dilate(defect_mask, kernel_inner, iterations=1)

    annulus = cv2.subtract(outer, inner)

    substrate_region = annulus.copy()
    substrate_region[trace_mask > 0] = 0

    ys, xs = np.where(substrate_region > 0)

    if len(xs) < 10:
        ys, xs = np.where(trace_mask == 0)

    pixels = image_bgr[ys, xs]
    median_color = np.median(pixels, axis=0).astype(np.uint8)

    return median_color


def render_open_circuit(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    blend_radius: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render open circuit by replacing removed trace pixels with local substrate color.

    Returns:
        synthetic image, substrate color used.
    """
    output = image_bgr.copy()

    substrate_color = get_local_substrate_color(
        image_bgr=image_bgr,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        radius=blend_radius
    )

    output[defect_mask > 0] = substrate_color

    # Smooth only the defect boundary to avoid a hard synthetic edge.
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary = cv2.dilate(defect_mask, boundary_kernel, iterations=1)
    boundary = cv2.subtract(boundary, cv2.erode(defect_mask, boundary_kernel, iterations=1))

    blurred = cv2.GaussianBlur(output, (5, 5), 0)
    output[boundary > 0] = blurred[boundary > 0]

    return output, substrate_color


def mask_to_yolo_bbox(
    defect_mask: np.ndarray,
    class_id: int
) -> Optional[str]:
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


def create_debug_overlay(
    image_bgr: np.ndarray,
    defect_mask: np.ndarray,
    bbox_line: Optional[str]
) -> np.ndarray:
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


def crop_around_mask(
    image_bgr: np.ndarray,
    defect_mask: np.ndarray,
    crop_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
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


def create_crop_debug_overlay(
    crop_bgr: np.ndarray,
    crop_defect_mask: np.ndarray,
    crop_yolo_line: Optional[str]
) -> np.ndarray:
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

def generate_one_sample(
    args,
    image: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    stem: str,
    sample_index: int
) -> Dict[str, Any]:
    defect_mask, center, geometry_info = sample_valid_open_circuit(
        image_bgr=image,
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        min_gap_length=args.min_gap_length,
        max_gap_length=args.max_gap_length,
        cut_width_multiplier=args.cut_width_multiplier,
        min_cut_width=args.min_cut_width,
        max_cut_width=args.max_cut_width,
        min_trace_half_width=args.min_trace_half_width,
        max_trace_half_width=args.max_trace_half_width,
        max_attempts=args.max_attempts,
        component_min_area=args.component_min_area,
        border_margin=args.border_margin,
        min_defect_area=args.min_defect_area,
        min_bbox_width=args.min_bbox_width,
        min_bbox_height=args.min_bbox_height,
        local_check_padding=args.local_check_padding,
        pad_v_min=args.pad_v_min,
        pad_s_max=args.pad_s_max,
        pad_dilate=args.pad_dilate,
        pca_radius=args.pca_radius,
        min_elongation=args.min_elongation
    )

    blend_radius = max(geometry_info["cut_width"], geometry_info["gap_length"])

    synthetic_image, substrate_color = render_open_circuit(
        image_bgr=image,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        blend_radius=blend_radius
    )

    yolo_line = mask_to_yolo_bbox(
        defect_mask=defect_mask,
        class_id=args.class_id
    )

    debug_overlay = create_debug_overlay(
        image_bgr=synthetic_image,
        defect_mask=defect_mask,
        bbox_line=yolo_line
    )

    if args.num_samples > 1:
        suffix = f"open_circuit_{sample_index:04d}"
    else:
        suffix = "open_circuit"

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
    crop_mask_path = None
    crop_debug_path = None
    crop_label_path = None
    crop_yolo_line = None

    if args.crop_size > 0:
        crop_image, crop_mask, crop_box = crop_around_mask(
            image_bgr=synthetic_image,
            defect_mask=defect_mask,
            crop_size=args.crop_size
        )

        crop_yolo_line = mask_to_yolo_bbox(
            defect_mask=crop_mask,
            class_id=args.class_id
        )

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
        "defect_type": "open_circuit",
        "class_id": int(args.class_id),
        "sample_index": int(sample_index),
        "center_xy": [int(center[0]), int(center[1])],
        "geometry": geometry_info,
        "substrate_color_bgr": substrate_color.astype(int).tolist(),
        "full_yolo_label": yolo_line,
        "full_bbox_px": get_bbox_pixels(defect_mask),
        "generation_parameters": {
            "min_gap_length": int(args.min_gap_length),
            "max_gap_length": int(args.max_gap_length),
            "cut_width_multiplier": float(args.cut_width_multiplier),
            "min_cut_width": int(args.min_cut_width),
            "max_cut_width": int(args.max_cut_width),
            "min_trace_half_width": float(args.min_trace_half_width),
            "max_trace_half_width": float(args.max_trace_half_width),
            "max_attempts": int(args.max_attempts),
            "component_min_area": int(args.component_min_area),
            "border_margin": int(args.border_margin),
            "min_defect_area": int(args.min_defect_area),
            "min_bbox_width": int(args.min_bbox_width),
            "min_bbox_height": int(args.min_bbox_height),
            "local_check_padding": int(args.local_check_padding),
            "pad_v_min": int(args.pad_v_min),
            "pad_s_max": int(args.pad_s_max),
            "pad_dilate": int(args.pad_dilate),
            "pca_radius": int(args.pca_radius),
            "min_elongation": float(args.min_elongation),
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

    print("[SUCCESS] Open circuit generation completed.")
    print(f"  Synthetic image: {synthetic_path}")
    print(f"  Defect mask:     {defect_mask_path}")
    print(f"  Debug overlay:   {debug_path}")
    print(f"  YOLO label:      {label_path}")
    if crop_path is not None:
        print(f"  Crop image:      {crop_path}")
        print(f"  Crop debug:      {crop_debug_path}")
        print(f"  Crop label:      {crop_label_path}")
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

    trace_mask_path = os.path.join(
        args.topology_dir,
        f"{stem}_trace_mask.png"
    )

    attack_candidate_path = os.path.join(
        args.topology_dir,
        f"{stem}_attack_candidate_mask.png"
    )

    image = load_bgr(args.image)
    trace_mask = load_gray(trace_mask_path)
    attack_candidate_mask = load_gray(attack_candidate_path)

    print(f"[INFO] Loaded image: {args.image}")
    print(f"[INFO] Loaded trace mask: {trace_mask_path}")
    print(f"[INFO] Loaded attack candidate mask: {attack_candidate_path}")
    print(f"[INFO] Number of samples: {args.num_samples}")

    all_metadata = []

    for i in range(1, args.num_samples + 1):
        print(f"\n[INFO] Generating open circuit sample {i}/{args.num_samples}...")
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
        summary_path = os.path.join(args.output_dir, f"{stem}_open_circuit_summary.json")
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
