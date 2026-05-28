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
        description="Generate rule-based short circuit defect on PCB trace."
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
        default=os.path.join("outputs", "short_circuit"),
        help="Directory to save generated short circuit results."
    )

    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="YOLO class id for short circuit. Adjust according to your class mapping."
    )

    parser.add_argument(
        "--min_gap_distance",
        type=int,
        default=4,
        help="Minimum center-to-center distance between adjacent traces to bridge, in pixels."
    )

    parser.add_argument(
        "--max_gap_distance",
        type=int,
        default=34,
        help="Maximum center-to-center distance between adjacent traces to bridge, in pixels."
    )

    parser.add_argument(
        "--bridge_width_multiplier",
        type=float,
        default=0.9,
        help="Bridge width multiplier based on the local trace width."
    )

    parser.add_argument(
        "--min_bridge_width",
        type=int,
        default=4,
        help="Minimum bridge width in pixels."
    )

    parser.add_argument(
        "--max_bridge_width",
        type=int,
        default=18,
        help="Maximum bridge width in pixels."
    )

    parser.add_argument(
        "--min_trace_half_width",
        type=float,
        default=1.4,
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
        default=20000,
        help="Maximum random attempts to find a valid short circuit location."
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
        default=24,
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
        default=3.0,
        help="Minimum PCA elongation ratio for accepting a straight trace segment."
    )

    parser.add_argument(
        "--min_parallel_cos",
        type=float,
        default=0.85,
        help="Minimum absolute cosine similarity between two local trace directions."
    )

    parser.add_argument(
        "--corridor_clearance_ratio",
        type=float,
        default=0.55,
        help="Maximum allowed overlap ratio between the bridge corridor and existing traces."
    )

    parser.add_argument(
        "--require_component_decrease",
        action="store_true",
        help="If set, require connected components to decrease after adding the bridge. Disabled by default because PCB traces can be globally connected elsewhere."
    )

    parser.add_argument(
        "--local_check_padding",
        type=int,
        default=28,
        help="Padding around defect for local connectivity validation."
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
        help="Number of short circuit samples to generate from the same image."
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
        default=12,
        help="Minimum added short-circuit area in pixels."
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
    dilate_radius: int = 24
) -> np.ndarray:
    """
    Detect bright low-saturation solder pads / joints and dilate them.

    Short-circuit defects should be generated on normal trace segments,
    not in pad-near or solder-joint regions.
    """
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


# -----------------------------------------------------------------------------
# Short-circuit geometry generation
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


def make_bridge_mask(
    shape: Tuple[int, int],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    bridge_width: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, p1, p2, 255, thickness=int(bridge_width), lineType=cv2.LINE_AA)

    # Convert anti-aliased line to clean binary mask and slightly round the corners.
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def get_inner_edge_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Extract inner edge pixels of trace regions.

    Short circuit bridges should start from trace boundaries rather than trace
    center pixels. This makes it easier to find the adjacent trace across a
    substrate gap.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    edge = cv2.subtract(binary_mask, eroded)
    return edge


def local_connectivity_decreased(
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    padding: int = 28
) -> bool:
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
    local_after[local_defect > 0] = 255

    before_cc = count_connected_components(local_before)
    after_cc = count_connected_components(local_after)

    return after_cc < before_cc


def is_valid_short_circuit(
    trace_mask: np.ndarray,
    bridge_mask: np.ndarray,
    min_defect_area: int = 12,
    min_bbox_width: int = 4,
    min_bbox_height: int = 4,
    local_check_padding: int = 28,
    corridor_clearance_ratio: float = 0.55,
    require_component_decrease: bool = False
) -> bool:
    """
    Validate whether the proposed bridge creates a plausible short circuit.

    For PCB images, two adjacent traces may already belong to the same global
    connected component because they are connected elsewhere through pads or
    board-level routing. Therefore, component decrease is useful but should not
    be mandatory by default.

    bridge_mask contains the full bridge geometry.
    defect_mask is the newly added copper region, i.e. bridge minus original trace.
    """
    defect_mask = bridge_mask.copy()
    defect_mask[trace_mask > 0] = 0

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

    overlap_area = int(np.count_nonzero((bridge_mask > 0) & (trace_mask > 0)))
    bridge_area = int(np.count_nonzero(bridge_mask))
    overlap_ratio = overlap_area / max(bridge_area, 1)
    if overlap_ratio > corridor_clearance_ratio:
        return False

    if require_component_decrease:
        before_cc = count_connected_components(trace_mask)
        after_mask = trace_mask.copy()
        after_mask[bridge_mask > 0] = 255
        after_cc = count_connected_components(after_mask)

        if after_cc >= before_cc:
            return False

        if not local_connectivity_decreased(
            trace_mask=trace_mask,
            defect_mask=bridge_mask,
            padding=local_check_padding
        ):
            return False

    return True


def sample_second_trace_point(
    trace_mask: np.ndarray,
    eligible_trace_mask: np.ndarray,
    labels: np.ndarray,
    pad_avoid_mask: np.ndarray,
    distance_map: np.ndarray,
    x1: int,
    y1: int,
    label1: int,
    tangent1: Tuple[float, float],
    min_gap_distance: int,
    max_gap_distance: int,
    pca_radius: int,
    min_elongation: float,
    min_parallel_cos: float,
    min_trace_half_width: float,
    max_trace_half_width: float
) -> Optional[Dict[str, Any]]:
    """
    Search from an edge point across the local normal direction to find a nearby,
    roughly parallel trace boundary.
    """
    tx1, ty1 = tangent1
    normal_candidates = [(-ty1, tx1), (ty1, -tx1)]
    random.shuffle(normal_candidates)

    h, w = trace_mask.shape[:2]

    for nx, ny in normal_candidates:
        seen_background = False
        background_run = 0.0

        search_limit = int(max_gap_distance + 2 * max_trace_half_width + 8)

        for d in np.arange(1.0, float(search_limit) + 0.51, 1.0):
            x = int(round(x1 + nx * d))
            y = int(round(y1 + ny * d))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if pad_avoid_mask[y, x] > 0:
                break

            is_trace = eligible_trace_mask[y, x] > 0

            if is_trace and not seen_background:
                continue

            if not is_trace:
                seen_background = True
                background_run += 1.0
                if background_run > max_gap_distance:
                    break
                continue

            if is_trace and seen_background:
                if background_run < float(min_gap_distance):
                    break

                trace_half_width2 = float(distance_map[y, x])
                if trace_half_width2 < min_trace_half_width or trace_half_width2 > max_trace_half_width:
                    break

                dir_info2 = estimate_local_trace_direction(
                    trace_mask=trace_mask,
                    x=x,
                    y=y,
                    window_radius=pca_radius
                )
                if dir_info2 is None:
                    break

                tangent2, elongation2 = dir_info2
                if elongation2 < min_elongation:
                    break

                parallel_cos = abs(float(tangent1[0] * tangent2[0] + tangent1[1] * tangent2[1]))
                if parallel_cos < min_parallel_cos:
                    break

                return {
                    "point2": (x, y),
                    "label2": int(labels[y, x]),
                    "tangent2": tangent2,
                    "elongation2": float(elongation2),
                    "distance": float(background_run),
                    "trace_half_width2": trace_half_width2,
                    "parallel_cos": parallel_cos,
                    "normal": (float(nx), float(ny))
                }

    return None


def sample_valid_short_circuit(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    min_gap_distance: int = 4,
    max_gap_distance: int = 34,
    bridge_width_multiplier: float = 0.9,
    min_bridge_width: int = 4,
    max_bridge_width: int = 18,
    min_trace_half_width: float = 1.4,
    max_trace_half_width: float = 12.0,
    max_attempts: int = 20000,
    component_min_area: int = 5000,
    border_margin: int = 40,
    min_defect_area: int = 12,
    min_bbox_width: int = 4,
    min_bbox_height: int = 4,
    pad_v_min: int = 120,
    pad_s_max: int = 90,
    pad_dilate: int = 24,
    pca_radius: int = 24,
    min_elongation: float = 3.0,
    min_parallel_cos: float = 0.85,
    corridor_clearance_ratio: float = 0.55,
    local_check_padding: int = 28,
    require_component_decrease: bool = False
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    """
    Sample a plausible short-circuit bridge between two nearby, approximately
    parallel trace boundaries.

    This version samples from trace edge pixels rather than trace center pixels,
    which is more suitable for bridge-like short circuit defects.
    """
    filtered_candidates = keep_large_components(
        attack_candidate_mask,
        min_area=component_min_area
    )
    filtered_candidates = remove_border_area(filtered_candidates, margin=border_margin)

    pad_avoid_mask = create_pad_avoid_mask(
        image_bgr=image_bgr,
        v_min=pad_v_min,
        s_max=pad_s_max,
        dilate_radius=pad_dilate
    )

    distance_map = cv2.distanceTransform(trace_mask, cv2.DIST_L2, 5)

    eligible = np.zeros_like(trace_mask)
    eligible[(filtered_candidates > 0)] = 255
    eligible[(distance_map < min_trace_half_width) | (distance_map > max_trace_half_width)] = 0
    eligible[pad_avoid_mask > 0] = 0

    edge_mask = get_inner_edge_mask(eligible, kernel_size=3)
    edge_mask = remove_border_area(edge_mask, margin=border_margin)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(trace_mask, connectivity=8)

    ys, xs = np.where(edge_mask > 0)
    if len(xs) == 0:
        raise RuntimeError(
            "No eligible trace-edge candidates found. "
            "Try relaxing pad_dilate, component_min_area, border_margin, or trace width thresholds."
        )

    candidate_points = list(zip(xs.tolist(), ys.tolist()))

    for attempt in range(1, max_attempts + 1):
        x1, y1 = random.choice(candidate_points)
        label1 = int(labels[y1, x1])
        if label1 <= 0:
            continue

        trace_half_width1 = float(distance_map[y1, x1])
        if trace_half_width1 < min_trace_half_width or trace_half_width1 > max_trace_half_width:
            continue

        dir_info1 = estimate_local_trace_direction(
            trace_mask=trace_mask,
            x=x1,
            y=y1,
            window_radius=pca_radius
        )
        if dir_info1 is None:
            continue

        tangent1, elongation1 = dir_info1
        if elongation1 < min_elongation:
            continue

        second = sample_second_trace_point(
            trace_mask=trace_mask,
            eligible_trace_mask=eligible,
            labels=labels,
            pad_avoid_mask=pad_avoid_mask,
            distance_map=distance_map,
            x1=x1,
            y1=y1,
            label1=label1,
            tangent1=tangent1,
            min_gap_distance=min_gap_distance,
            max_gap_distance=max_gap_distance,
            pca_radius=pca_radius,
            min_elongation=min_elongation,
            min_parallel_cos=min_parallel_cos,
            min_trace_half_width=min_trace_half_width,
            max_trace_half_width=max_trace_half_width
        )
        if second is None:
            continue

        x2, y2 = second["point2"]

        avg_trace_width = 2.0 * ((max(trace_half_width1, 1.0) + second["trace_half_width2"]) / 2.0)
        bridge_width = int(round(avg_trace_width * bridge_width_multiplier))
        bridge_width = int(np.clip(bridge_width, min_bridge_width, max_bridge_width))

        bridge_mask = make_bridge_mask(
            shape=trace_mask.shape,
            p1=(x1, y1),
            p2=(x2, y2),
            bridge_width=bridge_width
        )

        if not is_valid_short_circuit(
            trace_mask=trace_mask,
            bridge_mask=bridge_mask,
            min_defect_area=min_defect_area,
            min_bbox_width=min_bbox_width,
            min_bbox_height=min_bbox_height,
            local_check_padding=local_check_padding,
            corridor_clearance_ratio=corridor_clearance_ratio,
            require_component_decrease=require_component_decrease
        ):
            continue

        defect_mask = bridge_mask.copy()
        defect_mask[trace_mask > 0] = 0

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        bbox = get_bbox_pixels(defect_mask)

        geometry_info = {
            "point1_xy": [int(x1), int(y1)],
            "point2_xy": [int(x2), int(y2)],
            "label1": int(label1),
            "label2": int(second["label2"]),
            "bridge_width": int(bridge_width),
            "gap_distance": float(second["distance"]),
            "trace_half_width1": float(trace_half_width1),
            "trace_half_width2": float(second["trace_half_width2"]),
            "tangent1": [float(tangent1[0]), float(tangent1[1])],
            "tangent2": [float(second["tangent2"][0]), float(second["tangent2"][1])],
            "elongation1": float(elongation1),
            "elongation2": float(second["elongation2"]),
            "parallel_cos": float(second["parallel_cos"]),
            "normal_direction": [float(second["normal"][0]), float(second["normal"][1])],
            "defect_area_px": int(np.count_nonzero(defect_mask)),
            "defect_bbox_px": bbox,
            "attempts_used": int(attempt)
        }

        print(f"[INFO] Attempt {attempt}: valid short circuit found.")
        print(f"[INFO] Gap distance: {second['distance']:.2f} px")
        print(f"[INFO] Bridge width: {bridge_width} px")
        print(f"[INFO] PCA elongation: {elongation1:.2f}, {second['elongation2']:.2f}")
        print(f"[INFO] Parallel cosine: {second['parallel_cos']:.4f}")
        print(f"[INFO] Defect area: {np.count_nonzero(defect_mask)} px")
        if bbox is not None:
            print(f"[INFO] Defect bbox size: {bbox['width']} x {bbox['height']} px")

        return defect_mask, (cx, cy), geometry_info

    raise RuntimeError(
        "Failed to find valid short circuit location. "
        "Try increasing max_attempts or relaxing gap / parallelism / pad constraints."
    )



# -----------------------------------------------------------------------------
# Rendering and export helpers
# -----------------------------------------------------------------------------

def get_local_trace_color(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    radius: int = 16
) -> np.ndarray:
    """
    Estimate local copper/trace color from nearby existing trace pixels.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(defect_mask, kernel, iterations=1)

    ring = dilated.copy()
    ring[defect_mask > 0] = 0

    candidate = np.zeros_like(ring)
    candidate[(ring > 0) & (trace_mask > 0)] = 255

    ys, xs = np.where(candidate > 0)
    if len(xs) == 0:
        ys, xs = np.where(trace_mask > 0)
        if len(xs) == 0:
            return np.array([30, 80, 40], dtype=np.uint8)

    pixels = image_bgr[ys, xs]
    median_bgr = np.median(pixels, axis=0).astype(np.uint8)
    return median_bgr


def render_short_circuit(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    blend_radius: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render short circuit by adding copper-colored bridge pixels.
    """
    rendered = image_bgr.copy()
    trace_color = get_local_trace_color(
        image_bgr=image_bgr,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        radius=max(6, blend_radius)
    )

    ys, xs = np.where(defect_mask > 0)
    rendered[ys, xs] = trace_color

    return rendered, trace_color


def mask_to_yolo_bbox(defect_mask: np.ndarray, class_id: int) -> Optional[str]:
    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = defect_mask.shape[:2]
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    x_center = (x_min + x_max) / 2.0 / w
    y_center = (y_min + y_max) / 2.0 / h
    bbox_w = (x_max - x_min + 1) / w
    bbox_h = (y_max - y_min + 1) / h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"


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
        "height": int(ys.max() - ys.min() + 1)
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
    defect_mask, center, geometry_info = sample_valid_short_circuit(
        image_bgr=image,
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        min_gap_distance=args.min_gap_distance,
        max_gap_distance=args.max_gap_distance,
        bridge_width_multiplier=args.bridge_width_multiplier,
        min_bridge_width=args.min_bridge_width,
        max_bridge_width=args.max_bridge_width,
        min_trace_half_width=args.min_trace_half_width,
        max_trace_half_width=args.max_trace_half_width,
        max_attempts=args.max_attempts,
        component_min_area=args.component_min_area,
        border_margin=args.border_margin,
        min_defect_area=args.min_defect_area,
        min_bbox_width=args.min_bbox_width,
        min_bbox_height=args.min_bbox_height,
        pad_v_min=args.pad_v_min,
        pad_s_max=args.pad_s_max,
        pad_dilate=args.pad_dilate,
        pca_radius=args.pca_radius,
        min_elongation=args.min_elongation,
        min_parallel_cos=args.min_parallel_cos,
        corridor_clearance_ratio=args.corridor_clearance_ratio,
        local_check_padding=args.local_check_padding,
        require_component_decrease=args.require_component_decrease
    )

    blend_radius = max(geometry_info["bridge_width"], int(round(geometry_info["gap_distance"])))

    synthetic_image, trace_color = render_short_circuit(
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
        suffix = f"short_circuit_{sample_index:04d}"
    else:
        suffix = "short_circuit"

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
        "defect_type": "short_circuit",
        "class_id": int(args.class_id),
        "sample_index": int(sample_index),
        "center_xy": [int(center[0]), int(center[1])],
        "geometry": geometry_info,
        "trace_color_bgr": trace_color.astype(int).tolist(),
        "full_yolo_label": yolo_line,
        "full_bbox_px": get_bbox_pixels(defect_mask),
        "generation_parameters": {
            "min_gap_distance": int(args.min_gap_distance),
            "max_gap_distance": int(args.max_gap_distance),
            "bridge_width_multiplier": float(args.bridge_width_multiplier),
            "min_bridge_width": int(args.min_bridge_width),
            "max_bridge_width": int(args.max_bridge_width),
            "min_trace_half_width": float(args.min_trace_half_width),
            "max_trace_half_width": float(args.max_trace_half_width),
            "max_attempts": int(args.max_attempts),
            "component_min_area": int(args.component_min_area),
            "border_margin": int(args.border_margin),
            "pad_v_min": int(args.pad_v_min),
            "pad_s_max": int(args.pad_s_max),
            "pad_dilate": int(args.pad_dilate),
            "pca_radius": int(args.pca_radius),
            "min_elongation": float(args.min_elongation),
            "min_parallel_cos": float(args.min_parallel_cos),
            "corridor_clearance_ratio": float(args.corridor_clearance_ratio),
            "local_check_padding": int(args.local_check_padding),
            "require_component_decrease": bool(args.require_component_decrease),
            "min_defect_area": int(args.min_defect_area),
            "min_bbox_width": int(args.min_bbox_width),
            "min_bbox_height": int(args.min_bbox_height),
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

    print("[SUCCESS] Short circuit generation completed.")
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
        print(f"\n[INFO] Generating short circuit sample {i}/{args.num_samples}...")
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
        summary_path = os.path.join(args.output_dir, f"{stem}_short_circuit_summary.json")
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
