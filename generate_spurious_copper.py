
import cv2
import numpy as np
import os
import argparse
import random
import json
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rule-based spurious copper defect on PCB trace."
    )

    parser.add_argument("--image", type=str, default=os.path.join("PCB_Dataset", "04.JPG"),
                        help="Path to original PCB image.")
    parser.add_argument("--topology_dir", type=str, default=os.path.join("outputs", "topology"),
                        help="Directory containing trace_mask and attack_candidate_mask.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "spurious_copper"),
                        help="Directory to save generated spurious copper results.")
    parser.add_argument("--class_id", type=int, default=5,
                        help="YOLO class id for spurious copper. Adjust according to your class mapping.")

    # Shape control
    parser.add_argument("--shape_mode", type=str, default="mixed",
                        choices=["mixed", "rectangle", "trapezoid", "ellipse"],
                        help="Spurious copper geometry mode.")
    parser.add_argument("--rectangle_prob", type=float, default=0.34,
                        help="Probability of choosing rectangle shape when shape_mode=mixed.")
    parser.add_argument("--trapezoid_prob", type=float, default=0.33,
                        help="Probability of choosing trapezoid shape when shape_mode=mixed.")
    parser.add_argument("--ellipse_prob", type=float, default=0.33,
                        help="Probability of choosing ellipse shape when shape_mode=mixed.")

    # Size control
    parser.add_argument("--min_major_length", type=int, default=10,
                        help="Minimum major-axis length (parallel to trace) in pixels.")
    parser.add_argument("--max_major_length", type=int, default=26,
                        help="Maximum major-axis length (parallel to trace) in pixels.")
    parser.add_argument("--min_minor_length", type=int, default=4,
                        help="Minimum minor-axis thickness in pixels.")
    parser.add_argument("--max_minor_length", type=int, default=12,
                        help="Maximum minor-axis thickness in pixels.")

    parser.add_argument("--major_length_multiplier", type=float, default=1.10,
                        help="Scale major length from local trace width.")
    parser.add_argument("--minor_length_multiplier", type=float, default=0.95,
                        help="Scale minor length from local trace width.")

    # Relative placement
    parser.add_argument("--min_detach_gap", type=int, default=3,
                        help="Minimum gap between spurious copper and main trace, in pixels.")
    parser.add_argument("--max_detach_gap", type=int, default=10,
                        help="Maximum gap between spurious copper and main trace, in pixels.")
    parser.add_argument("--center_offset_jitter", type=float, default=0.35,
                        help="Random center shift along tangent, relative to major length.")
    parser.add_argument("--angle_jitter_deg", type=float, default=12.0,
                        help="Random angular jitter around trace tangent, in degrees.")

    # Trapezoid / ellipse details
    parser.add_argument("--trap_far_ratio_min", type=float, default=0.55,
                        help="Minimum far-side half-width ratio for trapezoid, relative to near-side half-width.")
    parser.add_argument("--trap_far_ratio_max", type=float, default=1.25,
                        help="Maximum far-side half-width ratio for trapezoid, relative to near-side half-width.")
    parser.add_argument("--ellipse_axis_scale_min", type=float, default=0.90,
                        help="Minimum ellipse axis scale factor.")
    parser.add_argument("--ellipse_axis_scale_max", type=float, default=1.15,
                        help="Maximum ellipse axis scale factor.")

    # Candidate filtering
    parser.add_argument("--max_trace_half_width", type=float, default=12.0,
                        help="Maximum distance-transform value for eligible trace pixels.")
    parser.add_argument("--max_attempts", type=int, default=20000,
                        help="Maximum random attempts to find a valid location.")
    parser.add_argument("--component_min_area", type=int, default=5000,
                        help="Only sample from connected components larger than this area.")
    parser.add_argument("--border_margin", type=int, default=40,
                        help="Avoid sampling too close to image border.")

    # Pad avoidance
    parser.add_argument("--pad_v_min", type=int, default=120,
                        help="Minimum grayscale value for detecting bright solder pads / joints.")
    parser.add_argument("--pad_s_max", type=int, default=90,
                        help="Maximum HSV saturation for detecting silver/bright solder pads.")
    parser.add_argument("--pad_dilate", type=int, default=18,
                        help="Dilation radius for pad avoidance mask, in pixels.")

    # Straightness constraint
    parser.add_argument("--pca_radius", type=int, default=22,
                        help="Local window radius for PCA-based trace validation.")
    parser.add_argument("--min_elongation", type=float, default=2.0,
                        help="Minimum PCA elongation ratio for accepting clean trace neighborhoods.")

    # Defect validity
    parser.add_argument("--min_defect_area", type=int, default=18,
                        help="Minimum spurious copper area in pixels.")
    parser.add_argument("--min_bbox_width", type=int, default=5,
                        help="Minimum defect bbox width in pixels.")
    parser.add_argument("--min_bbox_height", type=int, default=5,
                        help="Minimum defect bbox height in pixels.")
    parser.add_argument("--min_component_gain", type=int, default=1,
                        help="Minimum increase in connected-component count after adding defect.")

    # Output
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate from the same image.")
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
                          dilate_radius: int = 18) -> np.ndarray:
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
# Geometry utilities
# -----------------------------------------------------------------------------

def normalize_vector(vx: float, vy: float) -> Optional[Tuple[float, float]]:
    norm = float(np.sqrt(vx * vx + vy * vy))
    if norm < 1e-6:
        return None
    return vx / norm, vy / norm


def rotate_vector(vx: float, vy: float, degrees: float) -> Tuple[float, float]:
    theta = np.deg2rad(degrees)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return c * vx - s * vy, s * vx + c * vy


def point_add(point: Tuple[float, float], vec: Tuple[float, float], scale: float) -> Tuple[float, float]:
    return point[0] + vec[0] * scale, point[1] + vec[1] * scale


def points_to_int32(points: List[Tuple[float, float]]) -> np.ndarray:
    return np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)


def draw_filled_polygon(shape: Tuple[int, int], points: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = points_to_int32(points)
    cv2.fillPoly(mask, [pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# -----------------------------------------------------------------------------
# Trace analysis
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

    normalized = normalize_vector(tx, ty)
    if normalized is None:
        return None

    return normalized, elongation


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
    return normalize_vector(vx, vy)


def choose_shape_mode(args) -> str:
    if args.shape_mode != "mixed":
        return args.shape_mode

    probs = np.array([
        max(0.0, float(args.rectangle_prob)),
        max(0.0, float(args.trapezoid_prob)),
        max(0.0, float(args.ellipse_prob))
    ], dtype=np.float64)

    if probs.sum() <= 1e-8:
        probs = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    probs /= probs.sum()
    return str(np.random.choice(["rectangle", "trapezoid", "ellipse"], p=probs))


# -----------------------------------------------------------------------------
# Spurious-copper shape builders
# -----------------------------------------------------------------------------

def build_oriented_rectangle(shape: Tuple[int, int], center: Tuple[float, float],
                             tangent: Tuple[float, float], normal: Tuple[float, float],
                             major_len: float, minor_len: float) -> np.ndarray:
    half_major = major_len / 2.0
    half_minor = minor_len / 2.0

    c = center
    p1 = point_add(point_add(c, tangent,  half_major), normal,  half_minor)
    p2 = point_add(point_add(c, tangent,  half_major), normal, -half_minor)
    p3 = point_add(point_add(c, tangent, -half_major), normal, -half_minor)
    p4 = point_add(point_add(c, tangent, -half_major), normal,  half_minor)

    return draw_filled_polygon(shape, [p1, p2, p3, p4])


def build_oriented_trapezoid(shape: Tuple[int, int], center: Tuple[float, float],
                             tangent: Tuple[float, float], normal: Tuple[float, float],
                             major_len: float, near_half_minor: float, far_half_minor: float) -> np.ndarray:
    half_major = major_len / 2.0
    c = center

    # left/right ends along tangent
    left_center = point_add(c, tangent, -half_major)
    right_center = point_add(c, tangent, half_major)

    p1 = point_add(left_center, normal, near_half_minor)
    p2 = point_add(right_center, normal, far_half_minor)
    p3 = point_add(right_center, normal, -far_half_minor)
    p4 = point_add(left_center, normal, -near_half_minor)

    return draw_filled_polygon(shape, [p1, p2, p3, p4])


def build_oriented_ellipse(shape: Tuple[int, int], center: Tuple[float, float],
                           angle_deg: float, major_len: float, minor_len: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    axes = (max(1, int(round(major_len / 2.0))), max(1, int(round(minor_len / 2.0))))
    cv2.ellipse(
        mask,
        center=(int(round(center[0])), int(round(center[1]))),
        axes=axes,
        angle=float(angle_deg),
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def angle_from_vector(v: Tuple[float, float]) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def build_spurious_copper_mask(shape: Tuple[int, int],
                               center: Tuple[float, float],
                               tangent: Tuple[float, float],
                               normal: Tuple[float, float],
                               major_len: int,
                               minor_len: int,
                               shape_type: str,
                               args) -> np.ndarray:
    if shape_type == "rectangle":
        return build_oriented_rectangle(
            shape=shape,
            center=center,
            tangent=tangent,
            normal=normal,
            major_len=major_len,
            minor_len=minor_len
        )

    if shape_type == "trapezoid":
        near_half = max(2.0, minor_len * 0.5)
        far_ratio = random.uniform(float(args.trap_far_ratio_min), float(args.trap_far_ratio_max))
        far_half = max(2.0, near_half * far_ratio)
        return build_oriented_trapezoid(
            shape=shape,
            center=center,
            tangent=tangent,
            normal=normal,
            major_len=major_len,
            near_half_minor=near_half,
            far_half_minor=far_half
        )

    if shape_type == "ellipse":
        scale_major = random.uniform(float(args.ellipse_axis_scale_min), float(args.ellipse_axis_scale_max))
        scale_minor = random.uniform(float(args.ellipse_axis_scale_min), float(args.ellipse_axis_scale_max))
        angle_deg = angle_from_vector(tangent)
        return build_oriented_ellipse(
            shape=shape,
            center=center,
            angle_deg=angle_deg,
            major_len=major_len * scale_major,
            minor_len=minor_len * scale_minor
        )

    raise ValueError(f"Unsupported shape_type: {shape_type}")


# -----------------------------------------------------------------------------
# Validity checks
# -----------------------------------------------------------------------------

def is_valid_spurious_copper(trace_mask: np.ndarray, defect_mask: np.ndarray, pad_avoid_mask: np.ndarray,
                             dist_to_trace: np.ndarray, min_gap: int, max_gap: int,
                             min_defect_area: int, min_bbox_width: int, min_bbox_height: int,
                             min_component_gain: int) -> bool:
    if np.any((defect_mask > 0) & (trace_mask > 0)):
        return False

    if np.any((defect_mask > 0) & (pad_avoid_mask > 0)):
        return False

    area = int(np.count_nonzero(defect_mask))
    if area < min_defect_area:
        return False

    ys, xs = np.where(defect_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False

    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
        return False

    # Detached but close to trace
    distances = dist_to_trace[defect_mask > 0]
    min_distance = float(np.min(distances))
    mean_distance = float(np.mean(distances))
    if min_distance < float(min_gap):
        return False
    if mean_distance > float(max_gap + max(min_bbox_width, min_bbox_height)):
        return False

    before_cc = count_connected_components(trace_mask)
    after_mask = trace_mask.copy()
    after_mask[defect_mask > 0] = 255
    after_cc = count_connected_components(after_mask)

    if after_cc < before_cc + int(min_component_gain):
        return False

    return True


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def sample_valid_spurious_copper(image_bgr: np.ndarray,
                                 trace_mask: np.ndarray,
                                 attack_candidate_mask: np.ndarray,
                                 args) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    filtered_candidates = keep_large_components(attack_candidate_mask, min_area=args.component_min_area)
    filtered_candidates = remove_border_area(filtered_candidates, margin=args.border_margin)

    pad_avoid_mask = create_pad_avoid_mask(
        image_bgr=image_bgr,
        v_min=args.pad_v_min,
        s_max=args.pad_s_max,
        dilate_radius=args.pad_dilate
    )

    distance_map_inside_trace = cv2.distanceTransform(trace_mask, cv2.DIST_L2, 5)
    dist_to_trace = cv2.distanceTransform(255 - trace_mask, cv2.DIST_L2, 5)

    eligible = np.zeros_like(trace_mask)
    eligible[(filtered_candidates > 0) & (distance_map_inside_trace <= args.max_trace_half_width)] = 255
    eligible[pad_avoid_mask > 0] = 0

    edge_mask = get_inner_edge_mask(eligible, kernel_size=3)
    edge_mask = remove_border_area(edge_mask, margin=args.border_margin)

    ys, xs = np.where(edge_mask > 0)
    if len(xs) == 0:
        raise RuntimeError(
            "No eligible trace-edge pixels found. Try relaxing pad_dilate, component_min_area, or border_margin."
        )

    h, w = trace_mask.shape[:2]

    for attempt in range(1, args.max_attempts + 1):
        idx = random.randint(0, len(xs) - 1)
        x = int(xs[idx])
        y = int(ys[idx])

        if x < args.border_margin or x >= w - args.border_margin:
            continue
        if y < args.border_margin or y >= h - args.border_margin:
            continue

        local_half_width = float(distance_map_inside_trace[y, x])
        if local_half_width > args.max_trace_half_width:
            continue

        dir_info = estimate_local_trace_direction(trace_mask, x=x, y=y, window_radius=args.pca_radius)
        if dir_info is None:
            continue

        tangent, elongation = dir_info
        if elongation < args.min_elongation:
            continue

        outward_normal = estimate_outward_normal(
            trace_mask=trace_mask,
            x=x,
            y=y,
            search_radius=max(10, int(args.max_minor_length * 2))
        )
        if outward_normal is None:
            continue

        # Minor angular jitter around tangent orientation
        tangent_rot = rotate_vector(tangent[0], tangent[1], random.uniform(-args.angle_jitter_deg, args.angle_jitter_deg))
        tangent_rot = normalize_vector(tangent_rot[0], tangent_rot[1])
        if tangent_rot is None:
            continue

        # Keep normal approximately outward
        normal_rot = rotate_vector(outward_normal[0], outward_normal[1], random.uniform(-args.angle_jitter_deg, args.angle_jitter_deg))
        normal_rot = normalize_vector(normal_rot[0], normal_rot[1])
        if normal_rot is None:
            continue

        trace_width = max(2.0 * local_half_width, 3.0)

        major_len = int(round(trace_width * args.major_length_multiplier + random.randint(args.min_major_length, args.max_major_length) * 0.5))
        minor_len = int(round(trace_width * args.minor_length_multiplier))
        minor_len = int(np.clip(minor_len, args.min_minor_length, args.max_minor_length))
        major_len = int(np.clip(major_len, args.min_major_length, args.max_major_length))

        detach_gap = random.randint(args.min_detach_gap, args.max_detach_gap)
        outward_extent = minor_len / 2.0 + detach_gap

        tangent_shift = random.uniform(-args.center_offset_jitter, args.center_offset_jitter) * major_len
        center = point_add((float(x), float(y)), normal_rot, outward_extent)
        center = point_add(center, tangent_rot, tangent_shift)

        shape_type = choose_shape_mode(args)
        defect_mask = build_spurious_copper_mask(
            shape=trace_mask.shape,
            center=center,
            tangent=tangent_rot,
            normal=normal_rot,
            major_len=major_len,
            minor_len=minor_len,
            shape_type=shape_type,
            args=args
        )

        ys_def, xs_def = np.where(defect_mask > 0)
        if len(xs_def) == 0 or len(ys_def) == 0:
            continue

        if xs_def.min() < args.border_margin or ys_def.min() < args.border_margin:
            continue
        if xs_def.max() >= w - args.border_margin or ys_def.max() >= h - args.border_margin:
            continue

        if not is_valid_spurious_copper(
            trace_mask=trace_mask,
            defect_mask=defect_mask,
            pad_avoid_mask=pad_avoid_mask,
            dist_to_trace=dist_to_trace,
            min_gap=args.min_detach_gap,
            max_gap=args.max_detach_gap,
            min_defect_area=args.min_defect_area,
            min_bbox_width=args.min_bbox_width,
            min_bbox_height=args.min_bbox_height,
            min_component_gain=args.min_component_gain
        ):
            continue

        cx = int(round((xs_def.min() + xs_def.max()) / 2.0))
        cy = int(round((ys_def.min() + ys_def.max()) / 2.0))
        bbox_w = int(xs_def.max() - xs_def.min() + 1)
        bbox_h = int(ys_def.max() - ys_def.min() + 1)
        defect_area = int(np.count_nonzero(defect_mask))

        geometry_info = {
            "shape_type": shape_type,
            "anchor_xy": [int(x), int(y)],
            "center_xy_float": [float(center[0]), float(center[1])],
            "major_length": int(major_len),
            "minor_length": int(minor_len),
            "detach_gap": int(detach_gap),
            "tangent_shift": float(tangent_shift),
            "local_trace_half_width": float(local_half_width),
            "tangent": [float(tangent_rot[0]), float(tangent_rot[1])],
            "normal": [float(normal_rot[0]), float(normal_rot[1])],
            "elongation": float(elongation),
            "defect_area_px": int(defect_area),
            "defect_bbox_size_px": [int(bbox_w), int(bbox_h)],
            "attempts_used": int(attempt)
        }

        print(f"[INFO] Valid spurious copper found after {attempt} attempts.")
        print(f"[INFO] Shape: {shape_type}")
        print(f"[INFO] Anchor: ({x}, {y})")
        print(f"[INFO] Center: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"[INFO] Major length: {major_len} px")
        print(f"[INFO] Minor length: {minor_len} px")
        print(f"[INFO] Detach gap: {detach_gap} px")
        print(f"[INFO] PCA elongation: {elongation:.2f}")
        print(f"[INFO] Defect area: {defect_area} px")
        print(f"[INFO] Defect bbox size: {bbox_w} x {bbox_h} px")

        return defect_mask, (cx, cy), geometry_info

    raise RuntimeError(
        "Failed to find valid spurious copper location. Try increasing max_attempts or relaxing gap / pad constraints."
    )


# -----------------------------------------------------------------------------
# Rendering and annotation
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


def render_spurious_copper(image_bgr: np.ndarray, trace_mask: np.ndarray,
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

    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


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
    debug[mask_bool] = cv2.addWeighted(debug[mask_bool], 0.4, color_layer[mask_bool], 0.6, 0)

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
    defect_mask, center, geometry_info = sample_valid_spurious_copper(
        image_bgr=image,
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        args=args
    )

    blend_radius = max(geometry_info["minor_length"], 5)

    synthetic_image, trace_color = render_spurious_copper(
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

    suffix = f"spurious_copper_{sample_index:04d}" if args.num_samples > 1 else "spurious_copper"
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
        crop_debug = create_debug_overlay(
            image_bgr=crop_image,
            defect_mask=crop_mask,
            bbox_line=crop_yolo_line
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
        "defect_type": "spurious_copper",
        "class_id": int(args.class_id),
        "sample_index": int(sample_index),
        "center_xy": [int(center[0]), int(center[1])],
        "geometry": geometry_info,
        "trace_color_bgr": trace_color.astype(int).tolist(),
        "full_yolo_label": yolo_line,
        "full_bbox_px": get_bbox_pixels(defect_mask),
        "generation_parameters": vars(args),
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

    print("[SUCCESS] Spurious copper generation completed.")
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
    print(f"[INFO] Shape mode: {args.shape_mode}")
    print(f"[INFO] Number of samples: {args.num_samples}")

    all_metadata = []

    for i in range(1, args.num_samples + 1):
        print(f"\n[INFO] Generating spurious copper sample {i}/{args.num_samples}...")
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
        summary_path = os.path.join(args.output_dir, f"{stem}_spurious_copper_summary.json")
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
