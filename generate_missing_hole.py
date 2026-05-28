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
        description="Generate rule-based missing hole defect on PCB solder pad."
    )

    parser.add_argument("--image", type=str, default=os.path.join("PCB_Dataset", "04.JPG"),
                        help="Path to original PCB image.")
    parser.add_argument("--topology_dir", type=str, default=os.path.join("outputs", "topology"),
                        help="Directory containing trace_mask and attack_candidate_mask.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "missing_hole"),
                        help="Directory to save generated missing hole results.")
    parser.add_argument("--class_id", type=int, default=3,
                        help="YOLO class id for missing hole. Adjust according to your class mapping.")

    # Pad detection
    parser.add_argument("--pad_v_min", type=int, default=115,
                        help="Minimum grayscale value for detecting bright silver pads / solder joints.")
    parser.add_argument("--pad_s_max", type=int, default=105,
                        help="Maximum HSV saturation for detecting silver pads / solder joints.")
    parser.add_argument("--pad_min_area", type=int, default=45,
                        help="Minimum connected-component area for pad candidates.")
    parser.add_argument("--pad_max_area", type=int, default=6000,
                        help="Maximum connected-component area for pad candidates.")
    parser.add_argument("--min_pad_width", type=int, default=7,
                        help="Minimum pad bbox width in pixels.")
    parser.add_argument("--min_pad_height", type=int, default=7,
                        help="Minimum pad bbox height in pixels.")
    parser.add_argument("--border_margin", type=int, default=30,
                        help="Avoid sampling too close to image border.")
    parser.add_argument("--trace_contact_dilate", type=int, default=20,
                        help="Dilation radius for validating pad-to-trace neighborhood support.")

    # Hole geometry
    parser.add_argument("--hole_shape_mode", type=str, default="circle",
                        choices=["circle", "ellipse", "mixed"],
                        help="Missing-hole geometry mode.")
    parser.add_argument("--circle_prob", type=float, default=0.85,
                        help="Probability of choosing circle when hole_shape_mode=mixed.")
    parser.add_argument("--ellipse_prob", type=float, default=0.15,
                        help="Probability of choosing ellipse when hole_shape_mode=mixed.")
    parser.add_argument("--min_hole_radius", type=int, default=5,
                        help="Minimum hole radius in pixels. Larger default prevents tiny dot-like defects.")
    parser.add_argument("--max_hole_radius", type=int, default=13,
                        help="Maximum hole radius in pixels.")
    parser.add_argument("--min_hole_radius_ratio", type=float, default=0.34,
                        help="Minimum hole-size ratio relative to local pad half-width.")
    parser.add_argument("--max_hole_radius_ratio", type=float, default=0.56,
                        help="Maximum hole-size ratio relative to local pad half-width.")
    parser.add_argument("--ellipse_axis_ratio_min", type=float, default=1.00,
                        help="Minimum major/minor axis ratio for ellipse hole.")
    parser.add_argument("--ellipse_axis_ratio_max", type=float, default=1.22,
                        help="Maximum major/minor axis ratio for ellipse hole.")
    parser.add_argument("--center_jitter_px", type=float, default=1.0,
                        help="Maximum random center jitter around pad centroid, in pixels.")
    parser.add_argument("--hole_edge_clearance", type=int, default=1,
                        help="Minimum clearance from generated hole to pad boundary, in pixels.")

    # Fill color control
    parser.add_argument("--green_h_min", type=int, default=35,
                        help="Minimum HSV hue for green trace / solder-mask fill color sampling.")
    parser.add_argument("--green_h_max", type=int, default=90,
                        help="Maximum HSV hue for green trace / solder-mask fill color sampling.")
    parser.add_argument("--green_s_min", type=int, default=45,
                        help="Minimum HSV saturation for green fill color sampling.")
    parser.add_argument("--green_v_min", type=int, default=25,
                        help="Minimum HSV value for green fill color sampling.")
    parser.add_argument("--fill_search_radius", type=int, default=28,
                        help="Local search radius for estimating the green trace color used inside missing hole.")
    parser.add_argument("--prefer_vertical_trace_color", action="store_true",
                        help="Prefer pixels above/below the pad center when estimating fill color.")

    # Validity
    parser.add_argument("--max_attempts", type=int, default=12000,
                        help="Maximum random attempts to find a valid missing hole location.")
    parser.add_argument("--min_defect_area", type=int, default=45,
                        help="Minimum hole area in pixels.")
    parser.add_argument("--min_bbox_width", type=int, default=7,
                        help="Minimum defect bbox width in pixels.")
    parser.add_argument("--min_bbox_height", type=int, default=7,
                        help="Minimum defect bbox height in pixels.")
    parser.add_argument("--min_remaining_pad_ratio", type=float, default=0.42,
                        help="Minimum remaining pad area ratio after carving the hole.")

    # Output
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of missing hole samples to generate from the same image.")
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


def remove_border_area(mask: np.ndarray, margin: int) -> np.ndarray:
    cleaned = mask.copy()
    h, w = cleaned.shape[:2]
    cleaned[:margin, :] = 0
    cleaned[h - margin:, :] = 0
    cleaned[:, :margin] = 0
    cleaned[:, w - margin:] = 0
    return cleaned


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=1)


def normalize_vector(vx: float, vy: float) -> Optional[Tuple[float, float]]:
    norm = float(np.sqrt(vx * vx + vy * vy))
    if norm < 1e-6:
        return None
    return vx / norm, vy / norm


def angle_from_vector(v: Tuple[float, float]) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


# -----------------------------------------------------------------------------
# Pad extraction and geometry analysis
# -----------------------------------------------------------------------------

def create_pad_mask(image_bgr: np.ndarray, v_min: int = 115, s_max: int = 105) -> np.ndarray:
    """
    Detect bright low-saturation silver pads / solder joints.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]

    pad_mask = np.zeros(gray.shape, dtype=np.uint8)
    pad_mask[(gray >= v_min) & (saturation <= s_max)] = 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_OPEN, kernel_open)
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_CLOSE, kernel_close)

    return pad_mask


def collect_pad_candidates(
    pad_mask: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    pad_min_area: int,
    pad_max_area: int,
    min_pad_width: int,
    min_pad_height: int,
    border_margin: int,
    trace_contact_dilate: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Collect pad connected components as candidate missing-hole locations.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pad_mask,
        connectivity=8
    )

    support_mask = dilate_mask(trace_mask, radius=trace_contact_dilate)
    attack_support = dilate_mask(attack_candidate_mask, radius=max(2, trace_contact_dilate // 2))
    support_mask = np.maximum(support_mask, attack_support)

    candidates = []
    height, width = pad_mask.shape[:2]

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < pad_min_area or area > pad_max_area:
            continue
        if w < min_pad_width or h < min_pad_height:
            continue
        if x < border_margin or y < border_margin:
            continue
        if x + w >= width - border_margin:
            continue
        if y + h >= height - border_margin:
            continue

        component_mask = np.zeros_like(pad_mask)
        component_mask[labels == label_id] = 255

        if np.count_nonzero((component_mask > 0) & (support_mask > 0)) == 0:
            continue

        candidates.append({
            "label_id": int(label_id),
            "area": int(area),
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "centroid_xy": [float(centroids[label_id][0]), float(centroids[label_id][1])]
        })

    return labels, candidates


def estimate_component_geometry(component_mask: np.ndarray) -> Optional[Dict[str, Any]]:
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 20:
        return None

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    center = np.mean(pts, axis=0)
    pts_centered = pts - center

    cov = np.cov(pts_centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]

    lambda1 = float(eigenvalues[order[0]])
    lambda2 = float(eigenvalues[order[1]])
    if lambda1 <= 1e-6:
        return None

    major_dir = eigenvectors[:, order[0]]
    minor_dir = eigenvectors[:, order[1]]

    major_dir = normalize_vector(float(major_dir[0]), float(major_dir[1]))
    minor_dir = normalize_vector(float(minor_dir[0]), float(minor_dir[1]))
    if major_dir is None or minor_dir is None:
        return None

    proj_major = pts_centered @ np.array(major_dir, dtype=np.float32)
    proj_minor = pts_centered @ np.array(minor_dir, dtype=np.float32)

    major_half_extent = float(max(abs(np.min(proj_major)), abs(np.max(proj_major))))
    minor_half_extent = float(max(abs(np.min(proj_minor)), abs(np.max(proj_minor))))

    return {
        "center_xy_float": [float(center[0]), float(center[1])],
        "major_dir": [float(major_dir[0]), float(major_dir[1])],
        "minor_dir": [float(minor_dir[0]), float(minor_dir[1])],
        "major_half_extent": float(major_half_extent),
        "minor_half_extent": float(minor_half_extent),
        "elongation": float(lambda1 / max(lambda2, 1e-6)),
        "angle_deg": angle_from_vector(major_dir)
    }


def choose_hole_shape(args) -> str:
    if args.hole_shape_mode != "mixed":
        return args.hole_shape_mode

    probs = np.array([
        max(0.0, float(args.circle_prob)),
        max(0.0, float(args.ellipse_prob))
    ], dtype=np.float64)

    if probs.sum() <= 1e-8:
        probs = np.array([1.0, 1.0], dtype=np.float64)

    probs /= probs.sum()
    return str(np.random.choice(["circle", "ellipse"], p=probs))


def find_safe_hole_center(
    component_mask: np.ndarray,
    preferred_center_xy: Tuple[float, float],
    safe_margin: int
) -> Optional[Tuple[int, int]]:
    safe_region = erode_mask(component_mask, radius=max(0, safe_margin))
    ys, xs = np.where(safe_region > 0)
    if len(xs) == 0:
        return None

    px, py = preferred_center_xy
    dist2 = (xs.astype(np.float32) - float(px)) ** 2 + (ys.astype(np.float32) - float(py)) ** 2
    idx = int(np.argmin(dist2))
    return int(xs[idx]), int(ys[idx])


# -----------------------------------------------------------------------------
# Missing-hole geometry generation
# -----------------------------------------------------------------------------

def make_circle_mask(shape: Tuple[int, int], center_xy: Tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        center=(int(center_xy[0]), int(center_xy[1])),
        radius=int(radius),
        color=255,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def make_ellipse_mask(
    shape: Tuple[int, int],
    center_xy: Tuple[int, int],
    axis_major: int,
    axis_minor: int,
    angle_deg: float
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(int(center_xy[0]), int(center_xy[1])),
        axes=(max(1, int(axis_major)), max(1, int(axis_minor))),
        angle=float(angle_deg),
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def is_valid_missing_hole(
    pad_component_mask: np.ndarray,
    defect_mask: np.ndarray,
    min_defect_area: int = 45,
    min_bbox_width: int = 7,
    min_bbox_height: int = 7,
    min_remaining_pad_ratio: float = 0.42
) -> bool:
    if np.any((defect_mask > 0) & (pad_component_mask == 0)):
        return False

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

    original_pad_area = int(np.count_nonzero(pad_component_mask))
    remaining_pad = pad_component_mask.copy()
    remaining_pad[defect_mask > 0] = 0
    remaining_pad_area = int(np.count_nonzero(remaining_pad))

    if original_pad_area <= 0:
        return False

    remaining_ratio = remaining_pad_area / max(original_pad_area, 1)
    if remaining_ratio < min_remaining_pad_ratio:
        return False

    # A small center hole should not destroy the whole pad.
    if count_connected_components(remaining_pad) < 1:
        return False

    return True


def sample_valid_missing_hole(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    args
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any], np.ndarray]:
    pad_mask = create_pad_mask(
        image_bgr=image_bgr,
        v_min=args.pad_v_min,
        s_max=args.pad_s_max
    )
    pad_mask = remove_border_area(pad_mask, margin=max(0, args.border_margin // 2))

    labels, candidates = collect_pad_candidates(
        pad_mask=pad_mask,
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        pad_min_area=args.pad_min_area,
        pad_max_area=args.pad_max_area,
        min_pad_width=args.min_pad_width,
        min_pad_height=args.min_pad_height,
        border_margin=args.border_margin,
        trace_contact_dilate=args.trace_contact_dilate
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No eligible pad candidates found. Try relaxing pad thresholds, pad area limits, "
            "border_margin, or trace_contact_dilate."
        )

    for attempt in range(1, args.max_attempts + 1):
        candidate = random.choice(candidates)
        label_id = int(candidate["label_id"])

        pad_component_mask = np.zeros_like(pad_mask)
        pad_component_mask[labels == label_id] = 255

        geom = estimate_component_geometry(pad_component_mask)
        if geom is None:
            continue

        cx0, cy0 = geom["center_xy_float"]
        jitter_x = random.uniform(-args.center_jitter_px, args.center_jitter_px)
        jitter_y = random.uniform(-args.center_jitter_px, args.center_jitter_px)

        safe_center = find_safe_hole_center(
            component_mask=pad_component_mask,
            preferred_center_xy=(cx0 + jitter_x, cy0 + jitter_y),
            safe_margin=max(1, args.hole_edge_clearance + 1)
        )
        if safe_center is None:
            continue

        center_x, center_y = safe_center

        dist_inside_pad = cv2.distanceTransform(pad_component_mask, cv2.DIST_L2, 5)
        max_allowed_radius = float(dist_inside_pad[center_y, center_x]) - float(args.hole_edge_clearance)
        if max_allowed_radius < float(args.min_hole_radius):
            continue

        local_pad_half_width = min(
            float(geom["major_half_extent"]),
            float(geom["minor_half_extent"])
        )
        radius_ratio = random.uniform(float(args.min_hole_radius_ratio), float(args.max_hole_radius_ratio))
        base_radius = local_pad_half_width * radius_ratio
        base_radius = min(base_radius, max_allowed_radius)
        if base_radius < float(args.min_hole_radius):
            continue

        shape_type = choose_hole_shape(args)
        defect_mask_full = None
        shape_info = {}

        if shape_type == "circle":
            radius = int(round(base_radius))
            radius = int(np.clip(
                radius,
                args.min_hole_radius,
                min(args.max_hole_radius, max(args.min_hole_radius, int(np.floor(max_allowed_radius))))
            ))
            defect_mask_full = make_circle_mask(
                shape=pad_component_mask.shape,
                center_xy=(center_x, center_y),
                radius=radius
            )
            shape_info = {
                "shape_type": "circle",
                "radius": int(radius)
            }

        elif shape_type == "ellipse":
            minor_axis = max(float(args.min_hole_radius), base_radius)
            axis_ratio = random.uniform(float(args.ellipse_axis_ratio_min), float(args.ellipse_axis_ratio_max))
            major_axis = minor_axis * axis_ratio

            major_axis = min(major_axis, max_allowed_radius)
            minor_axis = min(minor_axis, max_allowed_radius)

            axis_major_int = int(round(np.clip(major_axis, args.min_hole_radius, args.max_hole_radius)))
            axis_minor_int = int(round(np.clip(minor_axis, args.min_hole_radius, args.max_hole_radius)))

            defect_mask_full = make_ellipse_mask(
                shape=pad_component_mask.shape,
                center_xy=(center_x, center_y),
                axis_major=axis_major_int,
                axis_minor=axis_minor_int,
                angle_deg=float(geom["angle_deg"])
            )
            shape_info = {
                "shape_type": "ellipse",
                "axis_major": int(axis_major_int),
                "axis_minor": int(axis_minor_int),
                "angle_deg": float(geom["angle_deg"])
            }

        else:
            continue

        safe_region = erode_mask(pad_component_mask, radius=max(0, args.hole_edge_clearance))
        defect_mask = defect_mask_full.copy()
        defect_mask[safe_region == 0] = 0

        original_full_area = int(np.count_nonzero(defect_mask_full))
        clipped_area = int(np.count_nonzero(defect_mask))
        if original_full_area <= 0:
            continue
        if clipped_area / max(original_full_area, 1) < 0.90:
            continue

        if not is_valid_missing_hole(
            pad_component_mask=pad_component_mask,
            defect_mask=defect_mask,
            min_defect_area=args.min_defect_area,
            min_bbox_width=args.min_bbox_width,
            min_bbox_height=args.min_bbox_height,
            min_remaining_pad_ratio=args.min_remaining_pad_ratio
        ):
            continue

        ys_def, xs_def = np.where(defect_mask > 0)
        cx = int(round((xs_def.min() + xs_def.max()) / 2.0))
        cy = int(round((ys_def.min() + ys_def.max()) / 2.0))
        bbox_w = int(xs_def.max() - xs_def.min() + 1)
        bbox_h = int(ys_def.max() - ys_def.min() + 1)
        defect_area = int(np.count_nonzero(defect_mask))

        pad_ys, pad_xs = np.where(pad_component_mask > 0)
        bbox_pad = {
            "x_min": int(pad_xs.min()),
            "y_min": int(pad_ys.min()),
            "x_max": int(pad_xs.max()),
            "y_max": int(pad_ys.max()),
            "width": int(pad_xs.max() - pad_xs.min() + 1),
            "height": int(pad_ys.max() - pad_ys.min() + 1)
        }

        geometry_info = {
            "pad_label_id": int(label_id),
            "pad_area_px": int(candidate["area"]),
            "pad_bbox_xywh": [int(v) for v in candidate["bbox_xywh"]],
            "pad_bbox_px": bbox_pad,
            "pad_centroid_xy": [float(candidate["centroid_xy"][0]), float(candidate["centroid_xy"][1])],
            "pad_geometry": {
                "center_xy_float": [float(v) for v in geom["center_xy_float"]],
                "major_dir": [float(v) for v in geom["major_dir"]],
                "minor_dir": [float(v) for v in geom["minor_dir"]],
                "major_half_extent": float(geom["major_half_extent"]),
                "minor_half_extent": float(geom["minor_half_extent"]),
                "elongation": float(geom["elongation"]),
                "angle_deg": float(geom["angle_deg"])
            },
            "hole_center_xy": [int(center_x), int(center_y)],
            "hole_edge_clearance": int(args.hole_edge_clearance),
            "local_max_allowed_radius": float(max_allowed_radius),
            "defect_area_px": int(defect_area),
            "defect_bbox_size_px": [int(bbox_w), int(bbox_h)],
            "attempts_used": int(attempt),
            **shape_info
        }

        print(f"[INFO] Valid missing hole found after {attempt} attempts.")
        print(f"[INFO] Pad label id: {label_id}")
        print(f"[INFO] Pad area: {candidate['area']} px")
        print(f"[INFO] Hole center: ({center_x}, {center_y})")
        print(f"[INFO] Shape: {shape_info['shape_type']}")
        print(f"[INFO] Defect area: {defect_area} px")
        print(f"[INFO] Defect bbox size: {bbox_w} x {bbox_h} px")

        return defect_mask, (cx, cy), geometry_info, pad_mask

    raise RuntimeError(
        "Failed to find valid missing hole location. "
        "Try increasing max_attempts or relaxing pad / hole constraints."
    )


# -----------------------------------------------------------------------------
# Rendering and annotation helpers
# -----------------------------------------------------------------------------

def make_green_candidate_mask(
    image_bgr: np.ndarray,
    pad_mask: np.ndarray,
    green_h_min: int,
    green_h_max: int,
    green_s_min: int,
    green_v_min: int
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[(h >= green_h_min) & (h <= green_h_max) &
         (s >= green_s_min) & (v >= green_v_min) &
         (pad_mask == 0)] = 255
    return mask


def robust_median_color(image_bgr: np.ndarray, mask: np.ndarray, min_pixels: int = 8) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    pixels = image_bgr[ys, xs]
    return np.median(pixels, axis=0).astype(np.uint8)


def get_local_hole_fill_color(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    pad_mask: np.ndarray,
    defect_mask: np.ndarray,
    geometry_info: Dict[str, Any],
    radius: int,
    green_h_min: int,
    green_h_max: int,
    green_s_min: int,
    green_v_min: int,
    prefer_vertical_trace_color: bool = False
) -> np.ndarray:
    """
    Estimate the color used inside the missing hole.

    The target style is a non-silver green center, similar to the trace / solder-mask
    color visible around the pad. This function intentionally excludes silver pad
    pixels, then prioritizes green pixels in the local neighborhood.
    """
    h_img, w_img = pad_mask.shape[:2]
    radius = max(6, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

    green_mask = make_green_candidate_mask(
        image_bgr=image_bgr,
        pad_mask=pad_mask,
        green_h_min=green_h_min,
        green_h_max=green_h_max,
        green_s_min=green_s_min,
        green_v_min=green_v_min
    )

    local = cv2.dilate(defect_mask, kernel, iterations=1)
    local[defect_mask > 0] = 0

    cx, cy = geometry_info.get("hole_center_xy", [None, None])
    pad_bbox = geometry_info.get("pad_bbox_px", None)

    if cx is not None and cy is not None and pad_bbox is not None:
        # Prefer a narrow strip through the pad center. For the common DIP-like
        # examples, this samples the green trace segment directly above/below
        # the silver pad instead of the silver pad itself.
        strip_half_width = max(2, int(round(0.18 * max(pad_bbox["width"], pad_bbox["height"]))))
        strip = np.zeros_like(pad_mask)
        x1 = max(0, int(cx) - strip_half_width)
        x2 = min(w_img, int(cx) + strip_half_width + 1)
        y1 = max(0, int(cy) - radius * 2)
        y2 = min(h_img, int(cy) + radius * 2 + 1)
        strip[y1:y2, x1:x2] = 255

        if not prefer_vertical_trace_color:
            # Also allow a horizontal strip for rotated pads or traces.
            y1h = max(0, int(cy) - strip_half_width)
            y2h = min(h_img, int(cy) + strip_half_width + 1)
            x1h = max(0, int(cx) - radius * 2)
            x2h = min(w_img, int(cx) + radius * 2 + 1)
            strip[y1h:y2h, x1h:x2h] = 255

        candidate = np.zeros_like(pad_mask)
        candidate[(strip > 0) & (green_mask > 0)] = 255
        color = robust_median_color(image_bgr, candidate, min_pixels=6)
        if color is not None:
            return color

    # Nearby green pixels around the missing-hole region.
    candidate = np.zeros_like(pad_mask)
    candidate[(local > 0) & (green_mask > 0)] = 255
    color = robust_median_color(image_bgr, candidate, min_pixels=8)
    if color is not None:
        return color

    # Nearby trace-mask pixels, excluding silver pad pixels.
    candidate = np.zeros_like(pad_mask)
    candidate[(local > 0) & (trace_mask > 0) & (pad_mask == 0)] = 255
    color = robust_median_color(image_bgr, candidate, min_pixels=6)
    if color is not None:
        return color

    # Global green pixels.
    color = robust_median_color(image_bgr, green_mask, min_pixels=20)
    if color is not None:
        return color

    # Global non-pad trace pixels.
    candidate = np.zeros_like(pad_mask)
    candidate[(trace_mask > 0) & (pad_mask == 0)] = 255
    color = robust_median_color(image_bgr, candidate, min_pixels=10)
    if color is not None:
        return color

    return np.array([30, 110, 30], dtype=np.uint8)


def render_missing_hole(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    pad_mask: np.ndarray,
    defect_mask: np.ndarray,
    geometry_info: Dict[str, Any],
    args,
    blend_radius: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render missing hole by replacing the pad center with local green trace color.
    """
    output = image_bgr.copy()

    fill_color = get_local_hole_fill_color(
        image_bgr=image_bgr,
        trace_mask=trace_mask,
        pad_mask=pad_mask,
        defect_mask=defect_mask,
        geometry_info=geometry_info,
        radius=max(blend_radius, int(args.fill_search_radius)),
        green_h_min=args.green_h_min,
        green_h_max=args.green_h_max,
        green_s_min=args.green_s_min,
        green_v_min=args.green_v_min,
        prefer_vertical_trace_color=bool(args.prefer_vertical_trace_color)
    )

    output[defect_mask > 0] = fill_color

    # Light edge feathering only. Do not blur the center; the center should stay
    # clearly green, not silver or a tiny red/debug-colored dot.
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.dilate(defect_mask, boundary_kernel, iterations=1)
    boundary = cv2.subtract(boundary, cv2.erode(defect_mask, boundary_kernel, iterations=1))

    feather = cv2.GaussianBlur(output, (3, 3), 0)
    output[(boundary > 0) & (defect_mask == 0)] = feather[(boundary > 0) & (defect_mask == 0)]

    return output, fill_color


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


def create_debug_overlay(image_bgr: np.ndarray, defect_mask: np.ndarray, bbox_line: Optional[str]) -> np.ndarray:
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

def generate_one_sample(
    args,
    image: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    stem: str,
    sample_index: int
) -> Dict[str, Any]:
    defect_mask, center, geometry_info, pad_mask = sample_valid_missing_hole(
        image_bgr=image,
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        args=args
    )

    blend_radius = max(10, int(round(np.sqrt(max(geometry_info["defect_area_px"], 1)))) + 6)

    synthetic_image, fill_color = render_missing_hole(
        image_bgr=image,
        trace_mask=trace_mask,
        pad_mask=pad_mask,
        defect_mask=defect_mask,
        geometry_info=geometry_info,
        args=args,
        blend_radius=blend_radius
    )

    yolo_line = mask_to_yolo_bbox(defect_mask=defect_mask, class_id=args.class_id)
    debug_overlay = create_debug_overlay(
        image_bgr=synthetic_image,
        defect_mask=defect_mask,
        bbox_line=yolo_line
    )

    suffix = f"missing_hole_{sample_index:04d}" if args.num_samples > 1 else "missing_hole"
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
        "defect_type": "missing_hole",
        "class_id": int(args.class_id),
        "sample_index": int(sample_index),
        "center_xy": [int(center[0]), int(center[1])],
        "geometry": geometry_info,
        "fill_color_bgr": fill_color.astype(int).tolist(),
        "full_yolo_label": yolo_line,
        "full_bbox_px": get_bbox_pixels(defect_mask),
        "generation_parameters": {
            "pad_v_min": int(args.pad_v_min),
            "pad_s_max": int(args.pad_s_max),
            "pad_min_area": int(args.pad_min_area),
            "pad_max_area": int(args.pad_max_area),
            "min_pad_width": int(args.min_pad_width),
            "min_pad_height": int(args.min_pad_height),
            "border_margin": int(args.border_margin),
            "trace_contact_dilate": int(args.trace_contact_dilate),
            "hole_shape_mode": str(args.hole_shape_mode),
            "circle_prob": float(args.circle_prob),
            "ellipse_prob": float(args.ellipse_prob),
            "min_hole_radius": int(args.min_hole_radius),
            "max_hole_radius": int(args.max_hole_radius),
            "min_hole_radius_ratio": float(args.min_hole_radius_ratio),
            "max_hole_radius_ratio": float(args.max_hole_radius_ratio),
            "ellipse_axis_ratio_min": float(args.ellipse_axis_ratio_min),
            "ellipse_axis_ratio_max": float(args.ellipse_axis_ratio_max),
            "center_jitter_px": float(args.center_jitter_px),
            "hole_edge_clearance": int(args.hole_edge_clearance),
            "green_h_min": int(args.green_h_min),
            "green_h_max": int(args.green_h_max),
            "green_s_min": int(args.green_s_min),
            "green_v_min": int(args.green_v_min),
            "fill_search_radius": int(args.fill_search_radius),
            "prefer_vertical_trace_color": bool(args.prefer_vertical_trace_color),
            "max_attempts": int(args.max_attempts),
            "min_defect_area": int(args.min_defect_area),
            "min_bbox_width": int(args.min_bbox_width),
            "min_bbox_height": int(args.min_bbox_height),
            "min_remaining_pad_ratio": float(args.min_remaining_pad_ratio),
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

    print("[SUCCESS] Missing hole generation completed.")
    print(f"  Synthetic image: {synthetic_path}")
    print(f"  Defect mask:     {defect_mask_path}")
    print(f"  Debug overlay:   {debug_path}")
    print(f"  YOLO label:      {label_path}")
    if crop_path is not None:
        print(f"  Crop image:      {crop_path}")
        print(f"  Crop mask:       {crop_mask_path}")
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
    print(f"[INFO] Hole shape mode: {args.hole_shape_mode}")
    print(f"[INFO] Number of samples: {args.num_samples}")

    all_metadata = []

    for i in range(1, args.num_samples + 1):
        print(f"\n[INFO] Generating missing hole sample {i}/{args.num_samples}...")
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
        summary_path = os.path.join(args.output_dir, f"{stem}_missing_hole_summary.json")
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
