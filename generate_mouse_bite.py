import cv2
import numpy as np
import os
import argparse
import random
import json
from datetime import datetime
from typing import Tuple, Optional, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rule-based mouse bite defect on PCB trace."
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
        default=os.path.join("outputs", "mouse_bite"),
        help="Directory to save generated mouse bite results."
    )

    parser.add_argument(
        "--class_id",
        type=int,
        default=4,
        help="YOLO class id for mouse bite. Adjust according to your class mapping."
    )

    parser.add_argument(
        "--min_radius",
        type=int,
        default=10,
        help="Minimum mouse bite radius in pixels."
    )

    parser.add_argument(
        "--max_radius",
        type=int,
        default=22,
        help="Maximum mouse bite radius in pixels."
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3000,
        help="Maximum random attempts to find a valid bite location."
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of mouse bite samples to generate from the same image."
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="Defect-centered crop size. Use 0 to disable crop output."
    )

    parser.add_argument(
        "--min_overlap_area",
        type=int,
        default=30,
        help="Minimum defect area in pixels."
    )

    parser.add_argument(
        "--min_bbox_width",
        type=int,
        default=6,
        help="Minimum defect bbox width in pixels."
    )

    parser.add_argument(
        "--min_bbox_height",
        type=int,
        default=4,
        help="Minimum defect bbox height in pixels."
    )

    parser.add_argument(
        "--max_overlap_ratio",
        type=float,
        default=0.85,
        help="Maximum ratio of bite circle overlapping the trace."
    )

    return parser.parse_args()


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


def keep_large_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Keep only large connected components.

    This helps avoid sampling mouse bite locations from isolated text such as
    P0 / P1 / P2 / P3 / DOWNLOADS.
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


def get_edge_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Extract inner edge of trace mask.

    Edge = mask - eroded(mask)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    edge = cv2.subtract(binary_mask, eroded)

    return edge


def count_connected_components(binary_mask: np.ndarray) -> int:
    """
    Count foreground connected components.
    """
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    return num_labels - 1


def make_circular_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def get_mask_bbox_pixels(defect_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return bbox as pixel coordinates: x_min, y_min, x_max, y_max.
    """
    ys, xs = np.where(defect_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def is_valid_mouse_bite(
    trace_mask: np.ndarray,
    bite_circle: np.ndarray,
    defect_mask: np.ndarray,
    min_overlap_area: int = 30,
    min_bbox_width: int = 6,
    min_bbox_height: int = 4,
    max_overlap_ratio: float = 0.85
) -> bool:
    """
    Validate mouse bite geometry.

    Requirements:
    1. Defect area must be large enough.
    2. Defect bounding box must be visually meaningful.
    3. Bite should not be mostly inside trace.
    4. Bite should not become an open circuit.
    """
    overlap_area = int(np.count_nonzero(defect_mask))
    circle_area = int(np.count_nonzero(bite_circle))

    if circle_area == 0:
        return False

    overlap_ratio = overlap_area / float(circle_area)

    if overlap_area < min_overlap_area:
        return False

    bbox = get_mask_bbox_pixels(defect_mask)

    if bbox is None:
        return False

    x_min, y_min, x_max, y_max = bbox
    bbox_width = int(x_max - x_min + 1)
    bbox_height = int(y_max - y_min + 1)

    if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
        return False

    if overlap_ratio > max_overlap_ratio:
        return False

    before_cc = count_connected_components(trace_mask)

    after_mask = trace_mask.copy()
    after_mask[defect_mask > 0] = 0

    after_cc = count_connected_components(after_mask)

    # Mouse bite should not fully cut the trace. If it does, it is closer to open circuit.
    if after_cc > before_cc:
        return False

    return True


def sample_valid_bite(
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    min_radius: int,
    max_radius: int,
    max_attempts: int,
    component_min_area: int,
    border_margin: int,
    min_overlap_area: int,
    min_bbox_width: int,
    min_bbox_height: int,
    max_overlap_ratio: float
) -> Tuple[np.ndarray, Tuple[int, int], int, Dict[str, Any]]:
    """
    Randomly sample a valid mouse bite defect.

    Returns:
        defect_mask, center, radius, geometry_metadata
    """
    eligible = keep_large_components(
        attack_candidate_mask,
        min_area=component_min_area
    )

    eligible = remove_border_area(eligible, margin=border_margin)

    edge = get_edge_mask(eligible)

    ys, xs = np.where(edge > 0)

    if len(xs) == 0:
        raise RuntimeError(
            "No eligible edge pixels found. Try reducing component_min_area or border_margin."
        )

    h, w = trace_mask.shape[:2]

    for attempt in range(max_attempts):
        idx = random.randint(0, len(xs) - 1)
        x = int(xs[idx])
        y = int(ys[idx])

        radius = random.randint(min_radius, max_radius)

        if x - radius < border_margin or x + radius >= w - border_margin:
            continue

        if y - radius < border_margin or y + radius >= h - border_margin:
            continue

        bite_circle = make_circular_mask(
            shape=trace_mask.shape,
            center=(x, y),
            radius=radius
        )

        defect_mask = cv2.bitwise_and(bite_circle, trace_mask)

        if is_valid_mouse_bite(
            trace_mask=trace_mask,
            bite_circle=bite_circle,
            defect_mask=defect_mask,
            min_overlap_area=min_overlap_area,
            min_bbox_width=min_bbox_width,
            min_bbox_height=min_bbox_height,
            max_overlap_ratio=max_overlap_ratio
        ):
            bbox = get_mask_bbox_pixels(defect_mask)
            assert bbox is not None

            x_min, y_min, x_max, y_max = bbox
            defect_area = int(np.count_nonzero(defect_mask))
            circle_area = int(np.count_nonzero(bite_circle))
            overlap_ratio = defect_area / float(circle_area)

            metadata = {
                "attempts_used": attempt + 1,
                "center": [int(x), int(y)],
                "radius": int(radius),
                "defect_area_px": int(defect_area),
                "circle_area_px": int(circle_area),
                "overlap_ratio": float(overlap_ratio),
                "bbox_xyxy_px": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "bbox_width_px": int(x_max - x_min + 1),
                "bbox_height_px": int(y_max - y_min + 1)
            }

            print(f"[INFO] Valid mouse bite found after {attempt + 1} attempts.")
            print(f"[INFO] Defect area: {defect_area} px")
            print(f"[INFO] Defect bbox size: {metadata['bbox_width_px']} x {metadata['bbox_height_px']} px")

            return defect_mask, (x, y), radius, metadata

    raise RuntimeError(
        "Failed to find valid mouse bite location. "
        "Try increasing max_attempts or adjusting radius/component_min_area."
    )


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


def render_mouse_bite(
    image_bgr: np.ndarray,
    trace_mask: np.ndarray,
    defect_mask: np.ndarray,
    radius: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render mouse bite by replacing removed trace pixels with local substrate color.

    Returns:
        synthetic image, substrate_color_bgr
    """
    output = image_bgr.copy()

    substrate_color = get_local_substrate_color(
        image_bgr=image_bgr,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        radius=radius
    )

    output[defect_mask > 0] = substrate_color

    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary = cv2.dilate(defect_mask, boundary_kernel, iterations=1)
    boundary = cv2.subtract(boundary, cv2.erode(defect_mask, boundary_kernel, iterations=1))

    blurred = cv2.GaussianBlur(output, (5, 5), 0)

    boundary_bool = boundary > 0
    output[boundary_bool] = blurred[boundary_bool]

    return output, substrate_color


def mask_to_yolo_bbox(
    defect_mask: np.ndarray,
    class_id: int
) -> Optional[str]:
    bbox = get_mask_bbox_pixels(defect_mask)

    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox
    h, w = defect_mask.shape[:2]

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
    """
    Crop a fixed-size patch around the defect mask.

    Returns:
        cropped image,
        cropped defect mask,
        crop_box: x1, y1, x2, y2
    """
    bbox = get_mask_bbox_pixels(defect_mask)

    if bbox is None:
        raise ValueError("Cannot crop around empty defect mask.")

    x_min, y_min, x_max, y_max = bbox

    h, w = defect_mask.shape[:2]

    cx = int(round((x_min + x_max) / 2.0))
    cy = int(round((y_min + y_max) / 2.0))

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

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    cropped_image = image_bgr[y1:y2, x1:x2]
    cropped_mask = defect_mask[y1:y2, x1:x2]

    return cropped_image, cropped_mask, (x1, y1, x2, y2)


def create_crop_debug_overlay(
    crop_bgr: np.ndarray,
    crop_defect_mask: np.ndarray
) -> np.ndarray:
    """
    Create debug overlay for cropped defect patch.
    """
    crop_yolo_line = mask_to_yolo_bbox(crop_defect_mask, class_id=0)
    # Use class_id=0 only to reuse bbox drawing coordinates; class id is ignored visually.
    return create_debug_overlay(crop_bgr, crop_defect_mask, crop_yolo_line)


def write_text_file(path: str, content: Optional[str]):
    with open(path, "w", encoding="utf-8") as f:
        if content is not None:
            f.write(content + "\n")


def write_json_file(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def build_output_paths(output_dir: str, stem: str, sample_index: int, num_samples: int) -> Dict[str, str]:
    """
    Build output file paths. For one sample, keep backward-compatible names.
    For multiple samples, append _0001, _0002, ...
    """
    if num_samples == 1:
        base = f"{stem}_mouse_bite"
    else:
        base = f"{stem}_mouse_bite_{sample_index:04d}"

    return {
        "synthetic": os.path.join(output_dir, f"{base}.png"),
        "mask": os.path.join(output_dir, f"{base}_mask.png"),
        "debug": os.path.join(output_dir, f"{base}_debug.png"),
        "label": os.path.join(output_dir, f"{base}.txt"),
        "crop": os.path.join(output_dir, f"{base}_crop.png"),
        "crop_mask": os.path.join(output_dir, f"{base}_crop_mask.png"),
        "crop_debug": os.path.join(output_dir, f"{base}_crop_debug.png"),
        "crop_label": os.path.join(output_dir, f"{base}_crop.txt"),
        "metadata": os.path.join(output_dir, f"{base}_metadata.json")
    }


def generate_one_sample(
    image: np.ndarray,
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    args,
    stem: str,
    sample_index: int
):
    defect_mask, center, radius, geometry_meta = sample_valid_bite(
        trace_mask=trace_mask,
        attack_candidate_mask=attack_candidate_mask,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        max_attempts=args.max_attempts,
        component_min_area=args.component_min_area,
        border_margin=args.border_margin,
        min_overlap_area=args.min_overlap_area,
        min_bbox_width=args.min_bbox_width,
        min_bbox_height=args.min_bbox_height,
        max_overlap_ratio=args.max_overlap_ratio
    )

    synthetic_image, substrate_color = render_mouse_bite(
        image_bgr=image,
        trace_mask=trace_mask,
        defect_mask=defect_mask,
        radius=radius
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

    crop_info = None
    crop_yolo_line = None

    if args.crop_size and args.crop_size > 0:
        crop_image, crop_mask, crop_box = crop_around_mask(
            image_bgr=synthetic_image,
            defect_mask=defect_mask,
            crop_size=args.crop_size
        )

        crop_debug = create_crop_debug_overlay(
            crop_bgr=crop_image,
            crop_defect_mask=crop_mask
        )

        crop_yolo_line = mask_to_yolo_bbox(
            defect_mask=crop_mask,
            class_id=args.class_id
        )

        crop_info = {
            "crop_box_xyxy_px": [int(v) for v in crop_box],
            "crop_size": int(args.crop_size),
            "crop_yolo_line": crop_yolo_line
        }
    else:
        crop_image = None
        crop_mask = None
        crop_debug = None

    paths = build_output_paths(
        output_dir=args.output_dir,
        stem=stem,
        sample_index=sample_index,
        num_samples=args.num_samples
    )

    save_image(paths["synthetic"], synthetic_image)
    save_image(paths["mask"], defect_mask)
    save_image(paths["debug"], debug_overlay)
    write_text_file(paths["label"], yolo_line)

    if args.crop_size and args.crop_size > 0:
        save_image(paths["crop"], crop_image)
        save_image(paths["crop_mask"], crop_mask)
        save_image(paths["crop_debug"], crop_debug)
        write_text_file(paths["crop_label"], crop_yolo_line)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": args.image,
        "defect_type": "mouse_bite",
        "class_id": int(args.class_id),
        "sample_index": int(sample_index),
        "seed": None if args.seed is None else int(args.seed),
        "center": [int(center[0]), int(center[1])],
        "radius": int(radius),
        "substrate_color_bgr": [int(v) for v in substrate_color.tolist()],
        "geometry": geometry_meta,
        "yolo_line": yolo_line,
        "crop": crop_info,
        "parameters": {
            "min_radius": int(args.min_radius),
            "max_radius": int(args.max_radius),
            "max_attempts": int(args.max_attempts),
            "component_min_area": int(args.component_min_area),
            "border_margin": int(args.border_margin),
            "min_overlap_area": int(args.min_overlap_area),
            "min_bbox_width": int(args.min_bbox_width),
            "min_bbox_height": int(args.min_bbox_height),
            "max_overlap_ratio": float(args.max_overlap_ratio)
        },
        "outputs": paths
    }

    write_json_file(paths["metadata"], metadata)

    print("[SUCCESS] Mouse bite sample generated.")
    print(f"  Synthetic image: {paths['synthetic']}")
    print(f"  Defect mask:     {paths['mask']}")
    print(f"  Debug overlay:   {paths['debug']}")
    print(f"  YOLO label:      {paths['label']}")
    if args.crop_size and args.crop_size > 0:
        print(f"  Crop image:      {paths['crop']}")
        print(f"  Crop debug:      {paths['crop_debug']}")
        print(f"  Crop label:      {paths['crop_label']}")
    print(f"  Metadata:        {paths['metadata']}")
    print(f"  Center:          {center}")
    print(f"  Radius:          {radius}")
    print(f"  YOLO line:       {yolo_line}")



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

    for sample_index in range(1, args.num_samples + 1):
        print(f"\n[INFO] Generating sample {sample_index}/{args.num_samples}")

        # Make samples reproducible but different when using --num_samples.
        if args.seed is not None:
            per_sample_seed = args.seed + sample_index - 1
            random.seed(per_sample_seed)
            np.random.seed(per_sample_seed)

        generate_one_sample(
            image=image,
            trace_mask=trace_mask,
            attack_candidate_mask=attack_candidate_mask,
            args=args,
            stem=stem,
            sample_index=sample_index
        )


if __name__ == "__main__":
    main()
