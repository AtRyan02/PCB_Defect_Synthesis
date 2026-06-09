import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_IMAGE_NAMES = [
    "01.JPG", "04.JPG", "05.JPG", "06.JPG", "07.JPG",
    "08.JPG", "09.JPG", "10.JPG", "11.JPG", "12.JPG",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate normal/background PCB crops with empty YOLO labels."
    )

    parser.add_argument("--dataset_dir", type=str, default="PCB_Dataset")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGE_NAMES)
    parser.add_argument("--topology_dir", type=str, default=os.path.join("outputs", "topology"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "normal"))

    parser.add_argument("--num_crops_per_image", type=int, default=20)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts_per_image", type=int, default=5000)

    parser.add_argument("--min_trace_ratio", type=float, default=0.03)
    parser.add_argument("--max_trace_ratio", type=float, default=0.65)
    parser.add_argument("--min_structure_ratio", type=float, default=0.05)
    parser.add_argument("--max_structure_ratio", type=float, default=0.85)
    parser.add_argument("--min_dark_feature_ratio", type=float, default=0.001)
    parser.add_argument("--hard_negative_prob", type=float, default=0.65)
    parser.add_argument("--border_margin", type=int, default=8)

    parser.add_argument("--no_debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def resolve_image_paths(dataset_dir: str, images: List[str]) -> List[str]:
    return [item if os.path.exists(item) else os.path.join(dataset_dir, item) for item in images]


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def load_binary_mask(path: str, shape_hw: Tuple[int, int], allow_missing: bool = False) -> np.ndarray:
    if not os.path.exists(path):
        if allow_missing:
            return np.zeros(shape_hw, dtype=np.uint8)
        raise FileNotFoundError(f"Missing topology mask: {path}")

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if allow_missing:
            return np.zeros(shape_hw, dtype=np.uint8)
        raise FileNotFoundError(f"Cannot read mask: {path}")

    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def save_image(path: str, image: np.ndarray):
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def write_empty_label(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def write_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def create_pad_mask(image_bgr: np.ndarray, v_min: int = 110, s_max: int = 95) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[(gray >= v_min) & (saturation <= s_max)] = 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def create_dark_feature_mask(image_bgr: np.ndarray, trace_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark = np.zeros_like(gray, dtype=np.uint8)
    dark[gray <= 65] = 255

    trace_dilated = cv2.dilate(
        trace_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    dark[trace_dilated > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    filtered = np.zeros_like(dark)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= 250 and w <= 30 and h <= 30:
            filtered[labels == label_id] = 255

    return filtered


def make_structure_mask(
    trace_mask: np.ndarray,
    attack_candidate_mask: np.ndarray,
    pad_mask: np.ndarray,
    dark_feature_mask: np.ndarray,
) -> np.ndarray:
    structure = np.zeros_like(trace_mask, dtype=np.uint8)
    for mask in [trace_mask, attack_candidate_mask, pad_mask, dark_feature_mask]:
        structure[mask > 0] = 255
    return structure


def crop_array(arr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return arr[y1:y2, x1:x2]


def random_crop_box(image_shape_hw: Tuple[int, int], crop_size: int, border_margin: int) -> Tuple[int, int, int, int]:
    h, w = image_shape_hw
    crop_size = int(crop_size)
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} is larger than image size {(w, h)}")

    half = crop_size // 2
    min_cx = max(half, half + border_margin)
    max_cx = min(w - (crop_size - half), w - (crop_size - half) - border_margin)
    min_cy = max(half, half + border_margin)
    max_cy = min(h - (crop_size - half), h - (crop_size - half) - border_margin)

    if min_cx > max_cx or min_cy > max_cy:
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        return x1, y1, x1 + crop_size, y1 + crop_size

    cx = random.randint(min_cx, max_cx)
    cy = random.randint(min_cy, max_cy)
    x1 = cx - half
    y1 = cy - half
    return int(x1), int(y1), int(x1 + crop_size), int(y1 + crop_size)


def compute_crop_stats(
    trace_crop: np.ndarray,
    structure_crop: np.ndarray,
    pad_crop: np.ndarray,
    dark_crop: np.ndarray,
) -> Dict[str, float]:
    area = float(trace_crop.shape[0] * trace_crop.shape[1])
    return {
        "trace_ratio": float(np.count_nonzero(trace_crop)) / max(area, 1.0),
        "structure_ratio": float(np.count_nonzero(structure_crop)) / max(area, 1.0),
        "pad_ratio": float(np.count_nonzero(pad_crop)) / max(area, 1.0),
        "dark_feature_ratio": float(np.count_nonzero(dark_crop)) / max(area, 1.0),
    }


def is_valid_normal_crop(stats: Dict[str, float], args) -> bool:
    return (
        args.min_trace_ratio <= stats["trace_ratio"] <= args.max_trace_ratio
        and args.min_structure_ratio <= stats["structure_ratio"] <= args.max_structure_ratio
    )


def is_hard_negative_crop(stats: Dict[str, float], args) -> bool:
    return stats["dark_feature_ratio"] >= args.min_dark_feature_ratio or stats["pad_ratio"] >= 0.01


def score_crop(stats: Dict[str, float]) -> float:
    return (
        stats["trace_ratio"] * 2.0
        + stats["structure_ratio"]
        + stats["pad_ratio"] * 2.0
        + stats["dark_feature_ratio"] * 3.0
    )


def make_debug_crop(crop_bgr: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    debug = crop_bgr.copy()
    text = (
        f"normal tr={stats['trace_ratio']:.2f} "
        f"pad={stats['pad_ratio']:.2f} dark={stats['dark_feature_ratio']:.3f}"
    )
    cv2.putText(debug, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, debug.shape[0] - 1), (0, 255, 255), 1)
    return debug


def sample_normal_crops_for_image(image_path: str, args) -> List[Dict]:
    stem = get_file_stem(image_path)
    image_bgr = load_bgr(image_path)
    h, w = image_bgr.shape[:2]
    shape_hw = (h, w)

    trace_mask_path = os.path.join(args.topology_dir, f"{stem}_trace_mask.png")
    attack_mask_path = os.path.join(args.topology_dir, f"{stem}_attack_candidate_mask.png")

    trace_mask = load_binary_mask(trace_mask_path, shape_hw=shape_hw, allow_missing=False)
    attack_candidate_mask = load_binary_mask(attack_mask_path, shape_hw=shape_hw, allow_missing=True)

    pad_mask = create_pad_mask(image_bgr)
    dark_feature_mask = create_dark_feature_mask(image_bgr, trace_mask)
    structure_mask = make_structure_mask(trace_mask, attack_candidate_mask, pad_mask, dark_feature_mask)

    accepted = []
    fallback_candidates = []

    attempts = 0
    while attempts < args.max_attempts_per_image and len(accepted) < args.num_crops_per_image:
        attempts += 1
        box = random_crop_box(shape_hw, args.crop_size, args.border_margin)

        trace_crop = crop_array(trace_mask, box)
        structure_crop = crop_array(structure_mask, box)
        pad_crop = crop_array(pad_mask, box)
        dark_crop = crop_array(dark_feature_mask, box)

        stats = compute_crop_stats(trace_crop, structure_crop, pad_crop, dark_crop)

        if not is_valid_normal_crop(stats, args):
            fallback_candidates.append((score_crop(stats), box, stats))
            continue

        prefer_hard = random.random() < args.hard_negative_prob
        if prefer_hard and not is_hard_negative_crop(stats, args):
            fallback_candidates.append((score_crop(stats), box, stats))
            continue

        accepted.append({
            "box_xyxy": box,
            "stats": stats,
            "attempt": attempts,
            "sampling_mode": "accepted",
        })

    if len(accepted) < args.num_crops_per_image:
        fallback_candidates = sorted(fallback_candidates, key=lambda x: x[0], reverse=True)
        used_boxes = set(tuple(item["box_xyxy"]) for item in accepted)

        for _, box, stats in fallback_candidates:
            if len(accepted) >= args.num_crops_per_image:
                break
            if tuple(box) in used_boxes:
                continue
            if stats["structure_ratio"] < max(0.01, args.min_structure_ratio * 0.3):
                continue

            accepted.append({
                "box_xyxy": box,
                "stats": stats,
                "attempt": attempts,
                "sampling_mode": "fallback_best_score",
            })
            used_boxes.add(tuple(box))

    return [{
        "source_image": image_path,
        "source_stem": stem,
        "image_bgr": image_bgr,
        **item,
    } for item in accepted]


def save_normal_crop(item: Dict, sample_index: int, args) -> Dict:
    stem = item["source_stem"]
    image_bgr = item["image_bgr"]
    box = item["box_xyxy"]
    stats = item["stats"]

    crop = crop_array(image_bgr, box)
    base_name = f"{stem}_normal_{sample_index:04d}_crop"

    crop_path = os.path.join(args.output_dir, f"{base_name}.png")
    label_path = os.path.join(args.output_dir, f"{base_name}.txt")
    debug_path = os.path.join(args.output_dir, f"{base_name}_debug.png")
    metadata_path = os.path.join(args.output_dir, f"{base_name}_metadata.json")

    if not args.overwrite and os.path.exists(crop_path):
        raise FileExistsError(f"Output exists: {crop_path}. Use --overwrite to replace existing files.")

    save_image(crop_path, crop)
    write_empty_label(label_path)

    if not args.no_debug:
        debug = make_debug_crop(crop, stats)
        save_image(debug_path, debug)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "defect_type": "normal_background",
        "class_id": None,
        "source_image": item["source_image"],
        "source_stem": stem,
        "sample_index": int(sample_index),
        "crop_box_xyxy": [int(v) for v in box],
        "crop_size": int(args.crop_size),
        "label_type": "empty_yolo_label",
        "stats": {k: float(v) for k, v in stats.items()},
        "sampling_mode": item["sampling_mode"],
        "attempt": int(item["attempt"]),
        "outputs": {
            "crop_image_path": crop_path,
            "crop_label_path": label_path,
            "crop_debug_path": None if args.no_debug else debug_path,
            "metadata_path": metadata_path,
        },
        "generation_parameters": {
            "num_crops_per_image": int(args.num_crops_per_image),
            "crop_size": int(args.crop_size),
            "min_trace_ratio": float(args.min_trace_ratio),
            "max_trace_ratio": float(args.max_trace_ratio),
            "min_structure_ratio": float(args.min_structure_ratio),
            "max_structure_ratio": float(args.max_structure_ratio),
            "min_dark_feature_ratio": float(args.min_dark_feature_ratio),
            "hard_negative_prob": float(args.hard_negative_prob),
            "border_margin": int(args.border_margin),
            "max_attempts_per_image": int(args.max_attempts_per_image),
            "seed": int(args.seed),
        }
    }

    write_json(metadata_path, metadata)
    return metadata


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.output_dir)

    image_paths = resolve_image_paths(args.dataset_dir, args.images)
    valid_image_paths = []

    for path in image_paths:
        if not os.path.exists(path):
            print(f"[WARNING] Image not found, skipped: {path}")
            continue
        valid_image_paths.append(path)

    if not valid_image_paths:
        raise RuntimeError("No valid source images found.")

    print("[INFO] Generate normal/background crops")
    print(f"[INFO] Valid source images: {len(valid_image_paths)}")
    print(f"[INFO] Crops per image: {args.num_crops_per_image}")
    print(f"[INFO] Crop size: {args.crop_size}")
    print(f"[INFO] Output dir: {args.output_dir}")

    all_metadata = []
    per_image_counts = {}

    for image_path in valid_image_paths:
        stem = get_file_stem(image_path)
        print(f"\n[INFO] Sampling normal crops from: {image_path}")

        try:
            sampled_items = sample_normal_crops_for_image(image_path, args)
        except Exception as exc:
            print(f"[ERROR] Failed to sample {stem}: {exc}")
            per_image_counts[stem] = 0
            continue

        print(f"[INFO] Accepted crops for {stem}: {len(sampled_items)} / {args.num_crops_per_image}")

        count = 0
        for idx, item in enumerate(sampled_items, start=1):
            try:
                metadata = save_normal_crop(item, sample_index=idx, args=args)
                all_metadata.append(metadata)
                count += 1
            except Exception as exc:
                print(f"[ERROR] Failed to save normal crop {stem} #{idx}: {exc}")

        per_image_counts[stem] = count

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": args.output_dir,
        "valid_source_images": valid_image_paths,
        "num_crops_per_image": int(args.num_crops_per_image),
        "total_crops": int(len(all_metadata)),
        "per_image_counts": per_image_counts,
        "note": (
            "These are background/negative YOLO samples. Their .txt label files "
            "are intentionally empty and should be used with --allow_empty_labels "
            "when building the YOLO dataset."
        ),
    }

    summary_path = os.path.join(args.output_dir, "normal_crops_summary.json")
    write_json(summary_path, summary)

    print("\n[SUCCESS] Normal crop generation finished.")
    print(f"[SUCCESS] Total crops: {len(all_metadata)}")
    print(f"[SUCCESS] Summary: {summary_path}")


if __name__ == "__main__":
    main()
