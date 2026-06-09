import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


CLASS_NAMES = {
    0: "short_circuit",
    1: "open_circuit",
    2: "spur",
    3: "missing_hole",
    4: "mouse_bite",
    5: "spurious_copper",
}

DEFAULT_SOURCE_DIRS = [
    os.path.join("outputs", "short_circuit"),
    os.path.join("outputs", "open_circuit"),
    os.path.join("outputs", "spur"),
    os.path.join("outputs", "missing_hole"),
    os.path.join("outputs", "mouse_bite"),
    os.path.join("outputs", "spurious_copper"),
]

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a YOLO-format PCB defect dataset from generated samples."
    )

    parser.add_argument(
        "--source_dirs",
        nargs="+",
        default=DEFAULT_SOURCE_DIRS,
        help="Source directories containing generated image/label pairs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_yolo_crop_rule",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--image_mode",
        type=str,
        default="crop",
        choices=["crop", "full", "crop_sd", "full_sd"],
        help=(
            "Which images to include: "
            "crop=*_crop.png, full=full rule image, "
            "crop_sd=*_crop_sd.png, full_sd=*_sd.png."
        ),
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="group",
        choices=["group", "random", "manual"],
        help=(
            "group: split by source image id such as 01/04/05; "
            "random: split individual samples randomly; "
            "manual: use --train_sources/--val_sources/--test_sources."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--train_sources", nargs="*", default=None)
    parser.add_argument("--val_sources", nargs="*", default=None)
    parser.add_argument("--test_sources", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--copy_mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="Copy files or create symlinks in the YOLO dataset.",
    )
    parser.add_argument(
        "--prefix_source_dir",
        action="store_true",
        help="Prefix output filenames with source directory name to avoid collisions.",
    )
    parser.add_argument(
        "--allow_empty_test",
        action="store_true",
        help="Allow empty test split when there are too few source groups.",
    )
    parser.add_argument(
        "--allow_empty_labels",
        action="store_true",
        help=(
            "Allow empty YOLO label files. This is required for normal/background "
            "negative crops."
        )
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Filename and label helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def get_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def extract_source_id_from_stem(stem: str) -> str:
    """
    Extract source image id from generated sample stem.

    Examples:
        04_short_circuit_0001_crop    -> 04
        12_spurious_copper_0003_crop  -> 12
        05_missing_hole_crop          -> 05
    """
    return stem.split("_")[0]


def is_auxiliary_image(filename: str) -> bool:
    lower = filename.lower()
    return (
        lower.endswith("_mask.png")
        or lower.endswith("_debug.png")
        or lower.endswith("_crop_mask.png")
        or lower.endswith("_crop_debug.png")
        or lower.endswith("_sd_mask.png")
        or lower.endswith("_sd_debug.png")
        or lower.endswith("_crop_sd_mask.png")
        or lower.endswith("_crop_sd_debug.png")
    )


def image_matches_mode(filename: str, image_mode: str) -> bool:
    lower = filename.lower()
    if get_ext(filename) not in VALID_IMAGE_EXTS:
        return False
    if is_auxiliary_image(filename):
        return False

    if image_mode == "crop":
        return lower.endswith("_crop.png")

    if image_mode == "crop_sd":
        return lower.endswith("_crop_sd.png")

    if image_mode == "full_sd":
        return lower.endswith("_sd.png") and not lower.endswith("_crop_sd.png")

    if image_mode == "full":
        if not lower.endswith(".png"):
            return False
        if lower.endswith("_crop.png"):
            return False
        if lower.endswith("_crop_sd.png"):
            return False
        if lower.endswith("_sd.png"):
            return False
        return True

    raise ValueError(f"Unsupported image_mode: {image_mode}")


def infer_label_path(image_path: str) -> str:
    root, _ = os.path.splitext(image_path)
    return root + ".txt"


def parse_yolo_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None
    return class_id, x, y, w, h


def validate_label_file(label_path: str, allow_empty_labels: bool = False) -> Tuple[bool, List[str], List[int]]:
    messages = []
    class_ids = []

    if not os.path.exists(label_path):
        return False, [f"missing_label:{label_path}"], []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        if allow_empty_labels:
            return True, [], []
        return False, [f"empty_label:{label_path}"], []

    for idx, line in enumerate(lines):
        parsed = parse_yolo_label_line(line)
        if parsed is None:
            messages.append(f"invalid_format_line_{idx + 1}:{line}")
            continue

        class_id, x, y, w, h = parsed
        class_ids.append(class_id)

        if class_id not in CLASS_NAMES:
            messages.append(f"invalid_class_id:{class_id}")
        if not (0.0 <= x <= 1.0):
            messages.append(f"x_center_out_of_range:{x}")
        if not (0.0 <= y <= 1.0):
            messages.append(f"y_center_out_of_range:{y}")
        if not (0.0 < w <= 1.0):
            messages.append(f"width_out_of_range:{w}")
        if not (0.0 < h <= 1.0):
            messages.append(f"height_out_of_range:{h}")

    return len(messages) == 0, messages, class_ids


# -----------------------------------------------------------------------------
# Sample collection
# -----------------------------------------------------------------------------

def collect_samples(source_dirs: List[str], image_mode: str, strict: bool, allow_empty_labels: bool = False):
    samples = []
    skipped = []

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            skipped.append({"source_dir": source_dir, "reason": "source_dir_not_found"})
            continue

        source_dir_name = os.path.basename(os.path.normpath(source_dir))

        for filename in sorted(os.listdir(source_dir)):
            if not image_matches_mode(filename, image_mode):
                continue

            image_path = os.path.join(source_dir, filename)
            label_path = infer_label_path(image_path)
            stem = get_stem(image_path)
            source_id = extract_source_id_from_stem(stem)

            valid, messages, class_ids = validate_label_file(label_path, allow_empty_labels=allow_empty_labels)
            if not valid:
                item = {
                    "image_path": image_path,
                    "label_path": label_path,
                    "source_dir": source_dir,
                    "reason": "invalid_label",
                    "messages": messages,
                }
                if strict:
                    raise ValueError(json.dumps(item, ensure_ascii=False, indent=2))
                skipped.append(item)
                continue

            samples.append({
                "image_path": image_path,
                "label_path": label_path,
                "stem": stem,
                "source_id": source_id,
                "source_dir": source_dir,
                "source_dir_name": source_dir_name,
                "class_ids": class_ids,
            })

    return samples, skipped


# -----------------------------------------------------------------------------
# Split logic
# -----------------------------------------------------------------------------

def check_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}: "
            f"{train_ratio}, {val_ratio}, {test_ratio}"
        )


def split_list_by_ratio(items: List, train_ratio: float, val_ratio: float, test_ratio: float):
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def assign_splits_group(samples: List[Dict], args):
    check_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_ids = sorted(set(sample["source_id"] for sample in samples))
    rng = random.Random(args.seed)
    rng.shuffle(source_ids)

    train_ids, val_ids, test_ids = split_list_by_ratio(
        source_ids, args.train_ratio, args.val_ratio, args.test_ratio
    )

    if len(test_ids) == 0 and not args.allow_empty_test:
        if len(val_ids) > 1:
            test_ids.append(val_ids.pop())
        elif len(train_ids) > 1:
            test_ids.append(train_ids.pop())
        else:
            raise ValueError("Test split is empty. Add more source images or use --allow_empty_test.")

    source_to_split = {}
    for sid in train_ids:
        source_to_split[sid] = "train"
    for sid in val_ids:
        source_to_split[sid] = "val"
    for sid in test_ids:
        source_to_split[sid] = "test"

    split_samples = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_samples[source_to_split[sample["source_id"]]].append(sample)

    return source_to_split, split_samples


def assign_splits_random(samples: List[Dict], args):
    check_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    rng = random.Random(args.seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    train, val, test = split_list_by_ratio(shuffled, args.train_ratio, args.val_ratio, args.test_ratio)
    return {}, {"train": train, "val": val, "test": test}


def assign_splits_manual(samples: List[Dict], args):
    if args.train_sources is None or args.val_sources is None or args.test_sources is None:
        raise ValueError("Manual split requires --train_sources, --val_sources, and --test_sources.")

    source_to_split = {}
    for sid in args.train_sources:
        source_to_split[str(sid)] = "train"
    for sid in args.val_sources:
        source_to_split[str(sid)] = "val"
    for sid in args.test_sources:
        source_to_split[str(sid)] = "test"

    split_samples = {"train": [], "val": [], "test": []}
    unassigned = sorted(set(s["source_id"] for s in samples if s["source_id"] not in source_to_split))
    if unassigned:
        print(f"[WARNING] Source ids not assigned in manual split: {unassigned}")

    for sample in samples:
        split = source_to_split.get(sample["source_id"])
        if split is not None:
            split_samples[split].append(sample)

    return source_to_split, split_samples


def assign_splits(samples: List[Dict], args):
    if args.split_mode == "group":
        return assign_splits_group(samples, args)
    if args.split_mode == "random":
        return assign_splits_random(samples, args)
    if args.split_mode == "manual":
        return assign_splits_manual(samples, args)
    raise ValueError(f"Unsupported split_mode: {args.split_mode}")


# -----------------------------------------------------------------------------
# Output materialization
# -----------------------------------------------------------------------------

def prepare_output_dirs(output_dir: str, overwrite: bool, dry_run: bool):
    if os.path.exists(output_dir):
        if overwrite:
            if not dry_run:
                shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")

    if dry_run:
        return

    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(output_dir, "images", split))
        ensure_dir(os.path.join(output_dir, "labels", split))


def copy_or_link(src: str, dst: str, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        raise ValueError(f"Unsupported copy_mode: {mode}")


def make_output_stem(sample: Dict, prefix_source_dir: bool) -> str:
    if prefix_source_dir:
        return f"{sample['source_dir_name']}_{sample['stem']}"
    return sample["stem"]


def materialize_dataset(split_samples: Dict[str, List[Dict]], args):
    copied = defaultdict(int)
    class_counts = {split: defaultdict(int) for split in ["train", "val", "test"]}
    source_counts = {split: defaultdict(int) for split in ["train", "val", "test"]}

    for split, samples in split_samples.items():
        for sample in samples:
            copied[split] += 1
            source_counts[split][sample["source_id"]] += 1
            for cid in sample["class_ids"]:
                class_counts[split][cid] += 1

            if args.dry_run:
                continue

            out_stem = make_output_stem(sample, args.prefix_source_dir)
            image_ext = get_ext(sample["image_path"])
            image_dst = os.path.join(args.output_dir, "images", split, out_stem + image_ext)
            label_dst = os.path.join(args.output_dir, "labels", split, out_stem + ".txt")

            copy_or_link(sample["image_path"], image_dst, args.copy_mode)
            copy_or_link(sample["label_path"], label_dst, args.copy_mode)

    return {
        "copied": {k: int(v) for k, v in copied.items()},
        "class_counts": {
            split: {str(cid): int(count) for cid, count in counts.items()}
            for split, counts in class_counts.items()
        },
        "source_counts": {
            split: dict(counts)
            for split, counts in source_counts.items()
        },
    }


def write_data_yaml(output_dir: str) -> str:
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_path = os.path.abspath(output_dir).replace(os.sep, "/")
    lines = [
        f"path: {abs_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for cid in sorted(CLASS_NAMES):
        lines.append(f"  {cid}: {CLASS_NAMES[cid]}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return yaml_path


def write_summary(output_dir: str, summary: Dict) -> str:
    path = os.path.join(output_dir, "build_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    samples, skipped = collect_samples(args.source_dirs, args.image_mode, args.strict, allow_empty_labels=args.allow_empty_labels)
    if not samples:
        raise RuntimeError("No valid samples found. Check --source_dirs, --image_mode, and labels.")

    print("[INFO] Build YOLO dataset")
    print(f"[INFO] Image mode: {args.image_mode}")
    print(f"[INFO] Split mode: {args.split_mode}")
    print(f"[INFO] Valid samples: {len(samples)}")
    print(f"[INFO] Skipped items: {len(skipped)}")

    source_ids = sorted(set(sample["source_id"] for sample in samples))
    print(f"[INFO] Source ids: {source_ids}")

    source_to_split, split_samples = assign_splits(samples, args)

    print("[INFO] Split sample counts:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(split_samples[split])}")

    prepare_output_dirs(args.output_dir, args.overwrite, args.dry_run)
    copy_summary = materialize_dataset(split_samples, args)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dirs": args.source_dirs,
        "output_dir": args.output_dir,
        "image_mode": args.image_mode,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "dry_run": bool(args.dry_run),
        "allow_empty_labels": bool(args.allow_empty_labels),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "valid_sample_count": len(samples),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "source_ids": source_ids,
        "source_to_split": source_to_split,
        "split_counts": {split: len(split_samples[split]) for split in ["train", "val", "test"]},
        "copy_summary": copy_summary,
    }

    data_yaml_path = None
    summary_path = None
    if not args.dry_run:
        data_yaml_path = write_data_yaml(args.output_dir)
        summary_path = write_summary(args.output_dir, summary)

    print("\n[INFO] Class counts by split:")
    for split in ["train", "val", "test"]:
        print(f"  {split}:")
        counts = copy_summary["class_counts"].get(split, {})
        for cid in sorted(CLASS_NAMES):
            print(f"    {cid} {CLASS_NAMES[cid]}: {counts.get(str(cid), 0)}")

    if args.dry_run:
        print("\n[INFO] Dry run completed. No files copied.")
    else:
        print(f"\n[SUCCESS] YOLO dataset saved to: {args.output_dir}")
        print(f"[SUCCESS] data.yaml: {data_yaml_path}")
        print(f"[SUCCESS] summary:   {summary_path}")


if __name__ == "__main__":
    main()
