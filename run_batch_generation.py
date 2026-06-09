import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple


DEFAULT_IMAGE_NAMES = [
    "01.JPG", "04.JPG", "05.JPG", "06.JPG", "07.JPG",
    "08.JPG", "09.JPG", "10.JPG", "11.JPG", "12.JPG",
]

DEFECT_ORDER = [
    "short_circuit",
    "open_circuit",
    "spur",
    "missing_hole",
    "mouse_bite",
    "spurious_copper",
]

DEFECT_SCRIPTS = {
    "short_circuit": "generate_short_circuit.py",
    "open_circuit": "generate_open_circuit.py",
    "spur": "generate_spur.py",
    "missing_hole": "generate_missing_hole.py",
    "mouse_bite": "generate_mouse_bite.py",
    "spurious_copper": "generate_spurious_copper.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-run topology extraction and six PCB defect generators."
    )

    parser.add_argument("--dataset_dir", type=str, default="PCB_Dataset")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGE_NAMES)
    parser.add_argument("--config", type=str, default=os.path.join("configs", "hsv_trace_config.json"))
    parser.add_argument("--topology_dir", type=str, default=os.path.join("outputs", "topology"))
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument("--run_topology", action="store_true")
    parser.add_argument("--force_topology", action="store_true")
    parser.add_argument("--defects", nargs="+", default=DEFECT_ORDER, choices=DEFECT_ORDER)

    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_seed_per_run", action="store_true")
    parser.add_argument("--crop_size", type=int, default=128)

    # Mouse bite recommended parameters
    parser.add_argument("--mouse_bite_min_radius", type=int, default=6)
    parser.add_argument("--mouse_bite_max_radius", type=int, default=14)
    parser.add_argument("--mouse_bite_max_attempts", type=int, default=3000)

    # Open circuit recommended parameters
    parser.add_argument("--open_circuit_pad_dilate", type=int, default=28)
    parser.add_argument("--open_circuit_min_gap_length", type=int, default=10)
    parser.add_argument("--open_circuit_max_gap_length", type=int, default=22)
    parser.add_argument("--open_circuit_min_elongation", type=float, default=4.0)
    parser.add_argument("--open_circuit_max_attempts", type=int, default=15000)

    # Short circuit recommended parameters
    parser.add_argument("--short_circuit_max_attempts", type=int, default=20000)
    parser.add_argument("--short_circuit_pad_dilate", type=int, default=24)
    parser.add_argument("--short_circuit_min_gap_distance", type=int, default=4)
    parser.add_argument("--short_circuit_max_gap_distance", type=int, default=34)
    parser.add_argument("--short_circuit_min_parallel_cos", type=float, default=0.85)

    # Spur recommended parameters
    parser.add_argument("--spur_max_attempts", type=int, default=15000)
    parser.add_argument("--spur_min_length", type=int, default=8)
    parser.add_argument("--spur_max_length", type=int, default=22)
    parser.add_argument("--spur_pad_dilate", type=int, default=22)

    # Spurious copper recommended parameters
    parser.add_argument("--spurious_copper_min_major_length", type=int, default=16)
    parser.add_argument("--spurious_copper_max_major_length", type=int, default=34)
    parser.add_argument("--spurious_copper_min_minor_length", type=int, default=8)
    parser.add_argument("--spurious_copper_max_minor_length", type=int, default=16)
    parser.add_argument("--spurious_copper_min_detach_gap", type=int, default=5)
    parser.add_argument("--spurious_copper_max_detach_gap", type=int, default=14)
    parser.add_argument("--spurious_copper_min_defect_area", type=int, default=50)
    parser.add_argument("--spurious_copper_min_bbox_width", type=int, default=8)
    parser.add_argument("--spurious_copper_min_bbox_height", type=int, default=8)
    parser.add_argument("--spurious_copper_max_attempts", type=int, default=20000)
    parser.add_argument("--spurious_copper_shape_mode", type=str, default="mixed",
                        choices=["mixed", "rectangle", "trapezoid", "ellipse"])
    parser.add_argument("--spurious_copper_pad_dilate", type=int, default=18)

    # Missing hole recommended parameters
    parser.add_argument("--missing_hole_max_attempts", type=int, default=12000)
    parser.add_argument("--missing_hole_min_radius", type=int, default=8)
    parser.add_argument("--missing_hole_max_radius", type=int, default=18)
    parser.add_argument("--missing_hole_min_radius_ratio", type=float, default=0.5)
    parser.add_argument("--missing_hole_max_radius_ratio", type=float, default=0.7)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--log_dir", type=str, default=os.path.join("outputs", "batch_logs"))

    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def resolve_image_paths(dataset_dir: str, images: List[str]) -> List[str]:
    resolved = []
    for item in images:
        resolved.append(item if os.path.exists(item) else os.path.join(dataset_dir, item))
    return resolved


def topology_files_exist(topology_dir: str, image_path: str) -> bool:
    stem = get_file_stem(image_path)
    trace_mask = os.path.join(topology_dir, f"{stem}_trace_mask.png")
    attack_mask = os.path.join(topology_dir, f"{stem}_attack_candidate_mask.png")
    return os.path.exists(trace_mask) and os.path.exists(attack_mask)


def quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def defect_output_dir(args, defect: str) -> str:
    return os.path.join(args.output_root, defect)


def make_seed(args, image_index: int, defect_index: int) -> int:
    if args.fixed_seed_per_run:
        return int(args.seed)
    return int(args.seed + image_index * 1000 + defect_index * 100)


def build_topology_command(args, image_path: str) -> List[str]:
    return [
        args.python,
        "extract_topology.py",
        "--image", image_path,
        "--config", args.config,
        "--output_dir", args.topology_dir,
    ]


def build_defect_command(args, defect: str, image_path: str, image_index: int, defect_index: int) -> List[str]:
    seed = make_seed(args, image_index=image_index, defect_index=defect_index)
    cmd = [
        args.python,
        DEFECT_SCRIPTS[defect],
        "--image", image_path,
        "--topology_dir", args.topology_dir,
        "--output_dir", defect_output_dir(args, defect),
        "--seed", str(seed),
        "--num_samples", str(args.num_samples),
        "--crop_size", str(args.crop_size),
    ]

    if defect == "mouse_bite":
        cmd += [
            "--min_radius", str(args.mouse_bite_min_radius),
            "--max_radius", str(args.mouse_bite_max_radius),
            "--max_attempts", str(args.mouse_bite_max_attempts),
        ]

    elif defect == "open_circuit":
        cmd += [
            "--pad_dilate", str(args.open_circuit_pad_dilate),
            "--min_gap_length", str(args.open_circuit_min_gap_length),
            "--max_gap_length", str(args.open_circuit_max_gap_length),
            "--min_elongation", str(args.open_circuit_min_elongation),
            "--max_attempts", str(args.open_circuit_max_attempts),
        ]

    elif defect == "short_circuit":
        cmd += [
            "--max_attempts", str(args.short_circuit_max_attempts),
            "--pad_dilate", str(args.short_circuit_pad_dilate),
            "--min_gap_distance", str(args.short_circuit_min_gap_distance),
            "--max_gap_distance", str(args.short_circuit_max_gap_distance),
            "--min_parallel_cos", str(args.short_circuit_min_parallel_cos),
        ]

    elif defect == "spur":
        cmd += [
            "--max_attempts", str(args.spur_max_attempts),
            "--min_spur_length", str(args.spur_min_length),
            "--max_spur_length", str(args.spur_max_length),
            "--pad_dilate", str(args.spur_pad_dilate),
        ]

    elif defect == "spurious_copper":
        cmd += [
            "--shape_mode", str(args.spurious_copper_shape_mode),
            "--pad_dilate", str(args.spurious_copper_pad_dilate),
            "--min_major_length", str(args.spurious_copper_min_major_length),
            "--max_major_length", str(args.spurious_copper_max_major_length),
            "--min_minor_length", str(args.spurious_copper_min_minor_length),
            "--max_minor_length", str(args.spurious_copper_max_minor_length),
            "--min_detach_gap", str(args.spurious_copper_min_detach_gap),
            "--max_detach_gap", str(args.spurious_copper_max_detach_gap),
            "--min_defect_area", str(args.spurious_copper_min_defect_area),
            "--min_bbox_width", str(args.spurious_copper_min_bbox_width),
            "--min_bbox_height", str(args.spurious_copper_min_bbox_height),
        ]

    elif defect == "missing_hole":
        cmd += [
            "--max_attempts", str(args.missing_hole_max_attempts),
            "--min_hole_radius", str(args.missing_hole_min_radius),
            "--max_hole_radius", str(args.missing_hole_max_radius),
            "--min_hole_radius_ratio", str(args.missing_hole_min_radius_ratio),
            "--max_hole_radius_ratio", str(args.missing_hole_max_radius_ratio),
        ]

    return cmd


def run_command(cmd: List[str], dry_run: bool = False) -> Tuple[int, str]:
    print(f"[CMD] {quote_cmd(cmd)}")
    if dry_run:
        return 0, ""

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = process.stdout or ""
    if output:
        print(output)
    return int(process.returncode), output


def count_generated_crop_images(output_dir: str, stem: str, defect: str) -> int:
    if not os.path.isdir(output_dir):
        return 0

    prefix = f"{stem}_{defect}"
    count = 0
    for name in os.listdir(output_dir):
        lower = name.lower()
        if not lower.endswith("_crop.png"):
            continue
        if lower.endswith("_crop_mask.png") or lower.endswith("_crop_debug.png"):
            continue
        if name.startswith(prefix):
            count += 1
    return count


def summarize_outputs(args, image_paths: List[str]) -> Dict[str, Dict[str, int]]:
    summary = {}
    for defect in args.defects:
        out_dir = defect_output_dir(args, defect)
        summary[defect] = {}
        for image_path in image_paths:
            stem = get_file_stem(image_path)
            summary[defect][stem] = count_generated_crop_images(out_dir, stem, defect)
    return summary


def main():
    args = parse_args()

    ensure_dir(args.log_dir)
    ensure_dir(args.topology_dir)
    ensure_dir(args.output_root)
    for defect in args.defects:
        ensure_dir(defect_output_dir(args, defect))

    image_paths = resolve_image_paths(args.dataset_dir, args.images)
    valid_image_paths = []

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found, skipped: {image_path}")
            continue
        valid_image_paths.append(image_path)

    if not valid_image_paths:
        raise RuntimeError("No valid input images found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"batch_generation_{timestamp}.log")
    summary_path = os.path.join(args.log_dir, f"batch_generation_{timestamp}_summary.json")

    records = []
    log_chunks = []

    print("[INFO] Batch PCB defect generation")
    print(f"[INFO] Valid images: {len(valid_image_paths)}")
    print(f"[INFO] Defects: {', '.join(args.defects)}")
    print(f"[INFO] Samples per image per defect: {args.num_samples}")
    print(f"[INFO] Output root: {args.output_root}")
    print(f"[INFO] Run topology: {args.run_topology}")
    print(f"[INFO] Dry run: {args.dry_run}")

    # -------------------------------------------------------------------------
    # Stage A: topology extraction
    # -------------------------------------------------------------------------
    if args.run_topology:
        print("\n[INFO] Stage A: topology extraction")
        for image_path in valid_image_paths:
            stem = get_file_stem(image_path)

            if topology_files_exist(args.topology_dir, image_path) and not args.force_topology:
                print(f"[INFO] Topology exists, skipped: {stem}")
                records.append({
                    "stage": "topology",
                    "image": image_path,
                    "stem": stem,
                    "status": "skipped_existing",
                })
                continue

            cmd = build_topology_command(args, image_path)
            returncode, output = run_command(cmd, dry_run=args.dry_run)
            log_chunks.append(f"\n\n===== TOPOLOGY {stem} =====\n{quote_cmd(cmd)}\nRETURN={returncode}\n{output}")

            records.append({
                "stage": "topology",
                "image": image_path,
                "stem": stem,
                "status": "success" if returncode == 0 else "failed",
                "returncode": returncode,
                "command": cmd,
            })

            if returncode != 0 and args.stop_on_error:
                print("[ERROR] Topology extraction failed. Stopping because --stop_on_error is set.")
                break

    # -------------------------------------------------------------------------
    # Stage B: defect generation
    # -------------------------------------------------------------------------
    print("\n[INFO] Stage B: defect generation")
    stop_requested = False

    for defect_index, defect in enumerate(args.defects):
        print(f"\n[INFO] Defect type: {defect}")

        for image_index, image_path in enumerate(valid_image_paths):
            stem = get_file_stem(image_path)

            if not topology_files_exist(args.topology_dir, image_path):
                print(f"[WARNING] Missing topology for {stem}; skipped {defect}.")
                records.append({
                    "stage": "defect",
                    "defect": defect,
                    "image": image_path,
                    "stem": stem,
                    "status": "skipped_missing_topology",
                })
                continue

            cmd = build_defect_command(
                args=args,
                defect=defect,
                image_path=image_path,
                image_index=image_index,
                defect_index=defect_index,
            )

            returncode, output = run_command(cmd, dry_run=args.dry_run)
            log_chunks.append(
                f"\n\n===== DEFECT {defect} {stem} =====\n"
                f"{quote_cmd(cmd)}\nRETURN={returncode}\n{output}"
            )

            records.append({
                "stage": "defect",
                "defect": defect,
                "image": image_path,
                "stem": stem,
                "status": "success" if returncode == 0 else "failed",
                "returncode": returncode,
                "command": cmd,
            })

            if returncode != 0 and args.stop_on_error:
                stop_requested = True
                print("[ERROR] Defect generation failed. Stopping because --stop_on_error is set.")
                break

        if stop_requested:
            break

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    output_counts = summarize_outputs(args, valid_image_paths)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": args.dataset_dir,
        "images": valid_image_paths,
        "defects": args.defects,
        "num_samples_per_image_per_defect": int(args.num_samples),
        "crop_size": int(args.crop_size),
        "output_root": args.output_root,
        "topology_dir": args.topology_dir,
        "dry_run": bool(args.dry_run),
        "records": records,
        "output_crop_counts": output_counts,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_chunks))

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n[INFO] Batch generation finished.")
    print(f"[INFO] Log saved to:     {log_path}")
    print(f"[INFO] Summary saved to: {summary_path}")

    print("\n[INFO] Generated crop count summary:")
    for defect, per_image in output_counts.items():
        total = sum(per_image.values())
        print(f"  {defect}: {total}")
        for stem, count in per_image.items():
            print(f"    {stem}: {count}")


if __name__ == "__main__":
    main()
