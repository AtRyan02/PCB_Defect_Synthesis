import argparse
import copy
import io
import json
import os
import shutil
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import requests


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read gray image: {path}")
    return img


def save_image(path: str, img: np.ndarray):
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


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


def write_label(path: str, yolo_line: Optional[str]):
    if yolo_line is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(yolo_line + "\n")


def create_debug_overlay(image_bgr: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
    debug = image_bgr.copy()
    color_layer = np.zeros_like(debug)
    color_layer[:, :, 2] = 255
    mask_bool = defect_mask > 0
    debug[mask_bool] = cv2.addWeighted(debug[mask_bool], 0.45, color_layer[mask_bool], 0.55, 0)
    return debug


def paste_crop_back(full_image: np.ndarray, crop_image: np.ndarray, crop_box_xyxy: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in crop_box_xyxy]
    out = full_image.copy()
    out[y1:y2, x1:x2] = crop_image
    return out


def make_soft_mask_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Reflect the currently tested ComfyUI workflow:
    use the crop image itself as the mask source and choose the red channel.
    Since OpenCV loads BGR, red channel = crop[:, :, 2].
    The result is a grayscale soft mask image.
    """
    red = crop_bgr[:, :, 2]
    return red


# -----------------------------------------------------------------------------
# ComfyUI API
# -----------------------------------------------------------------------------

class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def upload_image(self, image_path: str, subfolder: str = "", overwrite: bool = True) -> Dict[str, Any]:
        url = f"{self.base_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/png")}
            data = {
                "type": "input",
                "overwrite": "true" if overwrite else "false",
                "subfolder": subfolder,
            }
            resp = requests.post(url, files=files, data=data, timeout=120)
            resp.raise_for_status()
            return resp.json()

    def queue_prompt(self, prompt_workflow: Dict[str, Any]) -> str:
        url = f"{self.base_url}/prompt"
        payload = {
            "prompt": prompt_workflow,
            "client_id": self.client_id,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/history/{prompt_id}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def wait_until_done(self, prompt_id: str, poll_sec: float = 1.0, timeout_sec: float = 600.0) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            hist = self.get_history(prompt_id)
            if prompt_id in hist:
                return hist[prompt_id]
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")
            time.sleep(poll_sec)

    def download_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        url = f"{self.base_url}/view"
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        return resp.content


# -----------------------------------------------------------------------------
# Workflow patching
# -----------------------------------------------------------------------------

def load_node_id_map(path: str) -> Dict[str, str]:
    data = read_json(path)
    return {k: str(v) for k, v in data.items()}


def set_text_input(workflow: Dict[str, Any], node_id: str, text: str):
    workflow[node_id]["inputs"]["text"] = text


def set_load_image_input(workflow: Dict[str, Any], node_id: str, image_name: str):
    workflow[node_id]["inputs"]["image"] = image_name


def set_ksampler_inputs(
    workflow: Dict[str, Any],
    node_id: str,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    seed: Optional[int] = None,
):
    workflow[node_id]["inputs"]["steps"] = int(steps)
    workflow[node_id]["inputs"]["cfg"] = float(cfg)
    workflow[node_id]["inputs"]["sampler_name"] = str(sampler_name)
    workflow[node_id]["inputs"]["scheduler"] = str(scheduler)
    workflow[node_id]["inputs"]["denoise"] = float(denoise)
    if seed is not None:
        workflow[node_id]["inputs"]["seed"] = int(seed)


def set_save_prefix(workflow: Dict[str, Any], node_id: str, prefix: str):
    workflow[node_id]["inputs"]["filename_prefix"] = prefix


def build_workflow_for_sample(
    base_workflow: Dict[str, Any],
    node_ids: Dict[str, str],
    crop_upload_name: str,
    mask_upload_name: str,
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    save_prefix: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    wf = copy.deepcopy(base_workflow)

    set_load_image_input(wf, node_ids["load_image"], crop_upload_name)
    set_load_image_input(wf, node_ids["load_mask"], mask_upload_name)
    set_text_input(wf, node_ids["positive"], positive_prompt)
    set_text_input(wf, node_ids["negative"], negative_prompt)
    set_ksampler_inputs(
        wf,
        node_ids["ksampler"],
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        seed=seed,
    )
    set_save_prefix(wf, node_ids["save_image"], save_prefix)
    return wf


def extract_first_saved_image_info(history_entry: Dict[str, Any]) -> Dict[str, Any]:
    outputs = history_entry.get("outputs", {})
    for _, node_out in outputs.items():
        images = node_out.get("images", [])
        if images:
            return images[0]
    raise RuntimeError("No output image found in ComfyUI history response.")


# -----------------------------------------------------------------------------
# Sample discovery
# -----------------------------------------------------------------------------

def list_crop_samples(input_dir: str) -> List[Dict[str, Any]]:
    entries = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith("_crop.png"):
            continue
        if name.lower().endswith("_crop_debug.png") or name.lower().endswith("_crop_mask.png"):
            continue

        crop_path = os.path.join(input_dir, name)
        stem = name[:-len("_crop.png")]

        sample = {
            "stem": stem,
            "crop_image": crop_path,
            "crop_mask": os.path.join(input_dir, f"{stem}_crop_mask.png"),
            "crop_label": os.path.join(input_dir, f"{stem}_crop.txt"),
            "full_image": os.path.join(input_dir, f"{stem}.png"),
            "full_mask": os.path.join(input_dir, f"{stem}_mask.png"),
            "full_label": os.path.join(input_dir, f"{stem}.txt"),
            "metadata": os.path.join(input_dir, f"{stem}_metadata.json"),
        }

        if not os.path.exists(sample["crop_mask"]):
            print(f"[WARN] Skip {name}: missing crop mask")
            continue
        if not os.path.exists(sample["full_image"]):
            print(f"[WARN] Skip {name}: missing full image")
            continue
        if not os.path.exists(sample["full_mask"]):
            print(f"[WARN] Skip {name}: missing full mask")
            continue
        if not os.path.exists(sample["metadata"]):
            print(f"[WARN] Skip {name}: missing metadata")
            continue

        entries.append(sample)
    return entries


# -----------------------------------------------------------------------------
# Main batch process
# -----------------------------------------------------------------------------

def harmonize_one_sample(
    sample: Dict[str, Any],
    args,
    client: ComfyUIClient,
    base_workflow: Dict[str, Any],
    node_ids: Dict[str, str],
):
    stem = sample["stem"]
    print(f"\n[INFO] Harmonizing sample: {stem}")

    crop_img = read_image(sample["crop_image"])
    crop_mask = read_gray(sample["crop_mask"])
    full_img = read_image(sample["full_image"])
    full_mask = read_gray(sample["full_mask"])
    metadata = read_json(sample["metadata"])

    # ------------------------------------------------------------------
    # Prepare temporary uploads for ComfyUI
    # ------------------------------------------------------------------
    tmp_dir = os.path.join(args.output_dir, "_tmp_inputs")
    ensure_dir(tmp_dir)

    tmp_crop_path = os.path.join(tmp_dir, f"{stem}_crop_for_comfy.png")
    save_image(tmp_crop_path, crop_img)

    if args.mask_mode == "soft_from_crop":
        mask_img = make_soft_mask_from_crop(crop_img)
        tmp_mask_path = os.path.join(tmp_dir, f"{stem}_softmask.png")
        save_image(tmp_mask_path, mask_img)
    elif args.mask_mode == "real_crop_mask":
        tmp_mask_path = os.path.join(tmp_dir, f"{stem}_mask_for_comfy.png")
        save_image(tmp_mask_path, crop_mask)
    else:
        raise ValueError(f"Unsupported mask_mode: {args.mask_mode}")

    up_crop = client.upload_image(tmp_crop_path)
    up_mask = client.upload_image(tmp_mask_path)

    crop_upload_name = up_crop["name"]
    mask_upload_name = up_mask["name"]

    wf = build_workflow_for_sample(
        base_workflow=base_workflow,
        node_ids=node_ids,
        crop_upload_name=crop_upload_name,
        mask_upload_name=mask_upload_name,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler_name,
        scheduler=args.scheduler,
        denoise=args.denoise,
        save_prefix=f"sd_harmonize/{stem}",
        seed=None if args.randomize_seed else args.seed,
    )

    prompt_id = client.queue_prompt(wf)
    hist_entry = client.wait_until_done(prompt_id, poll_sec=args.poll_sec, timeout_sec=args.timeout_sec)
    out_info = extract_first_saved_image_info(hist_entry)
    raw_bytes = client.download_image(
        filename=out_info["filename"],
        subfolder=out_info.get("subfolder", ""),
        folder_type=out_info.get("type", "output"),
    )

    decoded = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode downloaded ComfyUI output image")
    sd_crop = decoded

    # ------------------------------------------------------------------
    # Save crop-level outputs
    # ------------------------------------------------------------------
    crop_sd_path = os.path.join(args.output_dir, f"{stem}_crop_sd.png")
    save_image(crop_sd_path, sd_crop)

    crop_sd_mask_path = os.path.join(args.output_dir, f"{stem}_crop_sd_mask.png")
    save_image(crop_sd_mask_path, crop_mask)

    crop_sd_label_path = os.path.join(args.output_dir, f"{stem}_crop_sd.txt")
    class_id = None
    if os.path.exists(sample["crop_label"]):
        shutil.copy2(sample["crop_label"], crop_sd_label_path)
        with open(sample["crop_label"], "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                try:
                    class_id = int(line.split()[0])
                except Exception:
                    class_id = None
    else:
        crop_yolo = None if class_id is None else mask_to_yolo_bbox(crop_mask, class_id)
        write_label(crop_sd_label_path, crop_yolo)

    crop_sd_debug = create_debug_overlay(sd_crop, crop_mask)
    crop_sd_debug_path = os.path.join(args.output_dir, f"{stem}_crop_sd_debug.png")
    save_image(crop_sd_debug_path, crop_sd_debug)

    # ------------------------------------------------------------------
    # Paste crop back into full image
    # ------------------------------------------------------------------
    crop_info = metadata.get("outputs", {}).get("crop")
    if crop_info is None:
        raise KeyError(f"No crop info found in metadata for {stem}")
    crop_box = crop_info["crop_box_xyxy"]
    full_sd = paste_crop_back(full_img, sd_crop, crop_box)

    full_sd_path = os.path.join(args.output_dir, f"{stem}_sd.png")
    save_image(full_sd_path, full_sd)

    full_sd_mask_path = os.path.join(args.output_dir, f"{stem}_sd_mask.png")
    save_image(full_sd_mask_path, full_mask)

    full_sd_label_path = os.path.join(args.output_dir, f"{stem}_sd.txt")
    if os.path.exists(sample["full_label"]):
        shutil.copy2(sample["full_label"], full_sd_label_path)
    else:
        yolo_line = None if class_id is None else mask_to_yolo_bbox(full_mask, class_id)
        write_label(full_sd_label_path, yolo_line)

    full_sd_debug = create_debug_overlay(full_sd, full_mask)
    full_sd_debug_path = os.path.join(args.output_dir, f"{stem}_sd_debug.png")
    save_image(full_sd_debug_path, full_sd_debug)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    md_out = copy.deepcopy(metadata)
    md_out["sd_harmonization"] = {
        "enabled": True,
        "comfy_url": args.comfy_url,
        "mask_mode": args.mask_mode,
        "model_note": args.model_note,
        "positive_prompt": args.positive_prompt,
        "negative_prompt": args.negative_prompt,
        "steps": int(args.steps),
        "cfg": float(args.cfg),
        "sampler_name": str(args.sampler_name),
        "scheduler": str(args.scheduler),
        "denoise": float(args.denoise),
        "seed_mode": "randomize" if args.randomize_seed else f"fixed:{args.seed}",
        "workflow_api_json": args.workflow_api_json,
        "node_ids_json": args.node_ids_json,
    }
    md_out.setdefault("outputs", {})
    md_out["outputs"]["sd_full_image_path"] = full_sd_path
    md_out["outputs"]["sd_full_mask_path"] = full_sd_mask_path
    md_out["outputs"]["sd_full_label_path"] = full_sd_label_path
    md_out["outputs"]["sd_full_debug_path"] = full_sd_debug_path
    md_out["outputs"]["sd_crop"] = {
        "crop_image_path": crop_sd_path,
        "crop_mask_path": crop_sd_mask_path,
        "crop_label_path": crop_sd_label_path,
        "crop_debug_path": crop_sd_debug_path,
        "crop_box_xyxy": crop_box,
    }

    md_out_path = os.path.join(args.output_dir, f"{stem}_sd_metadata.json")
    write_json(md_out_path, md_out)

    print(f"[OK] Saved crop SD image: {crop_sd_path}")
    print(f"[OK] Saved full SD image: {full_sd_path}")
    print(f"[OK] Saved metadata:      {md_out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch SD harmonization for short_circuit samples via ComfyUI")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing generated short_circuit outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save SD harmonized outputs")
    parser.add_argument("--workflow_api_json", type=str, required=True, help="Exported ComfyUI workflow_api.json")
    parser.add_argument("--node_ids_json", type=str, required=True, help="JSON mapping for node ids in workflow_api.json")
    parser.add_argument("--comfy_url", type=str, default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--mask_mode", type=str, default="soft_from_crop", choices=["soft_from_crop", "real_crop_mask"], help="How to feed the second Load Image (as mask) node")
    parser.add_argument("--positive_prompt", type=str, default="realistic PCB inspection image, matte dark green PCB traces, same local lighting and texture")
    parser.add_argument("--negative_prompt", type=str, default="white blob, silver material, solder, metallic object, changed geometry, extra traces, removed traces, blur")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sampler_name", type=str, default="euler")
    parser.add_argument("--scheduler", type=str, default="simple")
    parser.add_argument("--denoise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize_seed", action="store_true", help="Use ComfyUI randomize mode instead of fixed seed")
    parser.add_argument("--poll_sec", type=float, default=1.0)
    parser.add_argument("--timeout_sec", type=float, default=600.0)
    parser.add_argument("--model_note", type=str, default="Mirror current manually tested ComfyUI workflow")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    samples = list_crop_samples(args.input_dir)
    if not samples:
        print("[WARN] No valid *_crop.png samples found.")
        return

    base_workflow = read_json(args.workflow_api_json)
    node_ids = load_node_id_map(args.node_ids_json)
    client = ComfyUIClient(args.comfy_url)

    print(f"[INFO] Found {len(samples)} crop samples.")
    print(f"[INFO] ComfyUI: {args.comfy_url}")
    print(f"[INFO] mask_mode: {args.mask_mode}")
    print(f"[INFO] denoise={args.denoise}, cfg={args.cfg}, steps={args.steps}")

    success = 0
    for sample in samples:
        try:
            harmonize_one_sample(sample, args, client, base_workflow, node_ids)
            success += 1
        except Exception as e:
            print(f"[ERROR] Failed on {sample['stem']}: {e}")

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "workflow_api_json": args.workflow_api_json,
        "node_ids_json": args.node_ids_json,
        "comfy_url": args.comfy_url,
        "mask_mode": args.mask_mode,
        "positive_prompt": args.positive_prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler_name": args.sampler_name,
        "scheduler": args.scheduler,
        "denoise": args.denoise,
        "processed": success,
        "requested": len(samples),
    }
    write_json(os.path.join(args.output_dir, "sd_harmonize_summary.json"), summary)
    print(f"\n[SUCCESS] Done: {success} / {len(samples)}")


if __name__ == "__main__":
    main()
