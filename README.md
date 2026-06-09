# PCB_Defect_Synthesis

A topology-guided synthetic data generation pipeline for PCB defect detection.

This project generates controllable PCB defect samples for downstream YOLO-based object detection. The current pipeline uses a **geometry-first** strategy: PCB topology is extracted first, defect geometry is injected by rules, and appearance is rendered using local PCB color statistics. Stable Diffusion / ComfyUI is treated only as an optional low-strength appearance harmonizer, not as the primary defect generator.

The generated samples include synthetic PCB images, binary defect masks, YOLO-format labels, defect-centered crops, debug visualizations, and metadata files.

---

## 1. Current Project Status

The project has progressed from single-image defect synthesis on `04.JPG` to a multi-image, six-class synthetic dataset pipeline.

Completed:

- HSV threshold calibration
- PCB trace topology extraction
- Rule-based generation for six PCB defect classes
- Multi-image batch generation across multiple PCB source images
- YOLO-format label export
- Defect-centered crop export
- Normal / background crop generation with empty YOLO labels
- YOLO dataset construction with grouped train / val / test split
- Crop-level YOLO26 training experiments
- Model selection comparison between YOLO26n and YOLO26s
- Spur v2 generation logic update to reduce severe false positives
- Spurious copper parameter refinement

Current best crop-level configuration:

```text
Dataset: crop rule-based synthetic dataset + normal/background crops + spur_v2
Model: YOLO26s
Image size: 320
Epochs: 30
```

Current best observed crop-level result:

```text
YOLO26s:
Precision     ≈ 0.944
Recall        ≈ 0.930
mAP50         ≈ 0.952
mAP50-95      ≈ 0.774
```

YOLO26s is selected as the default detector for subsequent horizontal comparison experiments.

---

## 2. Project Motivation

Real PCB defect samples are difficult to collect, expensive to annotate, and often highly imbalanced across defect categories. This project explores a controllable synthetic data factory for PCB defect generation, with the goal of improving YOLO-based detection under long-tail industrial inspection scenarios.

The current design follows a **geometry-first** strategy:

1. Extract PCB trace topology from the original image.
2. Generate defect geometry in the binary structural domain.
3. Render the defect back into the original image using local PCB color statistics.
4. Export labels automatically in YOLO format.
5. Optionally use Stable Diffusion / ComfyUI as a low-strength appearance harmonizer.

This design avoids common prompt-only SD problems, such as unrealistic metallic artifacts, solder-like blobs, white blobs, hallucinated components, or topology distortion.

---

## 3. Supported Defect Types

| Class ID | Defect Type | Status | Script | Description |
|---:|---|---|---|---|
| 0 | Short Circuit | Implemented | `generate_short_circuit.py` | Adds a conductive bridge between adjacent traces |
| 1 | Open Circuit | Implemented | `generate_open_circuit.py` | Cuts through a PCB trace to create a discontinuity |
| 2 | Spur | Implemented, v2 updated | `generate_spur.py` | Adds a protruding copper branch from an existing trace |
| 3 | Missing Hole | Implemented | `generate_missing_hole.py` | Replaces the center of a silver pad / solder joint with local trace-like color |
| 4 | Mouse Bite | Implemented | `generate_mouse_bite.py` | Removes a small notch from the edge of a PCB trace |
| 5 | Spurious Copper | Implemented, parameters refined | `generate_spurious_copper.py` | Adds detached or irregular extra copper residue near traces |

Important note:

```text
The earlier Missing Pad / Missing Hole plan has been refined to Missing Hole.
The current implementation focuses on missing-hole defects in silver pad / solder-joint regions.
```

---

## 4. Project Structure

```text
PCB_Defect_Synthesis/
├── configs/
│   └── hsv_trace_config.json
├── PCB_Dataset/
│   ├── 01.JPG
│   ├── 04.JPG
│   ├── 05.JPG
│   ├── 06.JPG
│   ├── 07.JPG
│   ├── 08.JPG
│   ├── 09.JPG
│   ├── 10.JPG
│   ├── 11.JPG
│   └── 12.JPG
├── outputs/
│   ├── topology/
│   ├── short_circuit/
│   ├── open_circuit/
│   ├── spur/
│   ├── missing_hole/
│   ├── mouse_bite/
│   ├── spurious_copper/
│   ├── normal/
│   └── batch_logs/
├── outputs_sd/
│   └── short_circuit/
├── dataset_yolo_crop_rule/
├── dataset_yolo_crop_rule_with_normal/
├── dataset_yolo_crop_rule_normal_spur_v2/
├── hsv_tracker.py
├── extract_topology.py
├── generate_short_circuit.py
├── generate_open_circuit.py
├── generate_spur.py
├── generate_missing_hole.py
├── generate_mouse_bite.py
├── generate_spurious_copper.py
├── generate_normal_crops.py
├── run_batch_generation.py
├── build_yolo_dataset.py
├── sd_harmonize_short_circuit.py
├── README.md
└── .gitignore
```

---

## 5. Environment Setup for Synthetic Generation

### 5.1 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 5.2 Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy requests
```

### 5.3 Test OpenCV Installation

```bash
python -c "import cv2; print(cv2.__version__)"
```

If a version number is printed, OpenCV is installed correctly.

---

## 6. Pipeline Overview

```text
Original PCB Images
        ↓
HSV Threshold Calibration
        ↓
Trace Topology Extraction
        ↓
Rule-Based Defect Generation
        ↓
Normal / Background Crop Generation
        ↓
YOLO Dataset Construction
        ↓
YOLO26 Training and Evaluation
        ↓
Optional SD / ComfyUI Appearance Harmonization
```

---

## 7. HSV Threshold Calibration

Script:

```text
hsv_tracker.py
```

Run:

```bash
python hsv_tracker.py --image PCB_Dataset/04.JPG
```

Adjust the HSV sliders until PCB traces are properly isolated. Save the configuration to:

```text
configs/hsv_trace_config.json
```

Example configuration:

```json
{
    "trace_hsv_lower": [35, 0, 0],
    "trace_hsv_upper": [90, 255, 75]
}
```

---

## 8. PCB Trace Topology Extraction

Script:

```text
extract_topology.py
```

Single-image example:

```bash
python extract_topology.py --image PCB_Dataset/04.JPG --config configs/hsv_trace_config.json --output_dir outputs/topology
```

Expected outputs:

```text
outputs/topology/
├── 04_raw_trace_mask.png
├── 04_trace_mask.png
├── 04_trace_overlay.png
├── 04_topology.png
├── 04_contours.png
├── 04_attack_candidate_mask.png
├── 04_attack_candidate_overlay.png
├── 04_attack_candidate_topology.png
└── 04_attack_candidate_contours.png
```

Important files:

| File | Description |
|---|---|
| `*_trace_mask.png` | Binary trace mask extracted from the original PCB |
| `*_attack_candidate_mask.png` | Candidate region for defect injection |
| `*_trace_overlay.png` | Debug overlay showing extracted trace regions |
| `*_topology.png` | Binary topology image for visualization |

---

## 9. Multi-Image Batch Generation

Script:

```text
run_batch_generation.py
```

The current dataset is generated from multiple PCB source images:

```text
01.JPG, 04.JPG, 05.JPG, 06.JPG, 07.JPG,
08.JPG, 09.JPG, 10.JPG, 11.JPG, 12.JPG
```

### 9.1 Dry Run

Use this first to check all child commands:

```powershell
python run_batch_generation.py --run_topology --num_samples 10 --dry_run
```

### 9.2 Full Batch Generation

Run topology extraction and six defect generators:

```powershell
python run_batch_generation.py --run_topology --num_samples 10
```

### 9.3 Defect-Only Generation

If topology files already exist:

```powershell
python run_batch_generation.py --num_samples 10
```

### 9.4 Generate Specific Defect Types

Example:

```powershell
python run_batch_generation.py --defects short_circuit spur spurious_copper --num_samples 10
```

### 9.5 Batch Logs

The script writes logs and summaries to:

```text
outputs/batch_logs/
├── batch_generation_<timestamp>.log
└── batch_generation_<timestamp>_summary.json
```

---

## 10. Generate Short Circuit Defects

Script:

```text
generate_short_circuit.py
```

Short circuit defects are generated by adding a copper-colored bridge between nearby trace boundaries.

Recommended parameters:

```bash
python generate_short_circuit.py --image PCB_Dataset/04.JPG --seed 42 --min_gap_distance 4 --max_gap_distance 34 --bridge_width_multiplier 0.9 --min_bridge_width 4 --max_bridge_width 18 --max_attempts 20000 --num_samples 20
```

Current batch defaults in `run_batch_generation.py`:

```text
--short_circuit_max_attempts 20000
--short_circuit_pad_dilate 24
--short_circuit_min_gap_distance 4
--short_circuit_max_gap_distance 34
--short_circuit_min_parallel_cos 0.85
```

---

## 11. Generate Open Circuit Defects

Script:

```text
generate_open_circuit.py
```

Open circuit defects are generated by cutting through an existing PCB trace and rendering the removed area with local substrate color.

Recommended parameters:

```bash
python generate_open_circuit.py --image PCB_Dataset/04.JPG --seed 42 --pad_dilate 28 --min_gap_length 10 --max_gap_length 22 --min_elongation 4.0 --max_attempts 15000 --num_samples 20
```

Current batch defaults:

```text
--open_circuit_pad_dilate 28
--open_circuit_min_gap_length 10
--open_circuit_max_gap_length 22
--open_circuit_min_elongation 4.0
--open_circuit_max_attempts 15000
```

---

## 12. Generate Spur Defects

Script:

```text
generate_spur.py
```

Spur defects are generated by adding a protruding copper branch from the edge of an existing trace.

### 12.1 Spur v2 Update

The spur generator has been updated after YOLO training showed severe spur false positives. The v2 logic makes spur defects more visually distinct from normal vias, dark holes, and small background texture.

Key improvements:

- longer and clearer spur geometry
- stronger minimum defect area
- minimum bbox long-side constraint
- mask elongation filtering
- dark feature / via avoidance
- stronger pad avoidance
- stricter geometry validation

### 12.2 Recommended Spur v2 Parameters

```bash
python generate_spur.py --image PCB_Dataset/04.JPG --seed 42 --min_spur_length 14 --max_spur_length 32 --spur_width_multiplier 0.65 --min_spur_width 4 --max_spur_width 10 --pad_dilate 30 --min_defect_area 45 --max_attempts 30000 --num_samples 20
```

Current recommended batch command:

```powershell
python run_batch_generation.py --defects spur --num_samples 10 --spur_min_length 14 --spur_max_length 32 --spur_pad_dilate 30 --spur_max_attempts 30000
```

---

## 13. Generate Missing Hole Defects

Script:

```text
generate_missing_hole.py
```

Missing hole defects are generated by replacing the center area of a silver pad / solder joint with a local trace-like or green board-region color.

Recommended parameters:

```bash
python generate_missing_hole.py --image PCB_Dataset/04.JPG --seed 42 --num_samples 20 --min_hole_radius 8 --max_hole_radius 18 --min_hole_radius_ratio 0.5 --max_hole_radius_ratio 0.7
```

These parameters generate a visually clear missing-hole region in the center of silver pads without making the defect unrealistically cover the whole pad.

---

## 14. Generate Mouse Bite Defects

Script:

```text
generate_mouse_bite.py
```

Mouse bite defects are generated by removing a small notch from the edge of an existing PCB trace.

Recommended parameters:

```bash
python generate_mouse_bite.py --image PCB_Dataset/04.JPG --seed 42 --min_radius 6 --max_radius 14 --max_attempts 3000 --num_samples 20
```

Current issue:

```text
Mouse bite is currently the weakest class in the best YOLO26s crop-level experiment.
Further optimization may require larger or more visually distinct bite geometry.
```

---

## 15. Generate Spurious Copper Defects

Script:

```text
generate_spurious_copper.py
```

Spurious copper defects are generated as detached extra copper residues near traces.

After early YOLO experiments, the original mixed and small-shape settings caused many false positives. The current recommended direction is to make spurious copper more stable and more visually distinguishable from small dark holes or normal background texture.

Recommended refined parameters:

```bash
python generate_spurious_copper.py --image PCB_Dataset/04.JPG --seed 42 --shape_mode ellipse --min_major_length 16 --max_major_length 34 --min_minor_length 8 --max_minor_length 16 --min_detach_gap 5 --max_detach_gap 14 --max_attempts 20000 --num_samples 20
```

Recommended batch command:

```powershell
python run_batch_generation.py --defects spurious_copper --num_samples 10 --spurious_copper_shape_mode ellipse --spurious_copper_min_major_length 16 --spurious_copper_max_major_length 34 --spurious_copper_min_minor_length 8 --spurious_copper_max_minor_length 16 --spurious_copper_min_detach_gap 5 --spurious_copper_max_detach_gap 14 --spurious_copper_min_defect_area 50 --spurious_copper_min_bbox_width 8 --spurious_copper_min_bbox_height 8
```

---

## 16. Generate Normal / Background Crops

Script:

```text
generate_normal_crops.py
```

Normal/background crops are used as negative samples to reduce YOLO false positives. They contain no defect annotations and therefore use empty `.txt` label files.

### 16.1 Generate Normal Crops

```powershell
python generate_normal_crops.py --dataset_dir PCB_Dataset --topology_dir outputs/topology --output_dir outputs/normal --num_crops_per_image 20 --crop_size 128 --seed 42
```

To overwrite existing normal crops:

```powershell
python generate_normal_crops.py --dataset_dir PCB_Dataset --topology_dir outputs/topology --output_dir outputs/normal --num_crops_per_image 20 --crop_size 128 --seed 42 --overwrite
```

### 16.2 Output Files

```text
outputs/normal/
├── 01_normal_0001_crop.png
├── 01_normal_0001_crop.txt
├── 01_normal_0001_crop_debug.png
├── 01_normal_0001_crop_metadata.json
└── normal_crops_summary.json
```

Important:

```text
The normal crop label file is intentionally empty.
Do not add a background class to data.yaml.
```

YOLO uses empty label files as background / negative samples.

---

## 17. Output File Types

Each defect generator produces several types of outputs:

| Output Type | Example | Used for Training? | Description |
|---|---|---:|---|
| Synthetic image | `04_mouse_bite.png` | Yes | Full PCB image with synthetic defect |
| Defect mask | `04_mouse_bite_mask.png` | Optional | Binary defect mask |
| YOLO label | `04_mouse_bite.txt` | Yes | YOLO-format annotation |
| Debug image | `04_mouse_bite_debug.png` | No | Synthetic image with red mask and green bbox |
| Crop image | `04_mouse_bite_crop.png` | Yes | Defect-centered local patch |
| Crop label | `04_mouse_bite_crop.txt` | Yes | YOLO label for crop image |
| Crop debug | `04_mouse_bite_crop_debug.png` | No | Crop image with mask and bbox |
| Metadata | `04_mouse_bite_metadata.json` | No | Generation parameters and sample information |

Debug images should not be used for training.

---

## 18. YOLO Label Format

Each generated defect automatically produces a YOLO-format annotation:

```text
class_id x_center y_center width height
```

Example:

```text
1 0.533704 0.373580 0.003599 0.008929
```

Current class mapping:

```yaml
names:
  0: short_circuit
  1: open_circuit
  2: spur
  3: missing_hole
  4: mouse_bite
  5: spurious_copper
```

For normal/background crops, the `.txt` label file is empty.

---

## 19. Build YOLO Dataset

Script:

```text
build_yolo_dataset.py
```

This script builds a YOLO-format dataset from generated crop or full-image samples.

### 19.1 Output Structure

```text
dataset_yolo_crop_rule_normal_spur_v2/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── build_summary.json
```

### 19.2 Build Crop Dataset with Normal Crops

```powershell
python build_yolo_dataset.py --image_mode crop --source_dirs outputs/short_circuit outputs/open_circuit outputs/spur outputs/missing_hole outputs/mouse_bite outputs/spurious_copper outputs/normal --output_dir dataset_yolo_crop_rule_normal_spur_v2 --allow_empty_labels --overwrite
```

### 19.3 Split Strategy

The default split mode is group-based:

```text
--split_mode group
```

This means samples are split by source image ID, such as `01`, `04`, `05`, etc. This prevents samples generated from the same original PCB image from being placed in both train and validation/test splits.

Manual split example:

```powershell
python build_yolo_dataset.py --image_mode crop --source_dirs outputs/short_circuit outputs/open_circuit outputs/spur outputs/missing_hole outputs/mouse_bite outputs/spurious_copper outputs/normal --output_dir dataset_yolo_crop_rule_normal_spur_v2 --split_mode manual --train_sources 01 04 05 06 07 08 --val_sources 09 10 --test_sources 11 12 --allow_empty_labels --overwrite
```

---

## 20. YOLO26 Environment Setup

Use a separate environment for YOLO26 training.

```powershell
cd D:\PolyU\PCB_Defect_Synthesis
conda create -n yolo26 python=3.11 -y
conda activate yolo26
```

Install PyTorch and Ultralytics:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U ultralytics
```

Check CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 21. YOLO26 Training

### 21.1 Recommended Current Model

YOLO26s is currently recommended for subsequent horizontal experiments.

Reason:

```text
YOLO26n showed insufficient stability on fine-grained classes such as spur and spurious_copper.
YOLO26s significantly improved crop-level detection performance.
```

### 21.2 Recommended Training Command

```powershell
yolo detect train model=yolo26s.pt data=dataset_yolo_crop_rule_normal_spur_v2/data.yaml imgsz=320 epochs=30 batch=16 device=0 project=runs_pcb name=yolo26s_crop_rule_normal_spur_v2_img320
```

### 21.3 Debug Training

```powershell
yolo detect train model=yolo26s.pt data=dataset_yolo_crop_rule_normal_spur_v2/data.yaml imgsz=320 epochs=1 batch=8 device=0 project=runs_pcb name=debug_1epoch
```

---

## 22. Current YOLO26 Experiment Summary

### 22.1 YOLO26n Baseline

Earlier YOLO26n experiments showed that most classes were learnable, but spur and spurious copper produced severe false positives.

Example issue:

```text
spurious_copper was initially confused with normal small holes and dark PCB texture.
After parameter refinement, the dominant false-positive issue moved to spur.
```

### 22.2 Normal Crops

Adding normal/background crops reduced some false positives, but did not fully solve spur misclassification when spur v1 geometry was too similar to normal small PCB structures.

### 22.3 Spur v2

After updating `generate_spur.py`, spur became more visually distinct. Combined with normal crops and YOLO26s, crop-level performance improved substantially.

### 22.4 YOLO26s Result Snapshot

Best observed configuration:

```text
Dataset: dataset_yolo_crop_rule_normal_spur_v2
Model: YOLO26s
imgsz: 320
epochs: 30
```

Observed validation summary:

```text
all:
P        ≈ 0.944
R        ≈ 0.930
mAP50    ≈ 0.952
mAP50-95 ≈ 0.774

short_circuit:
P        ≈ 0.980
R        ≈ 1.000
mAP50    ≈ 0.995

open_circuit:
P        ≈ 1.000
R        ≈ 0.924
mAP50    ≈ 0.995

spur:
P        ≈ 1.000
R        ≈ 0.958
mAP50    ≈ 0.995

missing_hole:
P        ≈ 0.970
R        ≈ 1.000
mAP50    ≈ 0.995

mouse_bite:
P        ≈ 0.832
R        ≈ 0.700
mAP50    ≈ 0.738

spurious_copper:
P        ≈ 0.882
R        ≈ 1.000
mAP50    ≈ 0.995
```

Current weakest class:

```text
mouse_bite
```

The next defect-generation improvement should focus on `generate_mouse_bite.py`.

---

## 23. Stable Diffusion / ComfyUI Appearance Harmonization

Stable Diffusion is not used as a free-form defect generator.

Its intended role is:

```text
Appearance Harmonizer
```

The defect geometry is still generated by rule-based topology operations. SD/ComfyUI may be used only for subtle local blending under strict constraints:

- low denoise strength
- defect-local mask
- no topology distortion
- no new metallic artifacts
- no white blob generation
- no hallucinated components

### 23.1 Short Circuit SD Harmonization

Script:

```text
sd_harmonize_short_circuit.py
```

The current short-circuit SD harmonization experiment showed limited visual benefit. Therefore, SD outputs are kept as optional comparison data rather than the main training source.

Example command:

```powershell
python sd_harmonize_short_circuit.py --input_dir outputs/short_circuit --output_dir outputs_sd/short_circuit --workflow_api_json workflows/short_circuit_workflow_api.json --node_ids_json workflows/short_circuit_node_ids.json --comfy_url http://127.0.0.1:8000 --mask_mode soft_from_crop --steps 16 --cfg 2.0 --sampler_name euler --scheduler simple --denoise 0.05
```

---

## 24. Suggested Experimental Design

### 24.1 Model Selection

Fixed dataset:

```text
dataset_yolo_crop_rule_normal_spur_v2
```

Compare:

```text
YOLO26n vs YOLO26s
```

Conclusion:

```text
YOLO26s is selected as the default detector for subsequent experiments.
```

### 24.2 Data Pipeline Ablation

Fixed model:

```text
YOLO26s
```

Suggested comparisons:

| Experiment | Data |
|---|---|
| Crop rule-only | Six synthetic defect classes, no normal crops |
| Crop rule + normal | Six classes + normal/background crops |
| Crop rule + normal + spur_v2 | Current best crop-level configuration |
| Crop rule + SD short circuit | Optional SD harmonized short-circuit comparison |

### 24.3 Data Scale Experiment

Fixed model:

```text
YOLO26s
```

Compare different numbers of generated samples:

```text
10 samples per image per class
20 samples per image per class
30 samples per image per class
```

### 24.4 Crop vs Full Image

Fixed model:

```text
YOLO26s
```

Compare:

```text
crop-level dataset
full-image dataset
```

This is important because crop-level training validates local defect learnability, while full-image training is closer to real inspection deployment.

---

## 25. Typical Current Workflow

```powershell
# 1. Calibrate HSV threshold
python hsv_tracker.py --image PCB_Dataset/04.JPG

# 2. Batch topology + defect generation
python run_batch_generation.py --run_topology --num_samples 10

# 3. Generate normal/background crops
python generate_normal_crops.py --dataset_dir PCB_Dataset --topology_dir outputs/topology --output_dir outputs/normal --num_crops_per_image 20 --crop_size 128 --seed 42 --overwrite

# 4. Build YOLO crop dataset
python build_yolo_dataset.py --image_mode crop --source_dirs outputs/short_circuit outputs/open_circuit outputs/spur outputs/missing_hole outputs/mouse_bite outputs/spurious_copper outputs/normal --output_dir dataset_yolo_crop_rule_normal_spur_v2 --allow_empty_labels --overwrite

# 5. Train YOLO26s
yolo detect train model=yolo26s.pt data=dataset_yolo_crop_rule_normal_spur_v2/data.yaml imgsz=320 epochs=30 batch=16 device=0 project=runs_pcb name=yolo26s_crop_rule_normal_spur_v2_img320
```

---

## 26. Current Progress Checklist

Completed:

- [x] HSV threshold calibration
- [x] PCB trace topology extraction
- [x] Mouse Bite generation
- [x] Open Circuit generation
- [x] Short Circuit generation
- [x] Spur generation
- [x] Spur v2 update
- [x] Spurious Copper generation
- [x] Spurious Copper parameter refinement
- [x] Missing Hole generation
- [x] Multi-image batch generation
- [x] YOLO label export
- [x] Defect mask export
- [x] Defect-centered crop export
- [x] Debug visualization export
- [x] Metadata export
- [x] Normal/background crop generation
- [x] YOLO dataset construction
- [x] YOLO26n crop-level training
- [x] YOLO26s crop-level training
- [x] YOLO26s selected for subsequent horizontal comparison

In progress / next steps:

- [ ] Optimize mouse_bite generation
- [ ] Build full-image YOLO dataset
- [ ] Compare crop-level vs full-image training
- [ ] Run data scale experiments
- [ ] Optionally compare rule-only vs rule + SD harmonized data
- [ ] Prepare final experimental tables and qualitative prediction visualizations

---

## 27. Notes

- Do not train on debug images.
- Do not create a background class for YOLO.
- Normal/background crops should use empty `.txt` label files.
- Use `--allow_empty_labels` when building datasets that include normal crops.
- For current crop-level experiments, use `imgsz=320`.
- Use YOLO26s as the default model for horizontal comparisons.
- YOLO26n can be retained as a lightweight baseline in the model selection experiment.
