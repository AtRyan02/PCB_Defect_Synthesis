# PCB_Defect_Synthesis

A topology-guided synthetic data generation pipeline for PCB defect detection.

This project aims to generate controllable PCB defect samples for downstream object detection tasks. Instead of relying on prompt-only Stable Diffusion generation, the current pipeline first extracts PCB trace topology, injects rule-based defect geometry, and then renders the defect appearance using local PCB color statistics.

The generated samples include synthetic PCB images, defect masks, YOLO-format labels, defect-centered crops, debug visualizations, and metadata files.

---

## 1. Project Motivation

Real PCB defect samples are often difficult to collect, expensive to annotate, and highly imbalanced across defect categories. This project explores a controllable synthetic data factory for PCB defect generation, with the goal of improving YOLO-based detection performance under long-tail industrial inspection scenarios.

The current design follows a **geometry-first** strategy:

1. Extract PCB trace topology from the original image.
2. Generate defect geometry in the binary structural domain.
3. Render the defect back into the original image using local PCB color statistics.
4. Export labels automatically in YOLO format.
5. Optionally use Stable Diffusion / ComfyUI later as an appearance harmonizer.

This design avoids common prompt-only SD problems, such as unrealistic metallic artifacts, solder-like blobs, or topology distortion.

---

## 2. Current Supported Defect Types

The current implementation supports:

| Defect Type | Status | Script | Description |
|---|---|---|---|
| Mouse Bite | Implemented | `generate_mouse_bite.py` | Removes a small notch from the edge of a PCB trace |
| Open Circuit | Implemented | `generate_open_circuit.py` | Cuts through a PCB trace to create a discontinuity |
| Short Circuit | Implemented | `generate_short_circuit.py` | Adds a conductive bridge between adjacent traces |
| Spur | Implemented | `generate_spur.py` | Adds a small protruding branch from an existing trace |
| Spurious Copper | Implemented | `generate_spurious_copper.py` | Adds detached or irregular extra copper residue near traces |
| Missing Hole | Implemented | `generate_missing_hole.py` | Replaces the center of a silver pad / solder joint with local trace-like color |

Current stable modules:

- `Mouse Bite Generator v1.0`
- `Open Circuit Generator v1.0`
- `Short Circuit Generator v1.0`
- `Spur Generator v1.0`
- `Spurious Copper Generator v1.0`
- `Missing Hole Generator v1.0`

Note:

```text
The previous Missing Pad / Missing Hole plan has been refined to Missing Hole.
The current implementation focuses on missing-hole defects in silver pad / solder-joint regions.
```

---

## 3. Project Structure

```text
PCB_Defect_Synthesis/
├── configs/
│   └── hsv_trace_config.json
├── PCB_Dataset/
│   ├── 01.JPG
│   ├── 04.JPG
│   ├── 05.JPG
│   └── ...
├── outputs/
│   ├── topology/
│   ├── mouse_bite/
│   ├── open_circuit/
│   ├── short_circuit/
│   ├── spur/
│   ├── spurious_copper/
│   └── missing_hole/
├── hsv_tracker.py
├── extract_topology.py
├── generate_mouse_bite.py
├── generate_open_circuit.py
├── generate_short_circuit.py
├── generate_spur.py
├── generate_spurious_copper.py
├── generate_missing_hole.py
├── README.md
└── .gitignore
```

---

## 4. Environment Setup

### 4.1 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4.2 Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy
```

### 4.3 Test OpenCV Installation

```bash
python -c "import cv2; print(cv2.__version__)"
```

If a version number is printed, OpenCV is installed correctly.

---

## 5. Pipeline Overview

The current pipeline contains three main stages:

```text
Original PCB Image
        ↓
HSV Threshold Calibration
        ↓
Trace Topology Extraction
        ↓
Rule-Based Defect Generation
        ↓
Appearance Rendering
        ↓
YOLO Label + Crop + Mask + Metadata Export
```

---

## 6. Step 1: HSV Threshold Calibration

Script:

```text
hsv_tracker.py
```

This script provides an interactive HSV threshold tracker for extracting dark PCB traces.

Run:

```bash
python hsv_tracker.py --image PCB_Dataset/04.JPG
```

Adjust the HSV sliders until the PCB traces are properly isolated.

After selecting suitable values, save the configuration to:

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

## 7. Step 2: PCB Trace Topology Extraction

Script:

```text
extract_topology.py
```

This script extracts the trace mask and topology map from the original PCB image using the saved HSV configuration.

Run:

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
| `04_trace_mask.png` | Binary trace mask extracted from the original PCB |
| `04_attack_candidate_mask.png` | Candidate region for defect injection |
| `04_trace_overlay.png` | Debug overlay showing extracted trace regions |
| `04_topology.png` | Binary topology image for visualization |

---

## 8. Step 3: Generate Mouse Bite Defects

Script:

```text
generate_mouse_bite.py
```

Mouse bite defects are generated by removing a small notch from the edge of an existing PCB trace. The removed area is rendered using local substrate color.

### 8.1 Single Sample Generation

```bash
python generate_mouse_bite.py --image PCB_Dataset/04.JPG --seed 42 --min_radius 6 --max_radius 14 --max_attempts 3000
```

### 8.2 Batch Generation

```bash
python generate_mouse_bite.py --image PCB_Dataset/04.JPG --seed 42 --min_radius 6 --max_radius 14 --max_attempts 3000 --num_samples 20
```

### 8.3 Output Files

Example outputs:

```text
outputs/mouse_bite/
├── 04_mouse_bite.png
├── 04_mouse_bite_mask.png
├── 04_mouse_bite_debug.png
├── 04_mouse_bite.txt
├── 04_mouse_bite_crop.png
├── 04_mouse_bite_crop_mask.png
├── 04_mouse_bite_crop_debug.png
├── 04_mouse_bite_crop.txt
└── 04_mouse_bite_metadata.json
```

For batch generation:

```text
04_mouse_bite_0001.png
04_mouse_bite_0001_mask.png
04_mouse_bite_0001_debug.png
04_mouse_bite_0001.txt
04_mouse_bite_0001_crop.png
04_mouse_bite_0001_crop_debug.png
04_mouse_bite_0001_metadata.json
...
```

### 8.4 Recommended Parameters

Current recommended parameters:

```bash
--min_radius 6
--max_radius 14
--max_attempts 3000
```

These values produce visually reasonable mouse bite defects without frequently cutting through the entire trace.

---

## 9. Step 4: Generate Open Circuit Defects

Script:

```text
generate_open_circuit.py
```

Open circuit defects are generated by cutting through an existing PCB trace. The removed trace region is rendered using local substrate color.

The generator includes:

- trace centerline sampling
- pad avoidance
- straight trace filtering
- local connectivity validation
- YOLO label export
- crop output
- metadata output

### 9.1 Single Sample Generation

```bash
python generate_open_circuit.py --image PCB_Dataset/04.JPG --seed 42 --pad_dilate 28 --min_gap_length 10 --max_gap_length 22 --min_elongation 4.0 --max_attempts 15000
```

### 9.2 Batch Generation

```bash
python generate_open_circuit.py --image PCB_Dataset/04.JPG --seed 42 --pad_dilate 28 --min_gap_length 10 --max_gap_length 22 --min_elongation 4.0 --max_attempts 15000 --num_samples 20
```

### 9.3 Output Files

Example outputs:

```text
outputs/open_circuit/
├── 04_open_circuit.png
├── 04_open_circuit_mask.png
├── 04_open_circuit_debug.png
├── 04_open_circuit.txt
├── 04_open_circuit_crop.png
├── 04_open_circuit_crop_mask.png
├── 04_open_circuit_crop_debug.png
├── 04_open_circuit_crop.txt
└── 04_open_circuit_metadata.json
```

For batch generation:

```text
04_open_circuit_0001.png
04_open_circuit_0001_mask.png
04_open_circuit_0001_debug.png
04_open_circuit_0001.txt
04_open_circuit_0001_crop.png
04_open_circuit_0001_crop_debug.png
04_open_circuit_0001_metadata.json
...
```

### 9.4 Recommended Parameters

Current recommended parameters:

```bash
--pad_dilate 28
--min_gap_length 10
--max_gap_length 22
--min_elongation 4.0
--max_attempts 15000
```

These parameters help the generator select long, straight trace regions while avoiding pads, corners, junctions, and text regions.

---

## 10. Step 5: Generate Short Circuit Defects

Script:

```text
generate_short_circuit.py
```

Short circuit defects are generated by adding a copper-colored bridge between nearby trace boundaries. The generated bridge uses local trace color statistics and is validated by distance, local trace width, PCA direction consistency, overlap ratio, and defect area constraints.

The generator includes:

- trace-edge candidate sampling
- adjacent-trace search along local normal direction
- PCA-based parallel trace validation
- solder pad / joint avoidance
- bridge-width control from local trace width
- YOLO label export
- crop output
- metadata output

### 10.1 Single Sample Generation

```bash
python generate_short_circuit.py --image PCB_Dataset/04.JPG --seed 42 --min_gap_distance 4 --max_gap_distance 34 --bridge_width_multiplier 0.9 --min_bridge_width 4 --max_bridge_width 18 --max_attempts 20000
```

### 10.2 Batch Generation

```bash
python generate_short_circuit.py --image PCB_Dataset/04.JPG --seed 42 --min_gap_distance 4 --max_gap_distance 34 --bridge_width_multiplier 0.9 --min_bridge_width 4 --max_bridge_width 18 --max_attempts 20000 --num_samples 20
```

### 10.3 Output Files

Example outputs:

```text
outputs/short_circuit/
├── 04_short_circuit.png
├── 04_short_circuit_mask.png
├── 04_short_circuit_debug.png
├── 04_short_circuit.txt
├── 04_short_circuit_crop.png
├── 04_short_circuit_crop_mask.png
├── 04_short_circuit_crop_debug.png
├── 04_short_circuit_crop.txt
└── 04_short_circuit_metadata.json
```

For batch generation:

```text
04_short_circuit_0001.png
04_short_circuit_0001_mask.png
04_short_circuit_0001_debug.png
04_short_circuit_0001.txt
04_short_circuit_0001_crop.png
04_short_circuit_0001_crop_debug.png
04_short_circuit_0001_metadata.json
...
```

### 10.4 Recommended Parameters

Current recommended parameters:

```bash
--min_gap_distance 4
--max_gap_distance 34
--bridge_width_multiplier 0.9
--min_bridge_width 4
--max_bridge_width 18
--max_attempts 20000
```

These parameters generate visible conductive bridges while avoiding overly thick copper blobs or pad-near artifacts.

---

## 11. Step 6: Generate Spur Defects

Script:

```text
generate_spur.py
```

Spur defects are generated by adding a short protruding copper branch from the edge of an existing trace. The branch direction is estimated from the local outward normal and includes a small angular jitter.

The generator includes:

- trace-edge sampling
- local PCA trace validation
- outward-normal estimation
- spur length and width control
- pad avoidance
- short-circuit rejection through connectivity validation
- YOLO label export
- crop output
- metadata output

### 11.1 Single Sample Generation

```bash
python generate_spur.py --image PCB_Dataset/04.JPG --seed 42 --min_spur_length 8 --max_spur_length 22 --spur_width_multiplier 0.65 --min_spur_width 3 --max_spur_width 10 --angle_jitter_deg 25 --max_attempts 15000
```

### 11.2 Batch Generation

```bash
python generate_spur.py --image PCB_Dataset/04.JPG --seed 42 --min_spur_length 8 --max_spur_length 22 --spur_width_multiplier 0.65 --min_spur_width 3 --max_spur_width 10 --angle_jitter_deg 25 --max_attempts 15000 --num_samples 20
```

### 11.3 Output Files

Example outputs:

```text
outputs/spur/
├── 04_spur.png
├── 04_spur_mask.png
├── 04_spur_debug.png
├── 04_spur.txt
├── 04_spur_crop.png
├── 04_spur_crop_mask.png
├── 04_spur_crop_debug.png
├── 04_spur_crop.txt
└── 04_spur_metadata.json
```

For batch generation:

```text
04_spur_0001.png
04_spur_0001_mask.png
04_spur_0001_debug.png
04_spur_0001.txt
04_spur_0001_crop.png
04_spur_0001_crop_debug.png
04_spur_0001_metadata.json
...
```

### 11.4 Recommended Parameters

Current recommended parameters:

```bash
--min_spur_length 8
--max_spur_length 22
--spur_width_multiplier 0.65
--min_spur_width 3
--max_spur_width 10
--angle_jitter_deg 25
--max_attempts 15000
```

These values produce small but visible protruding branches while reducing the chance of accidentally connecting to nearby traces.

---

## 12. Step 7: Generate Spurious Copper Defects

Script:

```text
generate_spurious_copper.py
```

Spurious copper defects are generated as detached extra copper residues near traces. The geometry can be rectangular, trapezoidal, elliptical, or mixed.

The generator includes:

- trace-edge anchor sampling
- local tangent and outward-normal estimation
- detached placement near trace boundaries
- mixed shape generation
- distance-to-trace validation
- component-gain validation
- YOLO label export
- crop output
- metadata output

### 12.1 Single Sample Generation

```bash
python generate_spurious_copper.py --image PCB_Dataset/04.JPG --seed 42 --shape_mode mixed --min_major_length 10 --max_major_length 26 --min_minor_length 4 --max_minor_length 12 --min_detach_gap 3 --max_detach_gap 10 --max_attempts 20000
```

### 12.2 Batch Generation

```bash
python generate_spurious_copper.py --image PCB_Dataset/04.JPG --seed 42 --shape_mode mixed --min_major_length 10 --max_major_length 26 --min_minor_length 4 --max_minor_length 12 --min_detach_gap 3 --max_detach_gap 10 --max_attempts 20000 --num_samples 20
```

### 12.3 Output Files

Example outputs:

```text
outputs/spurious_copper/
├── 04_spurious_copper.png
├── 04_spurious_copper_mask.png
├── 04_spurious_copper_debug.png
├── 04_spurious_copper.txt
├── 04_spurious_copper_crop.png
├── 04_spurious_copper_crop_mask.png
├── 04_spurious_copper_crop_debug.png
├── 04_spurious_copper_crop.txt
└── 04_spurious_copper_metadata.json
```

For batch generation:

```text
04_spurious_copper_0001.png
04_spurious_copper_0001_mask.png
04_spurious_copper_0001_debug.png
04_spurious_copper_0001.txt
04_spurious_copper_0001_crop.png
04_spurious_copper_0001_crop_debug.png
04_spurious_copper_0001_metadata.json
...
```

### 12.4 Recommended Parameters

Current recommended parameters:

```bash
--shape_mode mixed
--min_major_length 10
--max_major_length 26
--min_minor_length 4
--max_minor_length 12
--min_detach_gap 3
--max_detach_gap 10
--max_attempts 20000
```

These values generate small detached copper residues near traces while avoiding direct overlap with existing trace and pad regions.

---

## 13. Step 8: Generate Missing Hole Defects

Script:

```text
generate_missing_hole.py
```

Missing hole defects are generated by replacing the center area of a silver pad / solder joint with a local trace-like or green board-region color. The intended visual style is that most silver pads remain normal, while selected abnormal pads show a visible green-colored center region.

The generator includes:

- silver pad / solder-joint detection
- connected-component pad filtering
- center-safe hole placement
- circle / ellipse hole geometry
- local non-silver / green-region color sampling
- YOLO label export
- crop output
- metadata output

### 13.1 Single Sample Generation

```bash
python generate_missing_hole.py --image PCB_Dataset/04.JPG --seed 42 --min_hole_radius 8 --max_hole_radius 18 --min_hole_radius_ratio 0.5 --max_hole_radius_ratio 0.7
```

### 13.2 Batch Generation

```bash
python generate_missing_hole.py --image PCB_Dataset/04.JPG --seed 42 --num_samples 20 --min_hole_radius 8 --max_hole_radius 18 --min_hole_radius_ratio 0.5 --max_hole_radius_ratio 0.7
```

### 13.3 Output Files

Example outputs:

```text
outputs/missing_hole/
├── 04_missing_hole.png
├── 04_missing_hole_mask.png
├── 04_missing_hole_debug.png
├── 04_missing_hole.txt
├── 04_missing_hole_crop.png
├── 04_missing_hole_crop_mask.png
├── 04_missing_hole_crop_debug.png
├── 04_missing_hole_crop.txt
└── 04_missing_hole_metadata.json
```

For batch generation:

```text
04_missing_hole_0001.png
04_missing_hole_0001_mask.png
04_missing_hole_0001_debug.png
04_missing_hole_0001.txt
04_missing_hole_0001_crop.png
04_missing_hole_0001_crop_debug.png
04_missing_hole_0001_metadata.json
...
```

### 13.4 Recommended Parameters

Current recommended parameters:

```bash
--min_hole_radius 8
--max_hole_radius 18
--min_hole_radius_ratio 0.5
--max_hole_radius_ratio 0.7
```

These parameters generate a visually clear missing-hole region in the center of silver pads without making the defect unrealistically cover the whole pad.

---

## 14. YOLO Label Format

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

Currently implemented:

```text
0: short_circuit
1: open_circuit
2: spur
3: missing_hole
4: mouse_bite
5: spurious_copper
```

---

## 15. Output File Types

Each generator produces several types of outputs:

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

Important:

```text
Debug images should not be used for training.
```

Only use clean synthetic images and crop images for model training.

---

## 16. Metadata

Each generated sample includes a metadata JSON file.

Metadata records:

- source image path
- defect type
- class ID
- sample index
- defect center
- geometry parameters
- defect area
- bounding box size
- substrate / trace / fill color
- YOLO label
- crop information
- generation parameters
- output paths

This makes the synthetic dataset reproducible, traceable, and easier to analyze.

---

## 17. Suggested Experimental Design

The generated data can support the following YOLO training configurations:

| Experiment | Training Data |
|---|---|
| Real Only | Real PCB defect data only |
| Synthetic Only | Generated defect data only |
| Real + Synthetic | Real data augmented with generated defects |
| Rule-only Synthetic | Rule-based rendering without SD harmonization |
| Rule + SD Harmonized | Rule-based geometry with SD/ComfyUI appearance refinement |

Evaluation metrics:

- mAP
- Precision
- Recall
- F1-score
- OOD robustness
- defect-wise performance

---

## 18. Future Work

Remaining work:

1. Build a full YOLO dataset export tool.
2. Generate a balanced multi-class synthetic dataset across all six defect types.
3. Train and evaluate YOLO on real-only, synthetic-only, and mixed datasets.
4. Analyze per-class performance, especially long-tail classes.
5. Integrate Controlled SD / ComfyUI for low-strength appearance harmonization if rule-based rendering is not visually sufficient.
6. Compare rule-only synthetic data with rule + SD harmonized data.

---

## 19. Notes on Stable Diffusion / ComfyUI

Stable Diffusion is not used as a free-form defect generator in the current pipeline.

Instead, the long-term role of SD / ComfyUI is:

```text
Appearance Harmonizer
```

The defect geometry is generated by rule-based topology operations. SD can later be used only for subtle local appearance blending under strict constraints, such as:

- low denoise strength
- defect-local mask
- ControlNet / Canny guidance
- no new metallic artifacts
- no topology distortion

This design preserves the controllability of the generated defect while still allowing generative AI to improve visual realism.

---

## 20. Typical Workflow

A typical workflow for one PCB image is:

```bash
# 1. Calibrate HSV threshold
python hsv_tracker.py --image PCB_Dataset/04.JPG

# 2. Extract trace topology
python extract_topology.py --image PCB_Dataset/04.JPG --config configs/hsv_trace_config.json --output_dir outputs/topology

# 3. Generate mouse bite samples
python generate_mouse_bite.py --image PCB_Dataset/04.JPG --seed 42 --min_radius 6 --max_radius 14 --max_attempts 3000 --num_samples 20

# 4. Generate open circuit samples
python generate_open_circuit.py --image PCB_Dataset/04.JPG --seed 42 --pad_dilate 28 --min_gap_length 10 --max_gap_length 22 --min_elongation 4.0 --max_attempts 15000 --num_samples 20

# 5. Generate short circuit samples
python generate_short_circuit.py --image PCB_Dataset/04.JPG --seed 42 --min_gap_distance 4 --max_gap_distance 34 --bridge_width_multiplier 0.9 --min_bridge_width 4 --max_bridge_width 18 --max_attempts 20000 --num_samples 20

# 6. Generate spur samples
python generate_spur.py --image PCB_Dataset/04.JPG --seed 42 --min_spur_length 8 --max_spur_length 22 --spur_width_multiplier 0.65 --min_spur_width 3 --max_spur_width 10 --angle_jitter_deg 25 --max_attempts 15000 --num_samples 20

# 7. Generate spurious copper samples
python generate_spurious_copper.py --image PCB_Dataset/04.JPG --seed 42 --shape_mode mixed --min_major_length 10 --max_major_length 26 --min_minor_length 4 --max_minor_length 12 --min_detach_gap 3 --max_detach_gap 10 --max_attempts 20000 --num_samples 20

# 8. Generate missing hole samples
python generate_missing_hole.py --image PCB_Dataset/04.JPG --seed 42 --num_samples 20 --min_hole_radius 8 --max_hole_radius 18 --min_hole_radius_ratio 0.5 --max_hole_radius_ratio 0.7
```

---

## 21. Current Progress

Current completed modules:

- [x] HSV threshold calibration
- [x] PCB trace topology extraction
- [x] Mouse Bite generation
- [x] Open Circuit generation
- [x] Short Circuit generation
- [x] Spur generation
- [x] Spurious Copper generation
- [x] Missing Hole generation
- [x] YOLO label export
- [x] Defect mask export
- [x] Defect-centered crop export
- [x] Debug visualization export
- [x] Metadata export

In progress / next steps:

- [ ] Full YOLO dataset export tool
- [ ] Balanced synthetic dataset generation
- [ ] Controlled SD / ComfyUI harmonization
- [ ] YOLO training and robustness evaluation
