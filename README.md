# Total Care: Autonomous Driving Object Detection & Hazard Warning System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v5%20%7C%20v9-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Total Care** is an end-to-end computer vision and deep learning framework built on **YOLOv5** and **YOLOv9** designed for real-time autonomous vehicle safety, exterior hazard detection, and in-cabin occupant monitoring.

---

## 🎯 Definite Use Cases & Objectives

The **Total Care Safety Engine** solves critical perception and safety challenges in modern autonomous driving systems by monitoring both exterior road conditions and interior vehicle safety:

| Use Case Domain | Functional Target | Detected Classes | Safety Impact |
| :--- | :--- | :--- | :--- |
| **Vulnerable Road Users (VRU)** | Pedestrians, Cyclists, Micro-mobility | `pedestrian`, `biker`, `scooter` | Prevents collisions with high-risk vulnerable road users |
| **Traffic Perception** | Vehicle & Traffic Control Identification | `car`, `trafficLight`, `trafficLight-Green`, `trafficLight-GreenLeft`, `trafficLight-Red` | Enables automated stop/go compliance and lane tracking |
| **Road Surface Hazards** | Physical Damage & Obstacles | `pothole`, `blocked_left`, `blocked_right`, `blocked_straight`, `free_path` | Triggers evasive steering / braking before hitting road damage |
| **Wildlife Collision Prevention** | Animal Road-Crossing Warnings | `deer`, `roe_deer`, `elk`, `waterdeer` | Mitigates rural/highway animal collisions |
| **In-Cabin Safety** | Driver & Occupant Compliance | `belt`, `no_belt` | Enforces seatbelt usage and triggers cabin safety alerts |

---

## 📂 System Directory Structure

```gcode
Totalcare_autonomous_driving_Yolov5/
├── data_config.yaml           # Dataset path configuration & 19-class mappings
├── requirements.txt           # Python dependencies (PyTorch, Ultralytics, OpenCV, etc.)
├── README.md                  # System documentation & execution guide
├── src/                       # Core python execution modules
│   ├── __init__.py            # Package initializer
│   ├── dataset_prep.py        # Dataset remapping, splitting & YAML config tool
│   ├── train.py               # Unified training script for YOLOv5 & YOLOv9
│   ├── detect.py              # Real-time inference & Hazard Alert Overlay Engine
│   └── export.py              # Model export utility (ONNX, TensorRT, TorchScript)
├── tests/                     # Baseline notebook experiments & verification
│   ├── self_driving_object_detection_asset_baseline.ipynb
│   └── yolo_pretrain_v5.ipynb
└── Updates/                   # Historical project iterations & Colab notebooks
    ├── 221129/                # Initial baseline experiments
    ├── 221130/                # Class expansion iteration
    ├── 221201/                # Alpha model release
    └── 240427/                # YOLOv9 model architecture updates
```

---

## 🛠️ Environment Setup & Installation

### Option 1: Local Python Environment
Ensure Python 3.8+ and PyTorch are installed.

```bash
# Clone the repository
git clone https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5.git
cd Totalcare_autonomous_driving_Yolov5

# (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Option 2: Google Colab Setup
To run training or inference inside Google Colab:
```python
!git clone https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5.git
%cd Totalcare_autonomous_driving_Yolov5
!pip install -r requirements.txt
```

---

## 🚀 How to Run - Step-by-Step Instructions

### Step 1: Prepare & Structure Dataset
To convert raw image and label folders into standard YOLO dataset splits (`images/train`, `images/val`, `labels/train`, `labels/val`) and generate `data_config.yaml`:

```bash
python3 src/dataset_prep.py \
    --raw-dir ./raw_dataset \
    --output-dir ./dataset \
    --val-split 0.2 \
    --test-split 0.1 \
    --config-out ./data_config.yaml
```

---

### Step 2: Train the Object Detection Model
Train a model using either **YOLOv5** (`yolov5s.pt`, `yolov5m.pt`) or **YOLOv9** (`yolov9c.pt`) architecture:

```bash
# Train YOLOv5s baseline (50 epochs, batch size 16, image resolution 416)
python3 src/train.py \
    --model yolov5s.pt \
    --data ./data_config.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 416 \
    --name total_care_v5s

# Train YOLOv9c model for enhanced accuracy
python3 src/train.py \
    --model yolov9c.pt \
    --data ./data_config.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 416 \
    --name total_care_v9c
```

**Training Logs & Output Weights:**
- Checkpoints and training metrics (loss curves, mAP@0.5, PR curves) are automatically saved to `./runs/train/<run_name>/weights/best.pt`.

---

### Step 3: Run Inference & Hazard Warning Engine
Run inference on single images, folders of images, or dashcam video streams. The `src/detect.py` script automatically applies the **Hazard Alert Overlay Engine** on top of detected bounding boxes.

```bash
# Run inference on a single image file
python3 src/detect.py \
    --weights ./runs/train/total_care_v5s/weights/best.pt \
    --source ./test_image.jpg \
    --conf 0.3 \
    --save-dir ./runs/detect

# Run inference on a folder of test images
python3 src/detect.py \
    --weights ./runs/train/total_care_v5s/weights/best.pt \
    --source ./test_images/ \
    --conf 0.25

# Run inference on dashcam video file
python3 src/detect.py \
    --weights ./runs/train/total_care_v5s/weights/best.pt \
    --source ./dashcam_video.mp4 \
    --save-dir ./runs/detect_video
```

---

### Step 4: Export Model for Embedded Hardware (NVIDIA Jetson / ONNX Runtime)
To deploy the trained model onto autonomous driving edge hardware:

```bash
# Export PyTorch weights (.pt) to ONNX format
python3 src/export.py \
    --weights ./runs/train/total_care_v5s/weights/best.pt \
    --format onnx \
    --imgsz 416

# Export to TensorRT engine (for NVIDIA Jetson / Orin)
python3 src/export.py \
    --weights ./runs/train/total_care_v5s/weights/best.pt \
    --format engine \
    --imgsz 416
```

---

## 🚨 Hazard Alert Overlay Engine Matrix

When high-risk objects are detected, `src/detect.py` highlights the region and displays warning banners:

| Detected Class | Bounding Box Color | On-Screen Visual Alert | Priority Level |
| :--- | :--- | :--- | :--- |
| `pothole` | 🔴 Red | `[CRITICAL] ROAD HAZARD: POTHOLE DETECTED` | High |
| `no_belt` | 🔴 Red | `[SAFETY ALERT] UNBELTED OCCUPANT DETECTED` | High |
| `trafficLight-Red` | 🔴 Red | `[TRAFFIC CONTROL] RED LIGHT AHEAD` | High |
| `deer`, `roe_deer`, `elk`, `waterdeer` | 🟠 Orange | `[COLLISION RISK] WILDLIFE DETECTED` | High |
| `pedestrian`, `biker`, `scooter` | 🟡 Yellow | `[VRU CAUTION] PEDESTRIAN / TWO-WHEELER` | Medium |
| `car`, `trafficLight-Green`, `free_path` | 🟢 Green | Standard Detection Bounding Box | Normal |

---

## 🌐 Camera-Centric Temporal Ego-centric World Model (KITTI Tracking Extension)

Beyond single-frame detection, this repository includes a 7-stage **Temporal Ego-centric World Model** pipeline (`Temporal Perception → Ego-centric World Modeling → Future Prediction → Risk Assessment → Safety-aware Control`) designed for **Google Colab CLI** and **Kaggle CLI** execution on KITTI Tracking (`training/image_02`, `training/label_02`, `training/calib`).

### Extension Modules & Documentation
- [FINAL_REPORT.md](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/FINAL_REPORT.md) — **최종 프로젝트 결과 보고서** (기본 평가표 7개 항목 매핑, KITTI 8,008장 데이터셋, `v1→v2→v3` 학습·벤치마크, RQ1–RQ4 검증, 시연 영상 및 GitHub 링크)
- [ML_LAYER_SPEC.md](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/ML_LAYER_SPEC.md) — **머신러닝·딥러닝 모델 Layer별 기술설명서 및 참고자료** (`YOLO11m` 24개 모듈/231 layers, `MobileSAM` TinyViT+MaskDecoder, `ByteTrack`, `World Model & Safety Shield` 상세)
- [architecture.md](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/architecture.md) — 7-stage pipeline, KITTI `P2` to Ego-centric coordinate transform, `Past → Present → Future` state transitions, `Observed / Inferred / Occluded / Unknown` model
- [dataset_audit.md](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/dataset_audit.md) — KITTI Tracking integrity checks, `DontCare`/distractor masking, sequence-disjoint 14/4/3 split
- [experiment_plan.md](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/experiment_plan.md) — Hypotheses, baselines, ablations, evaluation metrics & GT availability
- [visualization_spec.md](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/visualization_spec.md) — Split-screen specification (`Left: Observed World` | `Right: Inferred Ego-centric World Model`) and 5-second staged demo timeline
- [src/kitti_prep.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/kitti_prep.py) — Archive/directory audit, sequence-disjoint YOLO export (`Car`, `Pedestrian`, `Cyclist`), and train-only rare-class oversampling (`train_weighted.txt`)
- [src/track_utils.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/track_utils.py) — ByteTrack multi-object tracking (`imgsz=960`), MobileSAM box-prompted instance masking (`trajectories_masks.json`, `mask_area_px`, `mask_fill_ratio`, `mask_bottom_x/y`), trajectory CSV/JSON, KITTI 18-column TrackEval export
- [src/world_model.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_model.py) — Calibrated `P2` inverse-variance depth fusion (flat-ground + class-height prior + SAM mask bottom contact), `KittiOxtsOdometry` (`Ego ↔ World ENU` + CTRV), World-frame 4D Kalman filter (`+0.5s` to `+2.0s`), occlusion cones, uncertainty separation, $C^1$ potential + `CollisionRiskSmoother` (`|dR/dt|` slew-rate clamping + hysteresis), and KITTI 3D evaluation
- [src/world_visualizer.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_visualizer.py) — Split-screen MP4 & keyframe renderer (Left Panel instance mask overlays + Right Panel BEV world model + smoothed Safety Shield HUD)
- [src/validation_experiments.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/validation_experiments.py) — Empirical validation suite for coordinate/ego-motion stability (RQ1), future prediction ablation (RQ2), uncertainty causality (RQ3), and Collision Detector gradient smoothing (RQ4)
- [kaggle/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/kaggle) / [kaggle_world_model/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/kaggle_world_model) / [colab/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/colab) — Remote GPU training, tracking & world-model runners (`kaggle kernels push`, Colab Drive `/content/drive/MyDrive/KITTI_Tracking`)
- [results/v3_bev_fixed/world_outputs/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v3_bev_fixed/world_outputs) & [logs/v3_bev_fixed/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/logs/v3_bev_fixed) — **Latest** Right-Panel BEV ellipse Jacobian projection, lateral-covariance corridor scoping, false-positive suppression, and `v2` MobileSAM + `yolo11m @ 960` quantitative outputs (`validation_0015.json`, `validation_0019.json`, rendered demos)
- [results/v2_improved/outputs/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v2_improved/outputs) & [logs/v2_improved/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/logs/v2_improved) — v2 improved perception/tracking archive (`yolo11m @ 960`, MobileSAM masking, depth fusion) preserved alongside [results/v1_detect_track/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v1_detect_track) and [results/world_model/](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/world_model)

### Quick Execution (Remote GPU via Kaggle CLI)
```bash
# Push & run KITTI audit -> sequence-split YOLO export -> GPU training -> tracking
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4

# Check execution status & download artifacts once complete
kaggle kernels status test013/totalcare-kitti-yolo-train-and-track
kaggle kernels output test013/totalcare-kitti-yolo-train-and-track -p results/kaggle_run
```

---

## 👥 Contributors & Contact

- **Project Lead:** Hyungwoo Lee (Contact: `010-3917-0557`)
- **Participants:**
  - Seungwon Lee ([@Heisnotanimposter](https://github.com/Heisnotanimposter))
  - Heejung Lim ([@heejjj](https://github.com/heejjj))

---
*Developed for Autonomous Driving Safety & Computer Vision Research.*
