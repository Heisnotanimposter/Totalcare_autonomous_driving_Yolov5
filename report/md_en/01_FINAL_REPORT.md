# Final Project Report: Camera-Centric Temporal Ego-centric World Model

**GitHub Repository:** https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5 *(retains the initial repository name while the internal codebase is built on the Ultralytics **YOLOv8 / YOLO11** pipeline)*  
**Dataset:** KITTI Object Tracking (`training/image_02`, `training/label_02`, `calib`, `oxts`) + `data_road` + deduped `kitti-stereo-image`  
**Execution Environment:** Kaggle CLI GPU (`2× Nvidia Tesla T4`, `Ultralytics 8.4.212`, `PyTorch 2.11.0+cu130`) + Google Colab CLI Integration

> **[Framework & Object Counting Note]** Following the course specification, this project first established a **`YOLOv8s` (`v1` Baseline)** and then scaled within the **same Ultralytics `train()` / `track()` pipeline** to **`YOLO11m` (`v2`)** and **`YOLO11l` (`v4`)** (`C3k2`, `C2PSA`). Per-frame active track counts (`Tracks: Observed=...` HUD, `trajectories.csv`, `road_context_stats.json`) also fulfill real-time object counting.

---

## 0. Instructor Evaluation Rubric Self-Check & Document Mapping

| Evaluation Criterion | Status | Evidence & Repository Location |
|---|:---:|---|
| **1. Data Collection (3+ classes, 100+ images)** | **Met** | `KITTI Tracking` 21 sequences (`0000`–`0020`), **`8,008` RGB images** (`14,371` train entries in `v4`), **`3` classes** (`Car`, `Pedestrian`, `Cyclist`). [`dataset_audit.md`](./dataset_audit.md) |
| **2. 5s+ Detection & Tracking Demo Video** | **Met** | Split-screen World Model demos (`0015` staged `8.0s` / full **`37.6s`**, `0019` staged `8.0s` / full **`105.9s`**) + 2D ByteTrack & SAM tracking videos. [`world_model_demo_0015.mp4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_full/world_model_demo_0015.mp4) |
| **3. Trained Model & Demo Videos Published on GitHub** | **Packaged (Push Pending)** | Trained weights `v2` [`best.pt`](./results/v2_improved/outputs/weights/best.pt) (`38.62 MB`), `v4` [`best.pt`](./results/v4_road_lane/detect_outputs/outputs/weights/best.pt) (`49 MB`), and demo MP4s kept within GitHub's 100MB limit via `.gitignore` (run final `git push` before submission). |
| **4. Contents Included in Final Report** | **Met** | This report (Sections 1–6) covers data curation, detection/segmentation/tracking, road/lane perception, and World Model safety evaluation. |
| **5. Training Process & Performance Benchmarks Included** | **Met** | Sections 2, 3, and 6 document `v1 → v2 → v3 → v4` hyperparameters, per-epoch convergence (`results.csv`), Detection/Tracking/Depth/Prediction/Road-Lane/RQ1–RQ4 benchmarks. |
| **6. GitHub Link & Description Included** | **Met** | Header and Section 4 specify repository URL, directory layout, reproduction kernels, and direct artifact paths. |
| **7. Per-Layer ML Technical Specification & References Attached** | **Met** | Full 24-module `YOLO11m` (`231 layers`), `MobileSAM`, `ByteTrack`, `SegFormer-B2`, and `World Model` layer specs + academic references in [`ML_LAYER_SPEC.md`](./ML_LAYER_SPEC.md). |

---

## 1. Data Collection & Sequence-Disjoint Preprocessing (`Stage 1`)

- **Raw Dataset Scale**: `KITTI Tracking` 21 sequences (`0000`–`0020`), **`8,008` frames** (`1242×375` and 3 adjacent resolutions, `10 Hz`, verified via `ZipFile.testzip()` / `PIL.Image.verify()` with `0` corrupt/missing files, [`kitti_step2_audit.json`](./results/v2_improved/outputs/kitti_step2_audit.json)).
- **Target Classes (3) & Sequence-Disjoint Split**: Strictly split by driving sequence so frames from the same sequence never leak across Train/Val/Test ([`kitti_export_manifest.json`](./results/v2_improved/outputs/kitti_export_manifest.json)).

| Split | Sequences | Frames | `Car` Boxes | `Pedestrian` Boxes | `Cyclist` Boxes | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Train** | 14 | `5,568` | `23,404` | `648` | `367` | Rare-class oversampling (`train_weighted.txt`: `6,974` entries) |
| **Val** | 4 | `832` | `2,075` | `1,777` | `754` | `0002, 0012, 0015, 0017` |
| **Test** | 3 | `1,608` | `1,818` | `9,044` | `817` | `0013, 0016, 0019` (completely unseen drives) |
| **Total** | **21** | **`8,008`** | **`27,297`** | **`11,469`** | **`1,938`** | `40,704` target objects + `DontCare`/Distractor masking |

- **Spurious Gradient Prevention Masking**: `DontCare` and adjacent distractor classes (`Van`, `Person`, `Person_sitting`) are masked with neutral gray (`114`) pixels during export to prevent false-negative penalties against TrackEval taxonomy.

---

## 2. Training Progression & Evolution (`v1 Baseline → v2 Improved → v3 BEV Fixed`)

### 2.1 Training Setup & Architecture Evolution
1. **`v1` Baseline (`YOLOv8s @ 640px`)**: `epochs=30`, `batch=16`, `129 layers / 11.14M params / 28.6 GFLOPs`.
2. **`v2` Improved (`YOLO11m @ 960px` + Rare-Class Oversampling + `MobileSAM`)**:
   - Upgraded input resolution from `640 → 960` with `C3k2 + C2PSA` spatial-attention `YOLO11m` (`231 layers / 20.06M params / 68.3 GFLOPs`).
   - Applied rare-class oversampling (`6,974` weighted train entries; Val/Test untouched), `cos_lr=True`, `mixup=0.1`, `close_mosaic=10`, `patience=12` on **`2× Tesla T4`** DDP (early-stopped at `Epoch 24`, Best = **`Epoch 12`**, ~108 min).
   - Coupled `mobile_sam.pt` (`38.8 MB`) Box-Prompt segmentation for pixel masks and ground-contact points (`mask_bottom_xy`).
3. **`v3` BEV Visual & False-Alarm Fix (`world_model.py`, `world_visualizer.py`)**:
   - Fixed >2x lateral distortion of longitudinal uncertainty ellipses in the BEV panel via Jacobian projection $J=\text{diag}(s_x,-s_y)$, lateral covariance decoupling, approach-speed gating (`approach_w`), and per-track EMA smoothing.

### 2.2 `v2 YOLO11m @ 960` Per-Epoch Convergence ([`results.csv`](./results/v2_improved/outputs/train_results/results.csv))

| Epoch | Elapsed (s) | `train/box_loss` | `train/cls_loss` | `train/dfl_loss` | Val `Precision` | Val `Recall` | Val `mAP@0.5` | Val `mAP@0.5:0.95` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 214.4 | 1.1077 | 0.8017 | 1.0848 | 0.4067 | 0.4114 | 0.4074 | 0.2240 |
| **5** | 1097.3 | 0.9687 | 0.6330 | 1.0241 | 0.5063 | 0.4627 | 0.4516 | 0.2611 |
| **10** | 2197.2 | 0.8181 | 0.5107 | 0.9597 | 0.6701 | 0.4487 | 0.4886 | 0.2894 |
| **12 (Best)** | 2637.2 | 0.7793 | 0.4813 | 0.9431 | **0.7166** | **0.4814** | **0.5231** | **0.3059** |
| **19** | 4179.2 | 0.6791 | 0.4135 | 0.9019 | 0.7905 | 0.4846 | 0.5297 | 0.2985 |
| **24 (Stop)** | 5278.9 | 0.6194 | 0.3725 | 0.8823 | 0.6714 | 0.4341 | 0.4539 | 0.2738 |

- Training curves & normalized confusion matrix: [`results.png`](./results/v2_improved/outputs/train_results/results.png), [`confusion_matrix_normalized.png`](./results/v2_improved/outputs/train_results/confusion_matrix_normalized.png)

---

## 3. Quantitative Performance Benchmarks

### 3.1 Stage 2 Object Detection (`v1` vs `v2`, [`eval_metrics.json`](./results/v2_improved/outputs/eval_metrics.json))

| Split / Metric | `v1` (`YOLOv8s @ 640`) | `v2` (`YOLO11m @ 960`) | Improvement (`Δ`) |
|---|---:|---:|---:|
| **Val `mAP@0.5` / `mAP@0.5:0.95`** | `0.4521 / 0.2471` | **`0.5235 / 0.3060`** | **`+0.0714 (+15.8%)`** / `+0.0589` |
| **Val `Precision / Recall`** | `0.6026 / 0.4230` | **`0.7170 / 0.4822`** | `+0.1144` / `+0.0592` |
| **Val AP50 (`Car / Ped / Cyc`)** | `0.7501 / 0.5141 / 0.0922` | **`0.8235 / 0.5871 / 0.1598`** | `+0.0734 / +0.0730 / +0.0676 (+73.3%)` |
| **Test `mAP@0.5` / `mAP@0.5:0.95`** | `0.3652 / 0.1695` | **`0.4269 / 0.2145`** | **`+0.0617 (+16.9%)`** / `+0.0450` |
| **Test AP50 (`Car / Ped / Cyc`)** | `0.6294 / 0.3497 / 0.1165` | **`0.6662 / 0.3631 / 0.2513`** | `+0.0368 / +0.0134 /` **`+0.1348 (+115.7%)`** |

### 3.2 Stage 3 Tracking & Stage 4–7 World Model (RQ1–RQ4) ([`validation_0015.json`](./results/v3_bev_fixed/world_outputs/validation_0015.json), [`validation_0019.json`](./results/v3_bev_fixed/world_outputs/validation_0019.json))

- **MobileSAM Masking**: Seq `0015` (376f) **`1,370`** instances (`41` tracks); Seq `0019` (1059f) **`3,788`** instances (`203` tracks) ([`tracking_summary.json`](./results/v2_improved/outputs/tracking_summary.json)).
- **Research Questions (RQ1–RQ4) Validation Summary (`v3_bev_fixed`)**:

| Validation Item | Sequence `0015` (376 frames, `37.6s`) | Sequence `0019` (1059 frames, `105.9s`) |
|---|---|---|
| **RQ1 Coordinate & Ego Comp (`Ego↔World` Error / Stationary World Speed)** | `2.161e-14 m` / **`0.227 m/s`** (uncomp `1.358 m/s`) | `3.352e-14 m` / **`0.756 m/s`** (uncomp `1.849 m/s`) |
| **RQ2 Future Prediction (`OXTS+CTRV` ADE/FDE vs Zero-Vel)** | **`3.6052 / 4.6872 m`** (vs Zero-Vel `5.2465 / 7.1198 m`) | **`2.7890 / 3.2453 m`** (vs Zero-Vel `3.2491 / 4.2215 m`) |
| **RQ2 2D Covariance `1σ` Coverage (theory `0.3935`)** | **`0.3939`** (diff `0.0004`) | **`0.4863`** |
| **RQ3 Uncertainty Causality (`Arm A Blind → B Nominal → C Degraded`)** | **`0.2220 → 0.2791 → 0.2792`** (`77 / 18` escalated) | **`0.5242 → 0.6290 → 0.6396`** (`521 / 121` escalated) |
| **RQ4 Gradient Smoothing (`max \|dR/dt\|` / Spikes / Flips)** | `5.29 → 1.10 /s (-79.2%)`, `>1.5/s` **`5 → 0 (-100%)`**, flips `34 → 22` | `5.76 → 1.10 /s (-80.9%)`, `>1.5/s` **`42 → 0 (-100%)`**, flips `140 → 78` |
| **`v2 → v3` False-Alarm Suppression (`Arm B` Mean Risk / Emergency)** | `0.6498 → 0.2791` (**`-57.0%`**), Emergency `104 → 16` (**`-84.6%`**) | `0.9053 → 0.6290` (**`-30.5%`**), Emergency `849 → 216` (**`-74.6%`**) |

### 3.3 Demo Videos & Keyframes
- `Left Panel`: Raw camera video + 2D BBoxes + `Car`/`Pedestrian`/`Cyclist` labels + Track ID + Confidence + **MobileSAM instance polygon masks**.
- `Right Panel`: Ego-centric BEV (`X` lateral / `Y` forward), Ego planned path (CTRV), fading past trails, future forecast (`+0.5s~+2.0s`), Jacobian-corrected `1-sigma` covariance ellipses, occlusion cones & rear Unknown hatching, compact `CONFLICT` rings, smoothed Safety Shield banner.
- Representative keyframes: [`seq_0015_frame_000045_stage5.png`](./results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png), [`seq_0015_frame_000055_stage5.png`](./results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000055_stage5.png), [`seq_0019_frame_000055_stage5.png`](./results/v3_bev_fixed/world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png)

---

## 4. GitHub Repository & Key Artifacts

- **Official GitHub Repository:** https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5
- **Core Source Code**: [`src/kitti_prep.py`](./src/kitti_prep.py), [`src/track_utils.py`](./src/track_utils.py), [`src/world_model.py`](./src/world_model.py), [`src/world_visualizer.py`](./src/world_visualizer.py), [`src/validation_experiments.py`](./src/validation_experiments.py), [`src/road_lane.py`](./src/road_lane.py)
- **Trained Weights**: `v2` [`best.pt`](./results/v2_improved/outputs/weights/best.pt) (`YOLO11m @ 960`, `38.62 MB`), `v4` [`best.pt`](./results/v4_road_lane/detect_outputs/outputs/weights/best.pt) (`YOLO11l @ 1280`, `49 MB`)
- **Demo Videos (5s+)**:
  - Val Seq `0015` (`v3` 8s staged / 37.6s full): [`staged MP4`](./results/v3_bev_fixed/world_outputs/world_model_0015_staged/world_model_demo_0015.mp4) / [`full MP4`](./results/v3_bev_fixed/world_outputs/world_model_0015_full/world_model_demo_0015.mp4)
  - Test Seq `0019` (`v3` 8s staged / 105.9s full): [`staged MP4`](./results/v3_bev_fixed/world_outputs/world_model_0019_staged/world_model_demo_0019.mp4) / [`full MP4`](./results/v3_bev_fixed/world_outputs/world_model_0019_full/world_model_demo_0019.mp4)
- **Remote GPU Reproduction Kernels**: [`kaggle/train_track_kitti.py`](./kaggle/train_track_kitti.py), [`kaggle_world_model/run_world_model_kaggle.py`](./kaggle_world_model/run_world_model_kaggle.py), [`kaggle_v4_road/train_road_lane_kitti.py`](./kaggle_v4_road/train_road_lane_kitti.py), [`kaggle_v4_detect/train_v4_detect_kitti.py`](./kaggle_v4_detect/train_v4_detect_kitti.py), [`kaggle_v4_world_model/run_v4_world_model_kaggle.py`](./kaggle_v4_world_model/run_v4_world_model_kaggle.py)

---

## 5. Per-Layer ML Technical Specification & References (Summary)

Full deep-learning, Kalman, and control layer details plus academic references are provided in [`ML_LAYER_SPEC.md`](./ML_LAYER_SPEC.md).

1. **`YOLO11m` (`231 layers`, `20.06M params`, `68.3 GFLOPs`)**: Backbone (`Conv` Stem → `C3k2` CSP Bottleneck → `SPPF` → `C2PSA` Spatial Attention), Neck (`PAN-FPN`), Head (`Decoupled Anchor-Free Detect` + `DFL` + `CIoU` + `BCE`).
2. **`MobileSAM` (`38.8 MB`)**: Knowledge-Distilled `TinyViT` Image Encoder (`MBConv` + `Windowed Self-Attention`), Box Prompt Positional Encoder, Two-Way Cross-Attention Mask Decoder + `4×` Transposed Conv.
3. **`ByteTrack`**: 8D Constant-Velocity BBox Kalman Filter + Two-Stage Hungarian Bipartite Matching (`High-Conf` + `Low-Conf` occlusion recovery).
4. **Temporal World Model & Safety Shield**: `P2` inverse-variance depth fusion + OXTS `SE(2)` compensation + 4D World-ENU Kalman filter + Jacobian BEV covariance ellipse projection + $C^1$ hazard potential & `CollisionRiskSmoother` slew-rate/hysteresis smoothing.
5. **`SegFormer-B2` (`nvidia/segformer-b2-finetuned-cityscapes-1024-1024`, added in v4)**: Hierarchical MiT-B2 Mix-Transformer encoder + All-MLP decoder (`Non-Road` / `Other Road` / `Ego-Lane` 3-class head) + BEV IPM warp & HLS+Sobel quadratic polynomial EMA lane tracker.

---

## 6. Version 4 Upgrade: Road/Non-Road & Lane Segmentation + `0.9` Accuracy (`v4_road_lane`)

Following the `v3_bev_fixed` release, three dedicated Kaggle kernels (`test013/totalcare-kitti-v4-road-lane`, `test013/totalcare-kitti-v4-detect`, `test013/totalcare-kitti-v4-world-model`) were executed to add **Road vs Non-Road classification & lane masking** and boost detection accuracy from **`~0.7` toward `~0.9`**, archived under [`results/v4_road_lane/`](./results/v4_road_lane/).

### 6.1 Road vs Non-Road & Ego-Lane Segmentation (`SegFormer-B2` + BEV IPM)
- **Dataset & Training**: Fine-tuned `SegFormer-B2` with a 3-class head (`Non-Road`, `Other Road`, `Ego-Lane`) on `tryingit/roadlane-detection-evaluation-2013` (KITTI `data_road`, `230 train / 59 val` stratified split, 25 epochs).
- **Measured Performance ([`road_eval_metrics.json`](./results/v4_road_lane/road_outputs/road_lane_outputs/road_eval_metrics.json))**:
  - **Road vs Non-Road**: **ROC AUC = `0.9992`**, **Pixel Acc = `0.9910` (`99.10%`)**, **Road F1 = `0.9747`** (`Precision 0.9690` / `Recall 0.9804`), Road IoU `0.9506`, mIoU `0.9698`.
  - **Ego-Lane (`um_lane`)**: **F1 = `0.9119`** (`Precision 0.9070` / `Recall 0.9169`), IoU `0.8381`.
- **Visualization & World Model Coupling**: Renders translucent Road (green), Non-Road (rose-plum), and Ego-Lane (cyan-teal) masks, left/right quadratic lane polylines, and per-object `[LANE]`/`[RD]`/`[NON-RD]` badges on the Left Panel, while projecting drivable road polygons, curvilinear BEV lanes, and non-road (`0.65` risk attenuation) awareness into the Right Panel Safety Shield.

### 6.2 Detector Accuracy Upgrade (`YOLO11l @ 1280` + Deduped Merge + Multi-Scale TTA)
- **Leak-Free Dataset Expansion**: Merged `leducnhuan/kitti-tracking` with `harshpdani/kitti-stereo-image` (`7,481` images scanned; `595` MD5/dHash matches against Val/Test `0002,0012,0015,0017,0013,0016,0019` strictly excluded → `6,886` unique images added, `14,371` total train entries) with `Van`/`Person_sitting` 114 gray masking.
- **Quantitative Comparison (`v2 → v4`, [`eval_metrics_v4.json`](./results/v4_road_lane/detect_outputs/outputs/eval_metrics_v4.json))**:
  - **Val (`All` difficulty)**: `v2 mAP@0.5 = 0.5235` → `v4 std = 0.6889` (`Precision 0.8685`) → **`v4 TTA = 0.7400`** (`Car 0.8475 / Ped 0.7712 / Cyc 0.6014`).
  - **Test (`All` difficulty)**: `v2 mAP@0.5 = 0.4269` → **`v4 TTA = 0.6705`** (`Precision 0.7984`, `Car 0.7804 / Ped 0.6441 / Cyc 0.5870`).
  - **KITTI Standard Difficulty Breakdown (TTA)**: Val **Easy `mAP@0.5 = 0.8677`** (`Precision 0.9750`), Test **Easy `mAP@0.5 = 0.8966`** (`~0.90`, `Car 0.9792 / Ped 0.9206 / Cyc 0.7899`), Test **Moderate `mAP@0.5 = 0.8404`**, Test **Hard `mAP@0.5 = 0.8025`**, Test **KITTI-All `mAP@0.5 = 0.7965`** (`mean_precision 0.8531`).
- **v4 World Model Integration & RQ1–RQ4 Verification**:
  - Val `0015` ([`full MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_full/world_model_demo_0015.mp4) / [`staged MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_staged/world_model_demo_0015.mp4)): `ADE 3.5836 m`, RQ3 `0.2277 < 0.2838 < 0.2916`, RQ4 spikes `8 → 0 (-100%)`.
  - Test `0019` ([`full MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0019_full/world_model_demo_0019.mp4) / [`staged MP4`](./results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0019_staged/world_model_demo_0019.mp4)): `ADE 2.0437 m`, RQ3 `0.5421 < 0.6261 < 0.6454`, RQ4 spikes `16 → 0 (-100%)`, flips `124 → 53 (-57.26%)`.

### 6.3 `v4` Training Dynamics, PR/F1/Confusion Matrix Analysis & Applied ML Techniques

Academic quantitative interpretation of the 12 training and evaluation artifacts in [`results/v4_road_lane/detect_outputs/outputs/train_results/`](./results/v4_road_lane/detect_outputs/outputs/train_results):

1. **Loss Convergence & `Close-Mosaic` Phase Transition (`results.csv`, `results.png`)**:
   - `train/box_loss` (`1.12833 → 0.53388`), `train/cls_loss` (`0.86791 → 0.29695`), and `train/dfl_loss` (`1.19982 → 0.89416`) decrease monotonically across 24 epochs. At **`Epoch 19`**, `close_mosaic=7` disables 4-tile Mosaic augmentation to align training with natural KITTI image statistics, producing a sharp step-drop in `box_loss` (`0.6653 → 0.60687`) and `cls_loss` (`0.4043 → 0.34428`).
   - `val/box_loss` decreases steadily (`1.60946 → 1.29238`) without localization overfitting; `Epoch 14` peaks in `Precision = 0.92008` and `mAP@0.5 = 0.71385`, while **`Epoch 15`** achieves peak `mAP@0.5:0.95 = 0.4236` and `Recall = 0.59069` (`best.pt`).

2. **Precision–Recall & F1–Confidence Curves (Val vs Held-Out Test)**:
   - **Validation (`BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`)**: Under TTA, `all classes mAP@0.5 = 0.740` (`Car 0.848 / Pedestrian 0.771 / Cyclist 0.601`), peak F1 **`0.71 at conf = 0.217`**, `Precision 1.00 at conf = 0.872`, max `Recall 0.84 at conf = 0.000`. `Car` sustains an F1 plateau near `0.84` across `conf 0.25–0.70`.
   - **Held-Out Test (`test_BoxPR_curve.png`, `test_BoxF1_curve.png`, `test_BoxP_curve.png`, `test_BoxR_curve.png`)**: On completely unseen sequences (`0013, 0016, 0019`), `all classes mAP@0.5 = 0.670` (`Car 0.780 / Pedestrian 0.644 / Cyclist 0.587`), peak F1 **`0.68 at conf = 0.568`**, `Precision 1.00 at conf = 0.941`, max `Recall 0.83 at conf = 0.000`. Notably, `Cyclist` AP50 drops only `-0.014` (`0.601 → 0.587`), and the high optimal Test confidence (`0.568`) confirms strong probability calibration.

3. **Error Decomposition via Normalized Confusion Matrix (`confusion_matrix_normalized.png`)**:
   - Diagonal true-positive rates reach `Car 0.90`, `Pedestrian 0.87`, and `Cyclist 0.55`, with low background miss rates (`Car 0.10`, `Pedestrian 0.08`, `Cyclist 0.16`).
   - The dominant `Cyclist` error mode is semantic confusion with `Pedestrian` (`0.26` of `Cyclist` GT predicted as `Pedestrian`) due to rider appearance overlap, alongside background false positives (`Car 0.40`, `Pedestrian 0.42`, `Cyclist 0.18`) from unlabeled distant/distractor instances at low confidence.

4. **Machine Learning Techniques Leveraged for Metric Improvement**:
   - **Leak-Free Hash-Deduped Dataset Expansion**: Strictly excluded `595` MD5/dHash matches against Val/Test from `kitti-stereo-image` (`7,481` images) and merged `6,886` unique images (`14,371` total train entries).
   - **Capacity & Resolution Scaling (`YOLO11l @ 1280px`)**: Combines `C3k2` CSP bottlenecks, `SPPF`, and `C2PSA` spatial attention with `1280px` input (doubling P3/8 small-object feature resolution).
   - **DFL + CIoU + Task-Aligned Assigner + Cosine LR / Close-Mosaic**: Distributional box-edge regression (`DFL, reg_max=16`), dynamic alignment-based label assignment (`TAL`), 3-epoch warmup with cosine decay (`lrf=0.01`), and `close_mosaic=7`.
   - **Imbalance & Gradient Hygiene (`Rare-Class Oversampling` + `114 Gray Masking`)**: Weighted sampling for `Cyclist`/`Pedestrian` and neutral gray (`114`) masking over `DontCare`/`Van`/`Person_sitting` to eliminate spurious false-negative gradients.
   - **Multi-Scale Test-Time Augmentation (`augment=True`)**: Flip and multi-scale inference ensembling boosts Val `mAP@0.5` from `0.6889 → 0.7400` (`+0.0511`).

---

## 7. Conclusions, Technical Challenges, and Reflections

Key engineering lessons, debugging experiences, and limitations encountered while iterating from `v1 → v2 → v3 → v4`:

1. **Guarding Against Temporal Data Leakage via Sequence-Disjoint Splits**:
   - Random frame-level splits inflate `mAP` on `10 Hz` driving video because adjacent frames are nearly identical. Enforcing a strict `14 / 4 / 3` sequence-disjoint split—and excluding `595` MD5/dHash-matched frames when merging `kitti-stereo-image` (`7,481` images)—kept Held-out Test (`0013, 0016, 0019`) evaluation scientifically valid.

2. **Debugging the `testing/image_02` vs `training/image_02` Path Mismatch in `v4`**:
   - Visual inspection of early `v4` outputs revealed misaligned road/lane masks against the camera frames. Root-cause analysis showed `rglob('image_02')` matched `testing/image_02` (`0015`: `701`f, `0019`: `404`f) instead of `training/image_02` (`0015`: `376`f, `0019`: `1,059`f). Adding an explicit `'training' in path.parts` guard and re-running fixed the 1:1 alignment, underscoring that visual sanity checks are indispensable alongside scalar metrics.

3. **Mitigating `Cyclist` vs `Pedestrian` Confusion and `2× T4` DDP Tuning**:
   - The normalized confusion matrix revealed `0.26` of `Cyclist` ground truths were misclassified as `Pedestrian`. Combining `Van`/`Person_sitting` `114`-gray masking, train-only rare-class oversampling, `1280px` scaling, `close_mosaic=7`, and TTA raised Test Easy `mAP@0.5` to `0.8966` (`Cyclist Test Easy 0.7899`).

4. **Resolving BEV Uncertainty Distortion & False-Alarm Spikes**:
   - Early `v2` BEV plots stretched longitudinal depth uncertainty laterally across adjacent lanes, triggering false `Emergency Intervention` alerts on sidewalk pedestrians. Jacobian projection ($J=\text{diag}(s_x,-s_y)$), approach-speed gating (`approach_w`), and `CollisionRiskSmoother` (`1.10/s` slew-rate clamp + `0.04` hysteresis) eliminated `>1.5/s` spikes by `100%` and cut `0015` false-alarm risk by `-57.0%`.

5. **Limitations & Future Work**:
   - Monocular flat-ground projection (`h_cam = 1.65m`) remains sensitive to road pitch variations at long range, and KITTI Tracking lacks closed-loop vehicle dynamics, restricting Safety Shield validation to offline advisory evaluation. Future work will fuse stereo/LiDAR depth and validate closed-loop control inside CARLA.
