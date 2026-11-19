# Per-Layer Machine Learning & Deep Learning Technical Specification and References

**Repository:** [Totalcare_autonomous_driving_Yolov5](https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5)  
**Target Pipeline:** `Stage 2 (YOLO11m/l + MobileSAM + SegFormer-B2) → Stage 3 (ByteTrack) → Stage 4–7 (Temporal Ego-centric World Model & Safety Shield)`

---

## 1. Object Detector: `YOLO11m` (`results/v2_improved/outputs/weights/best.pt`) & `YOLO11l` (`v4`)

- **Model Specification (Measured from Kaggle `2x Tesla T4` Training Logs)**:
  - **Architecture Scale (`YOLO11m`)**: `231 layers`, `20,055,321 parameters`, `68.3 GFLOPs` (fused inference: `125 layers`, `20,032,345 parameters`, `67.8 GFLOPs`).
  - **Input Resolution**: `960 × 960` RGB in `v2` (`1280 × 1280` in `v4` `YOLO11l`, `AMP FP16` training, `nc = 3`: `Car`, `Pedestrian`, `Cyclist`).
  - **Improvement over Baseline (`v1 YOLOv8s`)**: Upgrades `YOLOv8s` (`129 layers`, `11.14M params`, `28.6 GFLOPs`, `640px`) via `C2f → C3k2` backbone efficiency, `C2PSA` Spatial Attention, and high-resolution input (`960px`/`1280px`) to preserve distant/small pedestrian and cyclist features.

### 1.1 `YOLO11m` Full 24-Module Layer-by-Layer Breakdown (`Layer 0` – `Layer 23`)

Exact network graph printed by Ultralytics `8.4.212` (`logs/v2_improved/totalcare-kitti-yolo-train-and-track.log`):

| Layer | Input (`from`) | Module (`Ultralytics`) | Output Channels / Args | Parameters | Stage | Technical Role |
|---:|---|---|---|---:|---|---|
| **0** | `-1` | `nn.modules.conv.Conv` | `[3, 64, 3, 2]` | `1,856` | Backbone (Stem) | `3×3` Conv (`stride=2`) + BatchNorm + SiLU for `960→480` downsampling (P1/2) and low-level edge/texture extraction |
| **1** | `-1` | `nn.modules.conv.Conv` | `[64, 128, 3, 2]` | `73,984` | Backbone (Stage 1) | `3×3` Conv (`stride=2`), `480→240` reduction (P2/4) and channel expansion |
| **2** | `-1` | `nn.modules.block.C3k2` | `[128, 256, 1, True, 0.25]` | `111,872` | Backbone (Stage 1) | Cross-Stage Partial Bottleneck (`c3k=False`, `e=0.25`) reducing compute via split channels and skip fusion |
| **3** | `-1` | `nn.modules.conv.Conv` | `[256, 256, 3, 2]` | `590,336` | Backbone (Stage 2) | `3×3` Conv (`stride=2`), `240→120` reduction (P3/8 — small-object feature map anchor) |
| **4** | `-1` | `nn.modules.block.C3k2` | `[256, 512, 1, True, 0.25]` | `444,928` | Backbone (Stage 2) | P3/8 refinement preserving contour cues for distant `Pedestrian`/`Cyclist` |
| **5** | `-1` | `nn.modules.conv.Conv` | `[512, 512, 3, 2]` | `2,360,320` | Backbone (Stage 3) | `3×3` Conv (`stride=2`), `120→60` reduction (P4/16 — medium-object feature map) |
| **6** | `-1` | `nn.modules.block.C3k2` | `[512, 512, 1, True]` | `1,380,352` | Backbone (Stage 3) | Deep `C3k` block (`c3k=True`) abstracting mid-range vehicles and cyclists |
| **7** | `-1` | `nn.modules.conv.Conv` | `[512, 512, 3, 2]` | `2,360,320` | Backbone (Stage 4) | `3×3` Conv (`stride=2`), `60→30` reduction (P5/32 — large near vehicles & global context) |
| **8** | `-1` | `nn.modules.block.C3k2` | `[512, 512, 1, True]` | `1,380,352` | Backbone (Stage 4) | High-level semantic feature extraction |
| **9** | `-1` | `nn.modules.block.SPPF` | `[512, 512, 5]` | `656,896` | Backbone (Context) | Spatial Pyramid Pooling - Fast: 3 sequential `5×5` MaxPool layers aggregating multi-scale receptive fields |
| **10** | `-1` | `nn.modules.block.C2PSA` | `[512, 512, 1]` | `990,976` | Backbone (Attention) | Cross-Stage Partial **Pyramid Squeeze / Spatial Attention**: Multi-head Self-Attention focusing on occluded lane objects |
| **11** | `-1` | `nn.Upsample` | `[None, 2, 'nearest']` | `0` | Neck (Top-Down FPN) | 2x nearest-neighbor upsampling from P5 (`30×30`) to P4 (`60×60`) |
| **12** | `[-1, 6]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Top-Down FPN) | Channel concatenation of Layer 11 (semantic) and Layer 6 (P4 spatial) (`512+512→1024`) |
| **13** | `-1` | `nn.modules.block.C3k2` | `[1024, 512, 1, True]` | `1,642,496` | Neck (Top-Down FPN) | Fused P4 feature refinement |
| **14** | `-1` | `nn.Upsample` | `[None, 2, 'nearest']` | `0` | Neck (Top-Down FPN) | 2x upsampling from P4 (`60×60`) to P3 (`120×120`) |
| **15** | `[-1, 4]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Top-Down FPN) | Concatenation of Layer 14 and Layer 4 (P3 high-res) (`512+512→1024`) |
| **16** | `-1` | `nn.modules.block.C3k2` | `[1024, 256, 1, True]` | `542,720` | Neck → **P3/8 Out** | Final small-object (`Pedestrian`, distant `Cyclist`) detection map (`256ch`) |
| **17** | `-1` | `nn.modules.conv.Conv` | `[256, 256, 3, 2]` | `590,336` | Neck (Bottom-Up PAN) | Path Aggregation bottom-up path: P3 (`120×120`) → `60×60` downsampling |
| **18** | `[-1, 13]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Bottom-Up PAN) | Combines localization (Layer 17) with semantics (Layer 13) (`256+512→768`) |
| **19** | `-1` | `nn.modules.block.C3k2` | `[768, 512, 1, True]` | `1,511,424` | Neck → **P4/16 Out** | Final medium-object detection map (`512ch`) |
| **20** | `-1` | `nn.modules.conv.Conv` | `[512, 512, 3, 2]` | `2,360,320` | Neck (Bottom-Up PAN) | P4 (`60×60`) → `30×30` downsampling |
| **21** | `[-1, 10]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Bottom-Up PAN) | Fuses bottom-up P4 with Backbone C2PSA output (Layer 10) (`512+512→1024`) |
| **22** | `-1` | `nn.modules.block.C3k2` | `[1024, 512, 1, True]` | `1,642,496` | Neck → **P5/32 Out** | Final large/near vehicle (`Car`) detection map (`512ch`) |
| **23** | `[16, 19, 22]` | `nn.modules.head.Detect` | `[nc=3, reg_max=16, [256,512,512]]` | `1,413,337` | **Head (Decoupled)** | Anchor-Free decoupled head: regression branch (`4×16=64` DFL) + classification branch (`nc=3` Depthwise Separable Conv) |

### 1.2 Loss Functions & Optimization Layers
- **Box Regression (`box=7.5`, `dfl=1.5`)**: **CIoU Loss** (center distance, aspect ratio, and overlap area) + **Distribution Focal Loss (DFL, `reg_max=16`)** predicting discrete probability distributions over box edges via fixed projection `model.23.dfl.conv.weight`.
- **Classification (`cls=0.5`)**: Per-class **Binary Cross-Entropy (BCE with Logits)** + **Task-Aligned Assigner (TAL)** dynamic positive/negative anchor assignment.
- **Cosine LR + MixUp + Close-Mosaic**: 3-epoch linear warmup (`warmup_bias_lr=0.1`), Cosine Annealing (`lrf=0.01`), and `close_mosaic=10` for clean natural-image convergence.

---

## 2. Instance Segmentation: `MobileSAM` (`src/track_utils.py`)

Accepts per-track 2D Bounding Boxes from ByteTrack as **Box Prompts** to produce pixel-level foreground masks, providing ground-contact anchors (`mask_bottom_xy`) and occupancy ratios (`mask_fill_ratio`) for Stage 4 monocular depth fusion.

| Module (MobileSAM) | Internal Layer Mechanism | I/O Specification | Role in Pipeline |
|---|---|---|---|
| **1. Image Encoder (`TinyViT`)** | `PatchEmbed` (Conv×2) → **Stage 1** (`MBConv` local channel-spatial) → **Stage 2–4** (`Windowed Multi-Head Self-Attention` + `MLP` + Patch Merging) → Neck (`1×1` + `3×3` Conv) | `BGR` image (`1242×375`) → `64×64×256` Feature Embedding | Lightweight `TinyViT` backbone (`38.8 MB`) via decoupled distillation from ViT-H |
| **2. Prompt Encoder** | **Random Fourier Positional Encoding** + learned corner-type embeddings on Box Prompt `[x1, y1, x2, y2]` | Bounding Box coords → `(N, 2, 256)` Sparse Prompt Tokens | Encodes detector boxes as spatial prompts to suppress background clutter |
| **3. Two-Way Transformer Mask Decoder** | **Cross-Attention** (`Prompt→Image`, `Image→Prompt`) 2 blocks + MLP → **`ConvTranspose2d` (`4×` Upsampling)** → Hypernetwork MLP Inner Product | Sparse Tokens + `64×64×256` → Binary Mask + Polygon Contour | `cv2.findContours` extracts polygons (`trajectories_masks.json`), `mask_fill_ratio`, and `mask_bottom_x/y` |

---

## 3. Multi-Object Tracking Layer: `ByteTrack` (`src/track_utils.py`)

- **State Estimation Filter (Linear Kalman Filter)**: 8D constant-velocity image-plane state $\mathbf{x} = [c_x, c_y, a, h, \dot{c}_x, \dot{c}_y, \dot{a}, \dot{h}]^T$ (center, aspect ratio, height, and velocities).
- **Two-Stage Hungarian Association**:
  1. **High-Confidence Matching (`conf >= 0.5`)**: Applies `linear_sum_assignment` on the IoU cost matrix between active tracks and high-confidence boxes.
  2. **Low-Confidence Recovery (`0.1 <= conf < 0.5`)**: Re-associates lower-confidence detections caused by partial occlusion or backlight to unmatched tracks, reducing ID switches.
- **Sequence Isolation Audit (`compute_track_trajectories`)**: Resets tracker state at sequence boundaries, indexes by composite `(sequence_id, track_id)`, and logs any frame-order, duplicate, or class-flip anomalies.

---

## 4. Road & Lane Segmentation Layer: `SegFormer-B2` + BEV IPM (`src/road_lane.py`, `v4`)

- **Semantic Segmentation (`nvidia/segformer-b2-finetuned-cityscapes-1024-1024`)**: Hierarchical **Mix-Transformer (MiT-B2)** encoder with Overlapped Patch Merging and Efficient Self-Attention + lightweight **All-MLP** decoder fine-tuned on KITTI `data_road` with a 3-class head (`Non-Road`, `Other Road`, `Ego-Lane`).
- **Curvilinear Lane Extraction (`TemporalRoadLaneProcessor`)**: Inverse Perspective Mapping (IPM) to BEV + HLS/Sobel lane-marking filtering + Ego-Lane boundary fallback + quadratic polynomial ($x = a y^2 + b y + c$) EMA temporal smoothing.

---

## 5. Temporal Ego-centric World Model & Safety Shield Layer (`src/world_model.py`, `src/world_visualizer.py`)

Differentiable Kalman-based backend unifying perception outputs into metric physics and temporal continuity:

1. **P2 Inverse-Variance Depth Fusion Layer (`KittiCalibration.ground_plane_bev_fusion`)**:
   - Combines flat-ground projection ($Y_{\text{ground}} = f_y h_{\text{cam}} / (v - c_y)$) anchored at `mask_bottom_xy` (`mask_fill_ratio >= 0.25`) with pitch-invariant class-height depth ($Y_{\text{height}} = f_y H_c / h_{\text{px}}$) via $Y_{\text{fused}} = \frac{w_g Y_g + w_h Y_h}{w_g + w_h}$.
2. **OXTS `SE(2)` Ego-Motion Compensation & CTRV Layer (`EgoPose2D`, `KittiOxtsOdometry`)**:
   - Mercator-projected `World-ENU` transform removes Ego translation and yaw ($\psi, v_f, w_z$) to eliminate false velocities on stationary objects (`round-trip error < 4e-14 m`).
3. **4D World-ENU Linear Kalman Filter (`ObjectWorldTrack`)**:
   - State $\mathbf{x}_w = [X_w, Y_w, v_{X,w}, v_{Y,w}]^T$, white-noise acceleration process noise ($\sigma_a = 1.1\text{ m/s}^2$), rotated measurement covariance $R_w$ scaled inversely by confidence and mask fill ratio, projected to BEV pixels via Jacobian $J=\text{diag}(s_x,-s_y)$.
4. **$C^1$ Continuous Hazard Potential + `CollisionRiskSmoother` Layer**:
   - Lateral-covariance (`lateral_uncertainty_m`) corridor intrusion, approach-speed gating (`approach_w`), non-road attenuation (`0.65` in `v4`), calibrated TTC, and probabilistic soft-union $1 - \prod_i (1 - r_i)$.
   - Asymmetric EMA (`alpha_up=0.36`, `alpha_down=0.22`), slew-rate clamping (`1.10/s`, `-0.85/s`), per-track EMA (`per_track_ema`), low-TTC debouncing (`low_ttc_streak`), and hysteresis (`0.04`) eliminate `>1.5/s` spikes by `100%`.

---

## 6. References

1. **Ultralytics YOLO11 / YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023–2025). *Ultralytics YOLO (`YOLO11`, `C3k2`, `C2PSA`, `YOLOv8`)* — https://docs.ultralytics.com/models/yolo11/
2. **SegFormer**: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers* (`NeurIPS 2021`, `arXiv:2105.15203`). — https://arxiv.org/abs/2105.15203
3. **MobileSAM**: Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S.-H., Lee, S., & Hong, C. S. (2023). *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications* (`arXiv:2306.14289`). — https://arxiv.org/abs/2306.14289
4. **Segment Anything (SAM)**: Kirillov, A., Mintun, E., Ravi, N., et al. (2023). *Segment Anything* (`ICCV 2023`, `arXiv:2304.02643`). — https://arxiv.org/abs/2304.02643
5. **ByteTrack**: Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (`ECCV 2022`). — https://arxiv.org/abs/2110.06864
6. **KITTI Object Tracking & Road Benchmark**: Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite* (`CVPR 2012`). — https://www.cvlibs.net/datasets/kitti/eval_tracking.php
7. **Kalman Filter & CTRV Motion Modeling**: Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*; Schubert, R., et al. (2008). *Comparison and Evaluation of Advanced Motion Models for Vehicle Tracking* (`FUSION 2008`).
8. **DFL & Task-Aligned Assigner**: Li, X., et al. (2020). *Generalized Focal Loss* (`NeurIPS 2020`); Feng, C., et al. (2021). *TOOD: Task-aligned One-stage Object Detection* (`ICCV 2021`).
