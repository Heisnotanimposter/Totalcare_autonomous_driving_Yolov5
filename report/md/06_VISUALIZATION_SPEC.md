# Visualization Specification: Temporal Ego-centric World Model

Implemented in [src/world_visualizer.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_visualizer.py).

## 1. Split-Screen Layout

- **Total Composite Resolution**: `1360 × 632` (`Left: 720 × 540` | `Right: 640 × 540` | `Bottom Telemetry HUD: 1360 × 92`) at `10 Hz` (matching KITTI native frame rate).

### Left Panel — Observed World
Displays **only** direct camera-frame observations:
- Original KITTI `image_02` RGB frame
- 2D Bounding Boxes color-coded by class (`Car`, `Pedestrian`, `Cyclist`)
- Class Label, sequence-scoped Track ID (`#id`), and Detection Confidence (`0.00–1.00`)

### Right Panel — Inferred World
Structured Ego-centric Bird's-Eye-View (`X: [-18m, +18m]`, `Y: [-10m, +52m]`):
1. **Ego Vehicle**: Yellow triangle anchored at `(0, 0)`
2. **Surrounding Objects**: Class-colored BEV footprints with ID and forward distance (`m`)
3. **Relative Position**: Metric range rings (`10m`–`50m`) in Ego coordinates
4. **Movement Direction & Calibrated Velocity**: Arrow & `m/s` label shown **only** when `P2` calibration is active and track hits `>= 3`
5. **Predicted Trajectory**: `+0.5s`, `+1.0s`, `+1.5s`, `+2.0s` Kalman forecast nodes
6. **Ego Planned Path**: Forward corridor (`±1.6m` half-width, `0–35m` horizon)
7. **Road Structure**: Nominal lane boundaries aligned to Ego corridor
8. **Occupancy / Obstacles**: Solid occupied regions vs angular shadow cones
9. **Collision Risk**: Conflict zone circles (`CONFLICT +t_s`) and risk highlight when `Risk Indicator >= 0.45`
10. **Uncertainty Visualization**: `1σ` covariance ellipses around each future horizon node + HUD breakdown (`Det`, `Depth ±m`, `Track`, `Pred +2s ±m`)
11. **Temporal State Separation**: Fading `Past World` trails (exponentially decayed weight) vs `Present World` boxes vs `Future World` probabilistic cones
12. **Spatial Limits**: Rear area (`Y < 0`) and outside camera horizontal FOV explicitly hatched and labeled `UNKNOWN (Outside Front Camera FOV)`

---

## 2. Information State Encoding

| State | Visual Encoding | Semantics |
|---|---|---|
| **Observed** | Solid filled BEV marker + white border; Left Panel 2D box | Directly detected in current frame inside FOV |
| **Inferred** | Hollow outline `[INF]`, velocity arrows, future ellipses, conflict circles | Estimated by ground-plane projection or Kalman kinematics |
| **Occluded** | Dark angular shadow wedges behind foreground vehicles; hollow coasting boxes | Line-of-sight blocked or temporarily missed (`<= 1.0s`) |
| **Unknown** | Dark diagonal hatched background (`Y < 0` and outside FOV cone) | Unobservable by single front camera; never hallucinated |

---

## 3. Five-Second Staged Demonstration Timeline (`--staged-demo`)

| Time Window | Frames (`10 Hz`) | Active Layers |
|---|---|---|
| **0.0–1.0s** | `0–9` | **Raw Observation**: Camera video + Ego `(0,0)` + FOV / `Unknown` initialization |
| **1.0–2.0s** | `10–19` | **Object Perception**: + 2D BBox, Class, Confidence, Track IDs, BEV detections, Occlusion cones |
| **2.0–3.0s** | `20–29` | **Temporal Understanding**: + Past World fading trails, calibrated velocity vectors |
| **3.0–4.0s** | `30–39` | **Future Prediction**: + Multi-horizon trajectories (`0.5s–2.0s`), `1σ` covariance ellipses, Ego planned path |
| **4.0–5.0s+** | `40+` | **Risk Assessment & Safety Shield**: + Risk Indicator gauge, Conflict zones, Advisory Safety Shield Decision |

Full-overlay mode (`--full-overlay`) renders all layers starting from `t = 0.0s`.

