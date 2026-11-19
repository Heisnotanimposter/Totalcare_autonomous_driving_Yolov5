# 머신러닝·딥러닝 모델 Layer별 기술설명서 및 참고자료

**Repository:** [Totalcare_autonomous_driving_Yolov5](https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5)  
**대상 파이프라인:** `Stage 2 (YOLO11m + MobileSAM) → Stage 3 (ByteTrack) → Stage 4–7 (Temporal Ego-centric World Model & Safety Shield)`

---

## 1. 객체 검출기: `YOLO11m` (`results/v2_improved/outputs/weights/best.pt`)

- **모델 규격 (Kaggle `2x Tesla T4` 학습 로그 실측치)**:
  - **전체 구성**: `231 layers`, `20,055,321 parameters`, `68.3 GFLOPs` (추론 융합 Fused 기준 `125 layers`, `20,032,345 parameters`, `67.8 GFLOPs`)
  - **입력 해상도**: `960 × 960` RGB (`AMP FP16` 학습, `nc = 3`: `Car`, `Pedestrian`, `Cyclist`)
  - **Baseline (`v1 YOLOv8s`) 대비 차이**: `YOLOv8s` (`129 layers`, `11.14M params`, `28.6 GFLOPs`, `640px`)에서 Backbone 효율화(`C2f → C3k2`) 및 Spatial Attention(`C2PSA`) 도입 + `960px` 고해상도 입력으로 소형·원거리 보행자/자전거 특징 손실을 개선.

### 1.1 `YOLO11m` 전체 24개 모듈 Layer별 상세 구조 (`Layer 0` – `Layer 23`)

아래 표는 Kaggle 학습 실행 시 Ultralytics `8.4.212` 엔진이 출력한 실제 네트워크 그래프(`logs/v2_improved/totalcare-kitti-yolo-train-and-track.log`)를 기준으로 정리한 Layer별 역할 명세입니다.

| Layer | 입력(`from`) | 모듈 (`Ultralytics`) | 출력 채널 / 인자 (`args`) | 파라미터 수 | 소속 구간 | Layer 기술적 역할 |
|---:|---|---|---|---:|---|---|
| **0** | `-1` | `nn.modules.conv.Conv` | `[3, 64, 3, 2]` | `1,856` | Backbone (Stem) | `3×3` Conv (`stride=2`) + BatchNorm + SiLU로 `960→480` 다운샘플링(P1/2) 및 에지·질감 저수준 특징 추출 |
| **1** | `-1` | `nn.modules.conv.Conv` | `[64, 128, 3, 2]` | `73,984` | Backbone (Stage 1) | `3×3` Conv (`stride=2`)로 `480→240` 축소(P2/4) 및 채널 확장 |
| **2** | `-1` | `nn.modules.block.C3k2` | `[128, 256, 1, True, 0.25]` | `111,872` | Backbone (Stage 1) | Cross-Stage Partial Bottleneck (`c3k=False`, `e=0.25`). 입력 채널 분할 후 작은 커널 Conv 병목과 Skip 결합으로 연산량 절감 |
| **3** | `-1` | `nn.modules.conv.Conv` | `[256, 256, 3, 2]` | `590,336` | Backbone (Stage 2) | `3×3` Conv (`stride=2`), `240→120` 축소(P3/8 — 소형 객체 특징 맵 기준점) |
| **4** | `-1` | `nn.modules.block.C3k2` | `[256, 512, 1, True, 0.25]` | `444,928` | Backbone (Stage 2) | P3/8 특징 강화 블록. 원거리 `Pedestrian`/`Cyclist` 윤곽 특징 보존 |
| **5** | `-1` | `nn.modules.conv.Conv` | `[512, 512, 3, 2]` | `2,360,320` | Backbone (Stage 3) | `3×3` Conv (`stride=2`), `120→60` 축소(P4/16 — 중형 객체 특징 맵) |
| **6** | `-1` | `nn.modules.block.C3k2` | `[512, 512, 1, True]` | `1,380,352` | Backbone (Stage 3) | `C3k` 블록 사용(`c3k=True`, 확장 깊이). 중거리 차량·자전거 형태 추상화 |
| **7** | `-1` | `nn.modules.conv.Conv` | `[512, 512, 3, 2]` | `2,360,320` | Backbone (Stage 4) | `3×3` Conv (`stride=2`), `60→30` 축소(P5/32 — 대형 근거리 차량·전역 문맥) |
| **8** | `-1` | `nn.modules.block.C3k2` | `[512, 512, 1, True]` | `1,380,352` | Backbone (Stage 4) | 고수준 의미 특징 추출 |
| **9** | `-1` | `nn.modules.block.SPPF` | `[512, 512, 5]` | `656,896` | Backbone (Context) | Spatial Pyramid Pooling - Fast. `5×5` MaxPool 직렬 3회 재사용으로 멀티스케일 수용 영역(Receptive Field) 통합 |
| **10** | `-1` | `nn.modules.block.C2PSA` | `[512, 512, 1]` | `990,976` | Backbone (Attention) | Cross-Stage Partial **Pyramid Squeeze / Spatial Attention**. Multi-head Self-Attention으로 가려진 차로 객체·장거리 문맥 가중치 집중 |
| **11** | `-1` | `nn.Upsample` | `[None, 2, 'nearest']` | `0` | Neck (Top-Down FPN) | P5(`30×30`) 특징을 P4(`60×60`) 크기로 최근접 2배 업샘플링 |
| **12** | `[-1, 6]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Top-Down FPN) | Layer 11(상위 문맥)과 Layer 6(P4 공간 특징) 채널 병합(`512+512→1024`) |
| **13** | `-1` | `nn.modules.block.C3k2` | `[1024, 512, 1, True]` | `1,642,496` | Neck (Top-Down FPN) | 병합된 P4 특징 정제 |
| **14** | `-1` | `nn.Upsample` | `[None, 2, 'nearest']` | `0` | Neck (Top-Down FPN) | P4(`60×60`)를 P3(`120×120`) 크기로 2배 업샘플링 |
| **15** | `[-1, 4]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Top-Down FPN) | Layer 14와 Layer 4(P3 고해상도 특징) 병합(`512+512→1024`) |
| **16** | `-1` | `nn.modules.block.C3k2` | `[1024, 256, 1, True]` | `542,720` | Neck → **P3/8 출력** | 소형 객체(`Pedestrian`, 원거리 `Cyclist`) 최종 검출 특징 맵 출력 (`256ch`) |
| **17** | `-1` | `nn.modules.conv.Conv` | `[256, 256, 3, 2]` | `590,336` | Neck (Bottom-Up PAN) | Path Aggregation Network 하향 경로: P3(`120×120`) → `60×60` 다운샘플링 |
| **18** | `[-1, 13]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Bottom-Up PAN) | 위치 정밀도(Layer 17) + 의미 특징(Layer 13) 결합(`256+512→768`) |
| **19** | `-1` | `nn.modules.block.C3k2` | `[768, 512, 1, True]` | `1,511,424` | Neck → **P4/16 출력** | 중형 객체 최종 검출 특징 맵 출력 (`512ch`) |
| **20** | `-1` | `nn.modules.conv.Conv` | `[512, 512, 3, 2]` | `2,360,320` | Neck (Bottom-Up PAN) | P4(`60×60`) → `30×30` 다운샘플링 |
| **21** | `[-1, 10]` | `nn.modules.conv.Concat` | `dim=1` | `0` | Neck (Bottom-Up PAN) | P4 하향 특징 + Backbone C2PSA 출력(Layer 10) 결합(`512+512→1024`) |
| **22** | `-1` | `nn.modules.block.C3k2` | `[1024, 512, 1, True]` | `1,642,496` | Neck → **P5/32 출력** | 근거리 대형 차량(`Car`) 최종 검출 특징 맵 출력 (`512ch`) |
| **23** | `[16, 19, 22]` | `nn.modules.head.Detect` | `[nc=3, reg_max=16, [256,512,512]]` | `1,413,337` | **Head (Decoupled)** | Anchor-Free 분리형 검출 헤드. 회귀 분기(`4×16=64` DFL 분포 예측) + 분류 분기(`nc=3` Depthwise Separable Conv) |

### 1.2 학습 손실 함수 및 최적화 Layer
- **Box Regression (`box=7.5`, `dfl=1.5`)**: **CIoU Loss** (중심 거리·종횡비·겹침 면적 동시 최적화) + **Distribution Focal Loss (DFL, `reg_max=16`)** (바운딩 박스 네 변을 이산 확률 분포로 예측 후 기댓값 산출, `model.23.dfl.conv.weight` 고정 가중치 투영).
- **Classification (`cls=0.5`)**: Class별 독립 **Binary Cross-Entropy (BCE with Logits)** + **Task-Aligned Assigner (TAL)** 양성/음성 앵커 동적 할당.
- **Cosine LR + MixUp + Close-Mosaic**: 3 epoch 선형 워밍업(`warmup_bias_lr=0.1`) 후 Cosine Annealing(`lrf=0.01`), 후반 10 epoch Mosaic 비활성화(`close_mosaic=10`)로 자연 영상 분포 정렬.

---

## 2. 인스턴스 세그멘테이션: `MobileSAM` (`src/track_utils.py`)

ByteTrack이 부여한 Track별 2D Bounding Box를 **Box Prompt**로 입력받아 픽셀 수준 객체 마스크를 생성하고, Stage 4 단안 깊이 추정의 지면 접지점(`mask_bottom_xy`)과 점유율 가중치(`mask_fill_ratio`)를 제공합니다.

| 구성 모듈 (MobileSAM) | 내부 Layer 메커니즘 | 입·출력 사양 | 파이프라인 내 기능 |
|---|---|---|---|
| **1. Image Encoder (`TinyViT`)** | `PatchEmbed` (Conv×2) → **Stage 1** (`MBConv` 블록, 지역 채널-공간 추출) → **Stage 2–4** (`Windowed Multi-Head Self-Attention` + `MLP` + Patch Merging Downsampling) → Neck (`1×1` + `3×3` Conv) | 입력 `BGR` 이미지 (`1242×375`) → `64×64×256` Feature Embedding | ViT-H 대신 지식 증류(Decoupled Distillation)로 경량화된 `TinyViT` 백본(`38.8 MB`) 사용 |
| **2. Prompt Encoder** | Box Prompt `[x1, y1, x2, y2]`의 좌상단·우하단 좌표에 **Random Fourier Positional Encoding** + 코너 타입 Learned Embedding 부여 | Bounding Box 좌표 → `(N, 2, 256)` Sparse Prompt Tokens | 객체 검출 박스를 공간적 프롬프트로 인코딩하여 배경 오검출 억제 |
| **3. Two-Way Transformer Mask Decoder** | Sparse Prompt와 Image Embedding 간 **Cross-Attention** (`Prompt→Image`, `Image→Prompt`) 2블록 + MLP → **`ConvTranspose2d` (`4×` Upsampling)** → Hypernetwork MLP Inner Product | Sparse Tokens + `64×64×256` → 인스턴스별 Binary Mask + Polygon Contour | `cv2.findContours`로 외곽선(`trajectories_masks.json`), `mask_fill_ratio` 및 최하단 중심(`mask_bottom_x/y`) 추출 |

---

## 3. 다중 객체 추적 Layer: `ByteTrack` (`src/track_utils.py`)

- **상태 추정 필터 (Linear Kalman Filter)**: 이미지 평면 8차원 등속 상태 $\mathbf{x} = [c_x, c_y, a, h, \dot{c}_x, \dot{c}_y, \dot{a}, \dot{h}]^T$ (중심 좌표, 종횡비, 높이 및 변화율).
- **2단계 이분 매칭 (Two-Stage Hungarian Association)**:
  1. **High-Confidence Matching (`conf >= 0.5`)**: 기존 활성 트랙과 고신뢰 검출 박스 간 IoU 비용 행렬에 `linear_sum_assignment` 적용.
  2. **Low-Confidence Recovery (`0.1 <= conf < 0.5`)**: 부분 가려짐(`Occlusion`)이나 역광으로 신뢰도가 하락한 잔여 검출을 미매칭 트랙에 재연결하여 ID Switch 감소.
- **시퀀스 격리 검증 (`compute_track_trajectories`)**: 시퀀스 경계에서 트래커 상태를 초기화하고 `(sequence_id, track_id)` 복수 키로 관리하며, 시간 역전/중복 프레임/클래스 전환을 감사 리포트에 기록.

---

## 4. Temporal Ego-centric World Model & Safety Shield Layer (`src/world_model.py`, `src/world_visualizer.py`)

딥러닝 인식 출력을 물리 좌표계와 시간 연속성 모델로 통합하는 미분 가능·칼만 기반 후단 Layer 구조입니다.

1. **P2 역분산 깊이 융합 Layer (`KittiCalibration.ground_plane_bev_fusion`)**:
   - MobileSAM 전경 최하단 접지점(`mask_bottom_xy`, `mask_fill_ratio >= 0.25` 조건) 기반 평면 지면 투영($Y_{\text{ground}} = f_y h_{\text{cam}} / (v - c_y)$)과 클래스 높이 사전($H_c$) 기반 피치 불변 깊이($Y_{\text{height}} = f_y H_c / h_{\text{px}}$)를 각 분산 역수로 융합: $Y_{\text{fused}} = \frac{w_g Y_g + w_h Y_h}{w_g + w_h}$.
2. **OXTS `SE(2)` Ego-Motion 보상 및 CTRV Layer (`EgoPose2D`, `KittiOxtsOdometry`)**:
   - Mercator 투영 기반 `World-ENU` 변환으로 자차 병진·회전($\psi, v_f, w_z$) 성분을 제거하여 정지 객체의 허위 속도를 감쇄(`왕복 오차 < 4e-14 m`).
3. **4D World-ENU 선형 칼만 필터 Layer (`ObjectWorldTrack`)**:
   - 상태 $\mathbf{x}_w = [X_w, Y_w, v_{X,w}, v_{Y,w}]^T$, 백색 가속도 공정 잡음($\sigma_a = 1.1\text{ m/s}^2$), YOLO 신뢰도·MobileSAM 채움 비율에 반비례하는 회전 측정 공분산 $R_w$ 적용. 미래 시점(`+0.5s ~ +2.0s`) 공분산을 야코비안 $J=\text{diag}(s_x,-s_y)$로 BEV 픽셀 타원으로 투영.
4. **$C^1$ 연속 위험 포텐셜 + `CollisionRiskSmoother` Layer**:
   - 횡방향 공분산(`lateral_uncertainty_m`) 기반 차로 침범 여부, 접근 속도 게이팅(`approach_w`), 보정 TTC, 확률적 soft-union $1 - \prod_i (1 - r_i)$로 프레임 위험도를 산출.
   - 비대칭 EMA(`alpha_up=0.36`, `alpha_down=0.22`) → $|dR/dt|$ 슬루율 제한(`1.10/s`, `-0.85/s`) → 트랙별 EMA(`per_track_ema`) → TTC 연속 프레임 디바운싱(`low_ttc_streak`) 및 히스테리시스(`0.04`)로 스파이크를 `100%` 제거.

---

## 5. 참고자료 (References)

1. **Ultralytics YOLO11 / YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023–2025). *Ultralytics YOLO (`YOLO11m`, `C3k2`, `C2PSA`, `YOLOv8`)* — https://docs.ultralytics.com/models/yolo11/
2. **MobileSAM**: Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S.-H., Lee, S., & Hong, C. S. (2023). *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications* (`arXiv:2306.14289`). — https://arxiv.org/abs/2306.14289
3. **Segment Anything (SAM)**: Kirillov, A., Mintun, E., Ravi, N., et al. (2023). *Segment Anything* (`ICCV 2023`, `arXiv:2304.02643`). — https://arxiv.org/abs/2304.02643
4. **ByteTrack**: Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (`ECCV 2022`). — https://arxiv.org/abs/2110.06864
5. **KITTI Object Tracking Benchmark**: Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite* (`CVPR 2012`). — https://www.cvlibs.net/datasets/kitti/eval_tracking.php
6. **Kalman Filter & CTRV Motion Modeling**: Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*; Schubert, R., et al. (2008). *Comparison and Evaluation of Advanced Motion Models for Vehicle Tracking* (`FUSION 2008`).
7. **DFL & Task-Aligned Assigner**: Li, X., et al. (2020). *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection* (`NeurIPS 2020`); Feng, C., et al. (2021). *TOOD: Task-aligned One-stage Object Detection* (`ICCV 2021`).
