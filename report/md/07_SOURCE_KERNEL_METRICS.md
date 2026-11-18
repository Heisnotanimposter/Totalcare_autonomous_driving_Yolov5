# 소스·커널·보고서·지표 통합 명세 (Appendix)

본 부록은 저장소 전체의 **소스 코드(`src/`)**, **원격 GPU/CPU 커널(`kaggle/`, `kaggle_world_model/`, `colab/`)**, **설계·평가 보고서(Markdown/LaTeX/PDF)**, **정량 지표 및 시연 산출물(`results/`, `logs/`)** 을 한눈에 추적할 수 있도록 정리한 종합 색인입니다.

---

## 1. 소스 코드 모듈 (`src/`)

| 파일 | 라인 수 | 핵심 역할 및 구현 내용 |
|---|---:|---|
| `src/kitti_prep.py` | 624 | KITTI Zip/디렉터리 CRC 및 이미지 디코드 무결성 검증, 시퀀스 비독립 14/4/3 분할, `DontCare`+Distractor(`Van`, `Person`, `Person_sitting`) 114 픽셀 마스킹, Train 전용 희소 클래스 오버샘플링(`train_weighted.txt`, `6,974` entries) |
| `src/track_utils.py` | 620 | ByteTrack (`imgsz=960`, `conf=0.25`) 다중 객체 추적, `MobileSAM` (`mobile_sam.pt`) Box-Prompt 인스턴스 세그멘테이션(`trajectories_masks.json`), 접지점(`mask_bottom_xy`) 추출, KITTI 18-column TrackEval 포맷 출력 |
| `src/world_model.py` | 1,235 | `P2` 역분산 깊이 융합(`ground_plane_bev_fusion`), `EgoPose2D` `SE(2)` OXTS ENU 보상 & CTRV 경로 예측, 4D World-ENU 선형 칼만 필터(`+0.5s~+2.0s`), `Observed/Inferred/Occluded/Unknown` 상태 머신, 횡방향 공분산 기반 차로 위험도 및 `CollisionRiskSmoother` (`\|dR/dt\|` 슬루율 제한 + TTC 디바운싱 + 히스테리시스) |
| `src/world_visualizer.py` | 710 | `1360×632` 좌우 분할 영상 렌더러 (`Left: Camera + MobileSAM Masks` \| `Right: BEV Inferred World`), `cov_to_pixel_ellipse` 야코비안 $J=\text{diag}(s_x,-s_y)$ 타원 투영, 5초 단계별(`--staged-demo`) 및 전체 프레임(`--full-overlay`) MP4·키프레임 출력 |
| `src/validation_experiments.py` | 499 | **RQ1**(좌표 왕복 및 정지 객체 Ego-motion 보상), **RQ2**(미래 예측 `OXTS+CTRV` vs `No-OXTS` vs `Zero-Vel` 및 2D 마할라노비스 커버리지), **RQ3**(불확실성 인과성 `Arm A/B/C`), **RQ4**(Gradient Smoothing 스파이크·플립 감소율) 자동 검증 |
| `src/dataset_prep.py`, `src/detect.py`, `src/train.py`, `src/export.py` | 377 | 기본 대시캠 9부류 YOLO 데이터 변환, 추론/위험 배너 시각화, YOLOv5/Ultralytics 학습·ONNX/TensorRT 내보내기 유틸리티 |

---

## 2. 원격 GPU 훈련·실행 커널 (`kaggle/`, `kaggle_world_model/`, `colab/`)

| 커널 / 노트북 | 라인 수 | 실행 환경 | 수행 단계 및 출력 |
|---|---:|---|---|
| `kaggle/train_track_kitti.py` | 342 | Kaggle `2× Tesla T4` (15GB×2) | `test013/totalcare-kitti-yolo-train-and-track` (v2 `COMPLETE`): Audit → YOLO Export → `YOLO11m @ 960` DDP 학습(24ep, ~108분) → Val/Test 평가 → `0015`/`0019` ByteTrack + MobileSAM 추출 → World Model 1차 검증 |
| `kaggle_world_model/run_world_model_kaggle.py` | 205 | Kaggle CPU (`kernel_sources` 연동) | `test013/totalcare-kitti-world-model-demo` (v7 `COMPLETE`): v2 `trajectories.csv` + `trajectories_masks.json` 마운트 → `v3` BEV 야코비안 타원 & 오경보 억제 패치 재렌더링(4종 MP4) → RQ1–RQ4 최종 재검증 |
| `colab/01_kitti_colab_audit_and_prep.ipynb` | 139 | Google Colab CLI / Drive | `/content/drive/MyDrive/KITTI_Tracking/` 경로 유지형 Stage 1 감사 및 YOLO Export 노트북 |
| `colab/02_traffic_detect_track_colab.ipynb` | 112 | Google Colab GPU | Colab 환경 `YOLO` 학습 및 `ByteTrack` 추적 실행 노트북 |

---

## 3. 보고서 및 기술 문서 (`FINAL_REPORT`, `ML_LAYER_SPEC` 외)

| 문서 | 라인 수 | 내용 요약 |
|---|---:|---|
| `FINAL_REPORT.md` | 117 | 기본 평가표 7개 항목(데이터, 5초+ 영상, GitHub 공개, 보고서, 벤치마크, 링크, Layer 설명서) 1:1 매핑 최종 보고서 |
| `ML_LAYER_SPEC.md` | 99 | `YOLO11m` 24개 모듈(`Layer 0~23`), `MobileSAM`, `ByteTrack`, `World Model & Safety Shield` Layer별 상세 및 7편 학술 참고자료 |
| `architecture.md` | 105 | 7단계 시스템 아키텍처, `P2` 카메라↔Ego 좌표 수식, `Past→Present→Future` 상태 전이 및 `CollisionRiskSmoother` 설계 |
| `dataset_audit.md` | 47 | KITTI `8,008`장 무결성 검증, `DontCare`/Distractor 마스킹 및 14/4/3 시퀀스 비독립 분할 명세 |
| `experiment_plan.md` | 138 | 연구 가설(H1–H3), 베이스라인(B0–B2), `v1→v2→v3` 정량 비교 및 RQ1–RQ4 실측 결과 기록 |
| `visualization_spec.md` | 54 | `1360×632` 좌우 분할 화면 인코딩 규격 및 5초 단계별(`0–5s+`) 시연 타임라인 |

---

## 4. 핵심 정량 지표 및 산출물 아카이브 (`results/`, `logs/`)

| 구분 | 아티팩트 경로 | 핵심 수치 요약 |
|---|---|---|
| **Stage 2 검출 모델 (`v2`)** | `results/v2_improved/outputs/weights/best.pt` (`38.62 MB`) | `YOLO11m @ 960` (`231 layers`, `20.06M params`), **Val `mAP@0.5 = 0.5235`** (`+15.8%` vs `v1 0.4521`), **Test `mAP@0.5 = 0.4269`** (`Cyclist Test AP50 +115.7%`) |
| **Stage 3 SAM+ByteTrack** | `results/v2_improved/outputs/tracking_summary.json` | `mobile_sam.pt` 인스턴스 마스크: `0015` **`1,370`건**(`41` 트랙), `0019` **`3,788`건**(`203` 트랙) |
| **Stage 4–7 최종 시연 (`v3`)** | `results/v3_bev_fixed/world_outputs/world_model_*` | `0015`(`37.6s`, `11.58 MB`) / `0019`(`105.9s`, `53.48 MB`) + 각 `8.0s` staged MP4 & 키프레임 |
| **RQ1–RQ4 최종 검증 (`v3`)** | `results/v3_bev_fixed/world_outputs/validation_0015.json`, `validation_0019.json` | **RQ1** `2.161e-14 m` / 정지체 `0.227 m/s`; **RQ2** `0015` ADE `3.6052 m` (`1σ 0.3939`); **RQ3** `A<B<C` 단조성(`0.2220<0.2791<0.2792`, `0.5242<0.6290<0.6396`); **RQ4** `>1.5/s` 스파이크 **`100%` 제거**, `v3` 오경보 Risk `-57.0%`(0015) / `-30.5%`(0019) |
| **실행 로그** | `logs/v1_detect_track/`, `logs/v2_improved/`, `logs/v3_bev_fixed/` | Kaggle DDP 학습(`231 layers` 요약 포함) 및 World Model 재렌더링 전체 콘솔 로그 보관 |
