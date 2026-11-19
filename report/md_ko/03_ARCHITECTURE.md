# 시스템 아키텍처: 카메라 중심 시계열 자차 중심 World Model

## 1. 시스템 개요

본 프로젝트는 단일 전방 카메라 영상을 입력받아 구조화된 **시계열 자차 중심 World Model(Temporal Ego-centric World Model)**, 다중 시점 미래 궤적 예측, 위험도 지표, 그리고 권고형 Safety Shield 출력을 생성하는 7단계 파이프라인을 구현합니다.

```text
[Stage 1: 입력 영상 (KITTI 10 Hz)]
        │
        ▼
[Stage 2: 시각 인식 (src/kitti_prep.py, YOLO11m/YOLO11l + MobileSAM + SegFormer-B2)]
        │  2D BBox, 클래스(Car, Pedestrian, Cyclist), 검출 신뢰도
        │  희소 클래스 오버샘플링 + 인스턴스 폴리곤 + 도로/비도로/자차로 세그멘테이션
        ▼
[Stage 3: 시계열 이해 (src/track_utils.py, ByteTrack + MobileSAM)]
        │  시퀀스 격리 Track ID, 프레임 ID, 타임스탬프, mask_area_px, mask_fill_ratio, mask_bottom_xy
        ▼
[Stage 4: 자차 중심 World Model (src/world_model.py, src/road_lane.py)]
        │  P2 역분산 깊이 융합 (평면 지면 + 클래스 높이 사전값 + 마스크 최하단 접지점)
        │  OXTS IMU/GPS 자차 움직임 보상 (EgoPose2D: Ego <-> World ENU)
        │  상태 머신: Past <-> Present <-> Future | Observed / Inferred / Occluded / Unknown
        ▼
[Stage 5: 미래 궤적 예측 (src/world_model.py, World-ENU Kalman + CTRV Ego)]
        │  t + 0.5s, 1.0s, 1.5s, 2.0s 평균 위치 + 신뢰도/마스크 기반 1-sigma 공분산 타원
        ▼
[Stage 6: 위험도 평가 (src/world_model.py, C1 포텐셜 + CollisionRiskSmoother)]
        │  부드러운 시그모이드 위험 포텐셜 + 비대칭 EMA + 슬루율 |dR/dt| 제한 + 보정 TTC
        ▼
[Stage 7: 안전 인지 제어 권고 (src/world_model.py & src/world_visualizer.py)]
           히스테리시스 디바운싱 권고형 Safety Shield + 좌우 분할 화면(마스크 + BEV World)
```

---

## 2. 좌표계 및 변환 수식

### 2.1 KITTI 카메라 좌표계
KITTI Tracking (`calib/XXXX.txt`, `label_02/XXXX.txt`) 기준:
- $X_{\text{cam}}$: 우측 방향 ($+$)
- $Y_{\text{cam}}$: 아래 방향 ($+$)
- $Z_{\text{cam}}$: 광학축 전방 ($+$)

### 2.2 자차 중심 BEV 좌표계
`KittiCalibration` ([src/world_model.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_model.py)) 에 정의됨:
- $X_{\text{ego}}$: 횡방향 우측 ($+$), 단위 m
- $Y_{\text{ego}}$: 종방향 전방 ($+$), 단위 m
- $Z_{\text{ego}}$: 수직 상방 ($+$), 단위 m

표준 수평 장착($h_{\text{cam}} = 1.65\text{ m}$, 피치/롤 0 가정) 기준:
$$X_{\text{ego}} = X_{\text{cam}}, \quad Y_{\text{ego}} = Z_{\text{cam}}, \quad Z_{\text{ego}} = h_{\text{cam}} - Y_{\text{cam}}$$

### 2.3 보정된 역분산 깊이 융합 및 해석적 불확실성
KITTI 내부 파라미터 $P_2$ ($f_x, f_y, c_x, c_y$), 지면 접지점 $(u_{\text{mid}}, v_{\text{bottom}})$ (`mask_fill_ratio >= 0.25`일 때 MobileSAM 전경 최하단 `mask_bottom_xy`, 그 외 2D 박스 하단 중심 사용), 그리고 클래스 표준 높이 사전값 $H_c$ (`CLASS_HEIGHT_PRIORS_M`: `Car=1.52m`, `Pedestrian=1.72m`, `Cyclist=1.68m`)가 주어질 때:
1. **평면 지면 투영 ($Y_{\text{ground}}$)** 및 피치 불확실성 $\sigma_\theta = 0.012\text{ rad}$:
   $$Y_{\text{ground}} = \frac{f_y \cdot h_{\text{cam}}}{v_{\text{bottom}} - c_y}, \qquad \sigma_{\text{ground}} = \sqrt{\left(\frac{Y_{\text{ground}}^2}{f_y h_{\text{cam}}}\sigma_v\right)^2 + \left(\frac{Y_{\text{ground}}^2}{h_{\text{cam}}}\sigma_\theta\right)^2}$$
2. **피치 불변 높이 사전 깊이 ($Y_{\text{height}}$)** (화면 경계 미절단 박스 대상):
   $$Y_{\text{height}} = \frac{f_y \cdot H_c}{h_{\text{px}}}, \qquad X_{\text{ego}} = \frac{(u_{\text{mid}} - c_x) \cdot Y_{\text{fused}}}{f_x}$$
3. **역분산 융합 (Inverse-Variance Fusion)**:
   $$w_g = \sigma_{\text{ground}}^{-2}, \quad w_h = \sigma_{\text{height}}^{-2}, \qquad Y_{\text{fused}} = \frac{w_g Y_{\text{ground}} + w_h Y_{\text{height}}}{w_g + w_h}, \qquad \sigma_Y = (w_g + w_h)^{-1/2}$$
World-ENU 칼만 필터링의 측정 공분산 $R_w$는 검출 신뢰도와 MobileSAM 마스크 채움 품질(`conf`, `mask_fill_ratio`)에 반비례하여 추가 스케일링됩니다.

**무결성 원칙:**
- $P_2$ 캘리브레이션이 누락된 경우 `TemporalWorldModel`은 `calibrated = False`로 설정하고 미터법 속도(`m/s`) 및 충돌 여유 시간(`TTC`) 산출을 **거부**합니다.
- 투영된 3차원 위치는 항상 **추정값(Inferred)**(단안 지면/높이 융합 추정치)으로 분류되며 **관측값(Observed)** 으로 표기하지 않습니다.

---

## 3. 자차 움직임 보상 및 시계열 World Model (`과거 → 현재 → 미래`)

`EgoPose2D`, `KittiOxtsOdometry`, `TemporalWorldModel`, `ObjectWorldTrack` ([src/world_model.py](file:///Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/src/world_model.py))에 구현됨:

1. **OXTS 자차 움직임 보상 (`Ego ↔ World ENU`)**:
   - `KittiOxtsOdometry`는 KITTI `oxts/<seq>.txt`의 `(lat, lon, yaw, vf, vl, wz)`를 0번 프레임 기준 메르카토르 투영을 통해 지역 미터법 **동-북(ENU)** 좌표계로 변환합니다.
   - `EgoPose2D.ego_to_world_xy`와 `world_to_ego_xy`는 순간 자차 중심 검출 좌표(`+X` 우측, `+Y` 전방)를 정지된 World-ENU 좌표계로 상호 변환합니다(실측 왕복 오차 `< 4e-14 m`).
   - 객체 속도와 4D 칼만 상태 `[X_w, Y_w, vX_w, vY_w]`는 **World-ENU**에서 필터링(자차 병진·요 회전 제거)되며, 자차의 계획 주행 경로는 `vf`와 `wz`를 이용한 등선회율·등속(`predict_ctrv`) 롤아웃으로 산출됩니다.

2. **시계열 상태 전이 (`이전 미래 → 현재 → 과거`)**:
   - 시각 $t_k$에서 활성 트랙의 World 좌표계 4D 운동학 칼만 상태와 공분산 $P$를 $\Delta t = t_k - t_{k-1}$만큼 전파한 뒤, 시각 $t_k$의 자차 BEV 좌표계로 재투영합니다.
   - 현재 프레임 검출과 매칭된 트랙은 회전된 측정 공분산 $R_w = R_{\text{ego}\to w} \, \text{diag}(\sigma_X^2, \sigma_Y^2) \, R_{\text{ego}\to w}^T$로 갱신되고 `observation_state = "Observed"`를 부여받습니다.
   - 시각 $t_k$에 미관측되었으나 `max_coast_sec` ($1.0\text{ s}$) 이내인 트랙은 지수 신뢰도 감쇠 $\exp(-\Delta t_{\text{miss}} / 0.6)$와 함께 `observation_state = "Occluded"`로 전이됩니다.
   - `max_coast_sec`를 초과하여 미관측된 트랙은 `last_seen_ts` 및 World 기준 위치 이력과 함께 `past_world.expired_tracks`로 보관됩니다.
   - `과거(Past)`, `현재(Present)`, `미래(Future)`는 시각 $t_k$ 기준 상대 타임스탬프로 색인되고 World-ENU에서 유지되므로, 자차 회전이나 후진 상황에서도 시계열 의미가 불변입니다.

3. **공간 정보 상태 (`Observed / Inferred / Occluded / Unknown`)**:
   - **Observed (관측)**: 전방 카메라 화각(FOV) 내에서 현재 프레임에 직접 검출·추적된 대상.
   - **Inferred (추정)**: 지면 투영 BEV 위치, 운동학 속도, 칼만 미래 궤적(`+0.5s`~`+2.0s`), 위험도 지표.
   - **Occluded (폐색)**: 전경 객체 후방으로 레이캐스트된 각도 음영 영역(`compute_occlusion_sectors`) 및 일시 유실 유지 트랙.
   - **Unknown (미관측 영역)**: 자차 후방($Y_{\text{ego}} < 0$) 및 전방 카메라 수평 화각(`2 * atan2(W / 2, fx)`) 외부 영역.

---

## 4. Safety Shield 정책 및 충돌 감지기 기울기 평활화 (권고 전용)

`assess_risk_and_safety_shield`는 다음 요소를 결합합니다:
1. **$C^1$ 연속 미분 가능 위험 포텐셜**: 개별 객체 위험 점수는 실제 **횡방향 공분산**(`lateral_uncertainty_m`, `cov_future_ego[0][0]`) 기반 차로 침범 판정, 접근 속도 게이팅(`approach_w`, 정지 차량·보도 오경보 방지), 보정 TTC, 그리고 소형 최조 충돌 구간 `conflict_zones` (`[1.0, 2.2] m`, 트랙당 최대 1개)에 부드러운 `_sigmoid` 곡선을 적용하며, 프레임 전체 위험도는 확률적 soft-union $1 - \prod_i (1 - r_i)$로 집계됩니다.
2. **종횡비 보정 BEV 불확실성 타원 (`BevCanvasMapper.cov_to_pixel_ellipse`)**: World-ENU $2\times 2$ 공분산을 야코비안 $J = \text{diag}(s_x, -s_y)$로 BEV 픽셀 좌표에 투영하여 종방향 깊이 불확실성이 인접 차로로 과대 확장되지 않도록 하며, 반투명 시점 채움(`t = +1.0s, +2.0s`)과 단일 `+2.0s` 외곽선으로 시각화합니다.
3. **기울기 평활화 (`CollisionRiskSmoother`)**: 비대칭 EMA(`alpha_up=0.36`, `alpha_down=0.22`), 명시적 슬루율 제한(`max_rise_per_sec=1.10/s`, `max_fall_per_sec=0.85/s`), 트랙별 위험도 EMA(`per_track_ema`), 저-TTC 연속 프레임 디바운싱, 그리고 Safety Shield 임계값 경계 히스테리시스(`hysteresis_margin=0.04`)를 통해 프레임 간 급격한 스파이크를 제거합니다.

출력은 엄격히 **운전자 권고(Advisory)** 용도로만 제공됩니다:
- `Normal Driving` (정상 주행)
- `Caution` (주의)
- `Speed Reduction` (감속 권고)
- `Braking Recommendation` (제동 권고)
- `Emergency Intervention` (긴급 개입 권고)
