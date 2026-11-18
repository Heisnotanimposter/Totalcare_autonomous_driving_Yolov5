# Total Care Traffic Extension — Google Colab + Google Drive Pipeline

이 문서는 기존 2022/2024년 프레임 독립 검출 기준선(`src/detect.py`, `src/train.py`, `Updates/`)을 그대로 보존하면서, **Google Colab + Google Drive** 환경에서 **다중 객체 추적(Tracking ID · 중심좌표 궤적 · 5초 MP4) 및 KITTI Tracking 2~4단계 실증**을 수행하기 위해 추가된 확장 모듈 안내입니다.

## 추가된 파일
- `colab/01_kitti_colab_audit_and_prep.ipynb` — Drive 마운트 후 S3 `KITTI Tracking`(`14.73 GiB` 이미지 + `2.15 MiB` 라벨) 확보, 2단계 무결성·8,008프레임 1:1 대응·해상도 검사, 3단계 시퀀스 단위 비겹침 분할, `DontCare`/`Distractor` 마스킹 Export
- `colab/02_traffic_detect_track_colab.ipynb` — 4단계 최소 실행 가능성 검증(검출 학습 + `model.track(persist=True)` 추적 + `trajectories.csv`/`json` 중심좌표 저장 + `≥5초` MP4 + KITTI 18컬럼 TrackEval 예측 포맷 출력)
- `src/kitti_prep.py` — `(sequence_id, track_id)` 시퀀스 범위 tracklet 파싱, 시퀀스 누수 검증, `DontCare`/인접 distractor(`Van`/`Person`) 마스킹 및 YOLO 변환기
- `src/track_utils.py` — `2022 src/detect.py`의 프레임 독립 추론 한계를 보완한 지속 추적·속도 추정·KITTI 18컬럼/궤적 내보내기 모듈

## 주의 사항 (감사 반영)
1. **시퀀스 범위 ID:** KITTI `track_id`는 전역 ID가 아닌 시퀀스 내부 고유 ID이므로 항상 `(sequence_id, track_id)` 복합 키로 취급합니다.
2. **평가 범위 분리:** KITTI 공식 TrackEval(`Kitti2DBox`)은 `Car`와 `Pedestrian` 2종만 평가(`Van`/`Person` distractor 무시)하므로, 수업 과제(3클래스+) 평가 시에는 별도 자체 프로토콜을 분리 명시해야 합니다.
3. **라이선스 (`CC BY-NC-SA 3.0`):** 조건 충족 시 재배포·2차 변경을 허용하나, Public GitHub 게시 대상(원본 샘플 여부·학습 가중치·원본 프레임 포함 5초 추론 영상)별 BY/NC/SA 적용 검토가 필요합니다.
