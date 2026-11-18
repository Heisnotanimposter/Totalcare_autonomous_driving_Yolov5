# TensorBoard 기준 학습·평가 지표 요약 (ROC / FROC · F1 Score · Scalars)

- **대상 경로**: `results/v2_improved/outputs/train_results/`
- **실제 TFRecord 이벤트 로그**: `tensorboard --logdir /Users/seungwonlee/Documents/GitHub/Totalcare_autonomous_driving_Yolov5/results/v2_improved/outputs/train_results/tensorboard_logs`
- **인터랙티브 대시보드**: `tensorboard_dashboard.html` (SCALARS, ROC/FROC & F1, CONFUSION MATRIX, HPARAMS 탭 구성)

## 1. F1 Score 요약 (Eval Point + Confidence Sweep Peak)
| Split | Precision | Recall | **Eval F1** | Peak Macro F1 (@Conf) | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| **Val (832 imgs)** | 0.7170 | 0.4822 | **0.5766** | **0.555** (@0.191) | 0.5235 | 0.3060 |
| **Test (1,608 imgs)** | 0.6732 | 0.4037 | **0.5047** | **0.487** (@0.322) | 0.4269 | 0.2145 |

## 2. Detection FROC (TPR vs FPPI) & One-vs-Rest Confusion-Matrix ROC
> 객체 검출에서는 배경(True Negative) 박스 수가 무한대이므로 검출기 표준 ROC 대응 지표인 **FROC (TPR vs 이미지당 오경보 FPPI)**와 매칭된 3-클래스 분류 행렬 기반 **One-vs-Rest ROC (TPR vs FPR)**를 함께 제공합니다.

| Split | Class | Peak F1 (@Conf) | FROC AUC (0–5 FPPI) | TPR @ 0.5 / 1.0 / 2.0 FPPI | OvR TPR | OvR FPR | OvR F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Val | Car | 0.848 (@0.371) | 0.8238 | 0.803 / 0.813 / 0.822 | 0.9989 | 0.0341 | 0.9813 |
| Val | Pedestrian | 0.601 (@0.197) | 0.6402 | 0.526 / 0.587 / 0.645 | 0.8931 | 0.1330 | 0.8453 |
| Val | Cyclist | 0.252 (@0.057) | 0.2633 | 0.210 / 0.248 / 0.248 | 0.3376 | 0.0411 | 0.4266 |
| Val | **Micro-All** | 0.672 (@0.311) | 0.6141 | 0.549 / 0.585 / 0.623 | — | — | — |
| Test | Car | 0.729 (@0.365) | 0.7094 | 0.669 / 0.697 / 0.716 | 0.9842 | 0.0587 | 0.8824 |
| Test | Pedestrian | 0.435 (@0.063) | 0.3744 | 0.261 / 0.320 / 0.374 | 0.8267 | 0.1411 | 0.8790 |
| Test | Cyclist | 0.372 (@0.371) | 0.5380 | 0.381 / 0.461 / 0.545 | 0.5116 | 0.0912 | 0.4117 |
| Test | **Micro-All** | 0.462 (@0.130) | 0.3728 | 0.266 / 0.328 / 0.381 | — | — | — |
