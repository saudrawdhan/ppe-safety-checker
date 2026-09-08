# PPE Safety Checker

A YOLOv8 model that spots missing safety gear on construction sites — no helmet, no vest, no mask — and flags it automatically on both images and video.

**Live demo:** https://huggingface.co/spaces/SaudRaw/ppe-safety-checker — upload an image or try a sample directly in the browser, no setup required.

## What it does

Give it a photo or a video of a construction site and it draws a box around every worker and every piece of PPE it finds: green if the gear is present, red if something's missing, with an alert banner on any violation. On video it also keeps a running FPS and violation count, and writes a timestamped alert log with a cooldown so the same violation isn't logged on every single frame.

## Dataset

[Construction Site Safety](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) (Roboflow, via Kaggle) — 2,801 images, pre-split into train (2,605) / valid (114) / test (82), labeled in YOLO format across 10 classes:

```
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
Person, Safety Cone, Safety Vest, machinery, vehicle
```

The three `NO-` classes are the violations the tool cares about.

## Model

Trained a few sizes before settling on one, fine-tuned from the public `yolov8n.pt`/`yolov8m.pt`/YOLO11m weights (100 epochs, 640px):

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8n | 0.88 | 0.70 | 0.78 | 0.48 |
| YOLOv8s | 0.94 | 0.80 | 0.86 | 0.58 |
| **YOLOv8m (chosen)** | **0.91** | **0.82** | **0.88** | **0.62** |
| YOLO11m | 0.95 | 0.80 | 0.87 | 0.63 |

Went with YOLOv8m. It's not the highest on precision, but it has the best recall of the group, and recall is what matters here — missing a real violation is worse than one extra false alarm. It also beats the numbers reported on Kaggle for this same dataset (mAP50 ≈ 0.81, recall ≈ 0.73).

Breaking the results down by class, the violation classes are clearly the hardest to catch:

| Class | Recall |
|---|---|
| NO-Hardhat | 0.71 |
| NO-Mask | 0.77 |
| NO-Safety Vest | 0.79 |
| Hardhat | 0.89 |
| Safety Vest | 0.93 |
| Mask | 0.95 |

My first guess was that this is a data problem — fewer examples of "missing" gear. That's not it: NO-Safety Vest actually has more training examples than Mask, yet Mask scores much higher. The real issue is that spotting something that *isn't there* is a harder visual problem than spotting something that is. I tested test-time augmentation to see if it would help; it lowered violation recall, so it's not used (the experiment is kept in the notebook as evidence). What did help was lowering the confidence threshold specifically for the violation classes (0.25 vs. 0.40 for equipment-present classes), trading a few extra false positives for catching more of the violations that matter.

## Beyond the base task

- **Live video monitoring** — frame-by-frame detection with an FPS overlay and a cooldown-based alert log, simulating a real site-camera feed.
- **Model comparison and per-class analysis** — built into the notebook so the model choice is evidence-based, not a guess.

## Running it

```bash
pip install ultralytics opencv-python pandas matplotlib pyyaml
```

Download the dataset above and put it under `data/` so it matches the paths in `data.yaml`. Then open `PPE_Safety_Checker.ipynb` and run it top to bottom. Training takes about an hour on a GPU — if you just want to see the results, the notebook already has all outputs saved, so nothing needs retraining.

The demo video and its alert log are in `outputs/` and don't require running anything. The dataset, trained weights, and sample media aren't stored in the repo (`.gitignore`) to keep it lightweight.

## Limitations and future work

Some of the model's wrong boxes are confident mistakes rather than borderline ones, so lowering the threshold further won't fix them. Closing that gap would take either more training data for the weakest class (NO-Hardhat specifically), temporal tracking (e.g. ByteTrack) so a box has to persist across a few frames before it counts, or a two-stage person-then-PPE classifier that attacks the absence-detection problem at its root. None of these are implemented here.

## Project structure

```
PPE_Safety_Checker.ipynb   Main notebook (training, evaluation, inference, video)
train.py                   Standalone training script
data.yaml                  Dataset configuration (10 classes)
outputs/                   Annotated demo video + alert log
```
