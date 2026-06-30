# PPE Safety Checker — Construction Site Safety

Detects missing Personal Protective Equipment (PPE) in construction-site images and video using a fine-tuned YOLOv8 model. When a worker is missing a helmet, vest, or mask, the system flags the frame, annotates the violation, raises an alert, and logs it.

---

## AI Usage

I led this project and used AI (Claude) as a coding assistant working under my direction. I chose the Safety Checker idea, brought the dataset, created the GitHub repository, and discussed the plan with the AI before writing any code. I made every model decision, ran all the training, and decided the improvements. Below is exactly how AI was used.

**What I asked the AI to help with**
- Helping me evaluate the mini-project I chose and confirm the dataset was usable before I committed to it.
- Writing the code for the tool from my instructions.
- Explaining the trade-offs — model size, confidence threshold, augmentation — so I could make the decisions.

**What parts of the code were generated or suggested by AI**
- The training script and the inference/annotation functions.
- The model-comparison table and the per-class evaluation cells — AI suggested these analyses and I chose to include them.
- The test-time-augmentation experiment — AI suggested it and I chose to test it.
- The video-monitoring loop with the FPS overlay and alert-log cooldown — my idea, AI wrote the code.

**What I changed, decided, and fixed myself**
- Chose the project and brought the dataset; created the repository and planned the approach.
- Made every model decision — trained YOLOv8n → v8s → v8m, evaluated YOLO11m, and chose YOLOv8m for its higher recall (the priority for a safety system).
- Decided to add the improvements — the live video monitoring with the alert log.
- Tuned the violation-detection confidence threshold down to 0.25 to favour recall.
- Drove the investigation into the low violation recall and decided the response.
- Removed model weight files that were accidentally added to Git, and excluded them from the repo going forward.

**What I understand now that I did not before**
- How YOLO transfer learning and the precision/recall trade-off work in practice.
- That a bigger model is not automatically better — I proved with a comparison table that YOLOv8m beat the larger YOLO11m, and that test-time augmentation actually hurt here.
- That detecting the *absence* of equipment is intrinsically harder than detecting its presence, and that the right response can be a tuned decision threshold rather than more model.

---

## What It Does

- Runs a YOLOv8m detector on construction-site **images** and **video**.
- Flags three violations: **NO-Hardhat**, **NO-Safety Vest**, **NO-Mask**.
- Draws **red** boxes on violations and **green** boxes on present equipment, with an `ALERT` banner.
- Saves every flagged image to `flagged/`.
- On video, overlays a live FPS read-out and violation count, and writes a **timestamped alert log** (`outputs/alerts_hardhat.csv`) with a cooldown so each violation is recorded once rather than on every frame.

---

## How to Run

1. **Install dependencies**
   ```
   pip install ultralytics opencv-python pandas matplotlib pyyaml
   ```
2. **Get the dataset** — Construction Site Safety (Roboflow), available on Kaggle:
   https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
   Place the `train/`, `valid/`, `test/` folders under `data/` so the paths match `data.yaml`.
3. **Open `PPE_Safety_Checker.ipynb`** and run the cells top to bottom.
   - The training cell (Section 3) reproduces the model; it takes ~60–90 min on a GPU. To skip it, use a model you have already trained and run the inference sections directly.
4. **View results without running anything** — the annotated demo video (`outputs/monitored_hardhat.mp4`) and the alert log are included in the repo, and the notebook keeps its saved outputs.

> Note: the dataset, trained weights, and sample media are not stored in the repo (`.gitignore`) to keep it lightweight. The annotated output video is included so the results can be reviewed directly.

---

## Dataset

- **Construction Site Safety** (Roboflow), 2,801 images, YOLO format, pre-split into train (2,605) / valid (114) / test (82).
- **10 classes:** Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle.
- The three `NO-` classes are treated as safety violations.

---

## Model

Fine-tuned from the public `yolov8n.pt` / `yolov8m.pt` weights on the dataset above (100 epochs, 640 px).

| Model | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| YOLOv8n | 0.88 | 0.70 | 0.78 | 0.48 |
| YOLOv8s | 0.94 | 0.80 | 0.86 | 0.58 |
| **YOLOv8m (chosen)** | **0.91** | **0.82** | **0.88** | **0.62** |
| YOLO11m | 0.95 | 0.80 | 0.87 | 0.63 |

**YOLOv8m was chosen** because it has the highest recall and mAP50. For a safety system, recall (catching every violation) matters more than precision (avoiding false alarms) — a missed violation is the dangerous failure. This beats the public Kaggle benchmark for this dataset (mAP50 ≈ 0.81, recall ≈ 0.73).

---

## Analysis

- **Per-class performance:** the violation classes are the model's weakest — NO-Hardhat (0.71), NO-Mask (0.77), NO-Safety Vest (0.79) — while their positive counterparts are strong (Mask 0.95, Safety Vest 0.93, Hardhat 0.89).
- **It is not a data-quantity problem:** NO-Safety Vest has plenty of training data yet low recall, while Mask has little data yet high recall. The cause is intrinsic — detecting the *absence* of equipment is harder than detecting a distinctive present object.
- **Test-time augmentation was tested and rejected:** it lowered violation recall on this dataset, so it is not used (the experiment is kept in the notebook as evidence).
- **Response — recall-favoring thresholds:** violation classes are flagged at a lower confidence (0.25) than equipment-present classes (0.40), deliberately trading a little precision for higher recall on the cases that matter.

---

## Improvement

These are the extra ideas I added to make the project stronger than the core image task:

- **Live video monitoring** — frame-by-frame detection with an FPS overlay and a cooldown-based alert log, simulating a real site-camera feed.
- **Model comparison and per-class analysis** — built into the notebook so my model choice is evidence-based, not a guess.

**Future work**
- More real data for NO-Hardhat specifically.
- Temporal tracking (e.g. ByteTrack) to stabilise video boxes and suppress flickering false positives.
- A two-stage person-then-PPE classifier to attack the absence-detection problem at its root.

---

## Project Structure

```
PPE_Safety_Checker.ipynb   Main notebook (training, evaluation, inference, video)
train.py                   Standalone training script
data.yaml                  Dataset configuration (10 classes)
outputs/                   Annotated demo video + alert log
.gitignore
README.md
```
