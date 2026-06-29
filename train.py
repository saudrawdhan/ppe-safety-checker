from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="ppe_model_m",
    project="runs",
    patience=15,
    device=0,
)

print("\nTraining complete. Best model saved to: runs/detect/runs/ppe_model_m/weights/best.pt")
