from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="ppe_model",
    project="runs",
    patience=10,
    device=0,
)

print("\nTraining complete. Best model saved to: runs/ppe_model/weights/best.pt")
