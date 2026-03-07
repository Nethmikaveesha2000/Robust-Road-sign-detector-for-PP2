from ultralytics import YOLO

DATASET_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized"
SAVE_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\YOLO8"

model = YOLO("yolov8n-cls.pt")

print("YOLOv8 Classification Model Loaded")

model.train(
    data=DATASET_PATH,
    epochs=30,
    imgsz=224,
    batch=32,

    project=SAVE_PATH,
    name="YOLOv8_Classifier",

    device="cpu",   # 🔥 FIX HERE
    patience=10
)

print("Training Completed")