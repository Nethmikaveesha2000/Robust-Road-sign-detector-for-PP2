from ultralytics import YOLO
import os

DATASET_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\YOLO_dataset\data.yaml"

SAVE_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\Video_weight\YOLO_weight"

os.makedirs(SAVE_PATH, exist_ok=True)

model = YOLO("yolov8n.pt")

model.train(
    data=DATASET_PATH,
    epochs=20,
    imgsz=320,
    batch=4,
    workers=1,
    device="cpu",
    project=SAVE_PATH,
    name="yolov8s_finetune",

    save=True,
    save_period=1
)

print("Training started...")