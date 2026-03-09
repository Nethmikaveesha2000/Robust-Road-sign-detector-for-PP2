import cv2
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# ===============================
# MODEL PATH
# ===============================

MODEL_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\src\yolov8s.pt"

model = YOLO(MODEL_PATH)

# ===============================
# SELECT FILE
# ===============================

Tk().withdraw()
file_path = askopenfilename(title="Select Image or Video")

if not file_path:
    print("No file selected.")
    exit()

# ===============================
# OUTPUT PATH
# ===============================

output_dir = "detection_results"
os.makedirs(output_dir, exist_ok=True)

file_name = os.path.basename(file_path)
output_path = os.path.join(output_dir, "detected_" + file_name)

# ===============================
# IMAGE DETECTION
# ===============================

if file_path.lower().endswith((".jpg", ".jpeg", ".png")):

    results = model.predict(file_path, conf=0.25)

    frame = results[0].plot()

    cv2.imshow("Detection", frame)
    cv2.imwrite(output_path, frame)

    print("Saved:", output_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# VIDEO DETECTION
# ===============================

elif file_path.lower().endswith((".mp4", ".avi", ".mov")):

    cap = cv2.VideoCapture(file_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(frame, conf=0.25)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        cv2.imshow("Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved:", output_path)

else:
    print("Unsupported file format.")