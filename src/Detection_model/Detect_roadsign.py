import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog
import os

# =====================================================
# LOAD YOUR CUSTOM YOLO MODEL
# =====================================================
MODEL_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\Detect_Model\RoadSignDetector_v22\weights\best.pt"
model = YOLO(MODEL_PATH)

print("✅ RoadSignDetector_v22 Loaded Successfully")

# =====================================================
# SELECT IMAGE
# =====================================================
Tk().withdraw()
FILE_PATH = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
)

if FILE_PATH == "":
    print("❌ No file selected!")
    exit()

# =====================================================
# RUN DETECTION
# =====================================================
results = model(FILE_PATH, conf=0.25)

boxes = results[0].boxes
image = cv2.imread(FILE_PATH)

if len(boxes) > 0:

    # 🔥 Select the highest confidence bounding box
    best_box = max(boxes, key=lambda box: float(box.conf[0]))

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    confidence = float(best_box.conf[0])
    class_id = int(best_box.cls[0])
    class_name = model.names[class_id]

    # Draw single best bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    label = f"{class_name} ({confidence:.2f})"
    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    print("✅ Road sign detected!")
    print("Class:", class_name)
    print("Confidence:", confidence)

else:
    print("❌ No road sign detected!")

# =====================================================
# SHOW RESULT
# =====================================================
cv2.imshow("Best Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save result
output_path = "best_detection_result.jpg"
cv2.imwrite(output_path, image)
print("📁 Saved as:", output_path)