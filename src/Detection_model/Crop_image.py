import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

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
# SETTINGS
# =====================================================
MARGIN_RATIO = 0.15   # 15% extra area

# =====================================================
# RUN DETECTION
# =====================================================
image = cv2.imread(FILE_PATH)
results = model(FILE_PATH, conf=0.25)

boxes = results[0].boxes

if len(boxes) > 0:

    # 🔥 Get best box only
    best_box = max(boxes, key=lambda box: float(box.conf[0]))

    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

    confidence = float(best_box.conf[0])
    class_id = int(best_box.cls[0])
    class_name = model.names[class_id]

    # -----------------------------
    # ADD MARGIN
    # -----------------------------
    width = x2 - x1
    height = y2 - y1

    margin_x = int(width * MARGIN_RATIO)
    margin_y = int(height * MARGIN_RATIO)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    # -----------------------------
    # CROP
    # -----------------------------
    cropped = image[y1:y2, x1:x2]

    # -----------------------------
    # DRAW DETECTION BOX ON ORIGINAL
    # -----------------------------
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        image,
        f"{class_name} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    print("✅ Road sign detected!")
    print("Class:", class_name)
    print("Confidence:", confidence)

    # =====================================================
    # DISPLAY
    # =====================================================
    cv2.imshow("Detected Image", image)
    cv2.imshow("Cropped Road Sign", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("❌ No road sign detected.")