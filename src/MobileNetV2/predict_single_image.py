import os
import json
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================
# CONFIG
# ============================================
IMG_SIZE = 224

MODEL_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\mobilenet_weights\phase2_epoch_015.weights.h5"
CLASS_MAPPING_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\mobilenet_weights\class_mapping.json"

# ============================================
# LOAD CLASS MAPPING
# ============================================
with open(CLASS_MAPPING_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
NUM_CLASSES = len(class_indices)

print("Detected classes:", NUM_CLASSES)

# ============================================
# BUILD MODEL (Same as training)
# ============================================
def build_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model

model = build_model()
model.load_weights(MODEL_PATH)

print("✅ Model Loaded Successfully")

# ============================================
# SELECT IMAGE
# ============================================
Tk().withdraw()
image_path = askopenfilename(title="Select Image for Prediction")

if not image_path:
    raise ValueError("No image selected.")

print("Selected:", image_path)

# ============================================
# PREPROCESS IMAGE (IMPORTANT)
# ============================================
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not readable.")

orig = image.copy()

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image, dtype=np.float32)
image = preprocess_input(image)  # MobileNetV2 preprocessing
image = np.expand_dims(image, axis=0)

# ============================================
# PREDICTION
# ============================================
pred = model.predict(image)[0]

predicted_index = np.argmax(pred)
confidence = float(pred[predicted_index])
predicted_class = index_to_class[predicted_index]

print("\nPredicted Class:", predicted_class)
print("Confidence:", confidence)

# Show Top 5 predictions
print("\nTop 5 Predictions:")
top5 = np.argsort(pred)[-5:][::-1]
for i in top5:
    print(index_to_class[i], float(pred[i]))

# ============================================
# SAVE RESULT IMAGE (No cv2.imshow error)
# ============================================
label_text = f"{predicted_class} ({confidence:.2f})"

cv2.putText(
    orig,
    label_text,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

output_path = "prediction_result.jpg"
cv2.imwrite(output_path, orig)

print(f"\n✅ Result saved as {output_path}")