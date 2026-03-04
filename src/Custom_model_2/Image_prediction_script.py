import os
import json
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow import keras
from tensorflow.keras import layers

# ============================================
# CONFIG
# ============================================
IMG_SIZE = 224

WEIGHT_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\Custom_model2_weights"
MODEL_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\Custom_model2_weights\epoch_026.weights.h5"
CLASS_MAPPING_PATH = os.path.join(WEIGHT_DIR, "class_mapping.json")

# ============================================
# LOAD CLASS MAPPING
# ============================================
with open(CLASS_MAPPING_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
NUM_CLASSES = len(class_indices)

print("Detected classes:", NUM_CLASSES)

# ============================================
# MODEL ARCHITECTURE (Same as training)
# ============================================

def conv_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def depthwise_block(x, filters, stride=1):
    shortcut = x

    x = layers.DepthwiseConv2D((3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (1,1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and shortcut.shape[-1] == filters:
        x = layers.Add()([x, shortcut])

    x = layers.ReLU()(x)
    return x


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, (3,3), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = conv_block(x, 64, stride=1)
    x = conv_block(x, 64, stride=1)

    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128, stride=1)

    x = depthwise_block(x, 256, stride=2)
    x = depthwise_block(x, 256, stride=1)

    x = depthwise_block(x, 512, stride=2)
    x = depthwise_block(x, 512, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model


model = build_model()
model.load_weights(MODEL_PATH)

print("✅ Custom Model 2 Loaded Successfully")

# ============================================
# SELECT IMAGE
# ============================================
Tk().withdraw()
image_path = askopenfilename(title="Select Image for Prediction")

if not image_path:
    raise ValueError("No image selected.")

print("Selected:", image_path)

# ============================================
# PREPROCESS IMAGE (MATCH TRAINING)
# ============================================
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not readable.")

orig = image.copy()

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype("float32") / 255.0   # 🔥 IMPORTANT (same as training)
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
# SAVE RESULT IMAGE
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

output_path = "prediction_result_custom_model2.jpg"
cv2.imwrite(output_path, orig)

print(f"\n✅ Result saved as {output_path}")