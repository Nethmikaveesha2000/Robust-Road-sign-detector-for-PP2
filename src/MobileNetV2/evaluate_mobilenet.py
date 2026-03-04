import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# CONFIG
# ============================================
IMG_SIZE = 224
BATCH_SIZE = 32

VALID_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Valid"
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
# DATA GENERATOR (NO SHUFFLE)
# ============================================
valid_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_gen = valid_aug.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

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

print("✅ Model Loaded")

# ============================================
# PREDICT
# ============================================
predictions = model.predict(valid_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = valid_gen.classes

# ============================================
# ACCURACY
# ============================================
accuracy = np.sum(y_pred == y_true) / len(y_true)
print("\n===== EVALUATION RESULTS =====")
print("Accuracy:", accuracy)

# ============================================
# PRECISION / RECALL / F1
# ============================================
report = classification_report(
    y_true,
    y_pred,
    target_names=[index_to_class[i] for i in range(NUM_CLASSES)]
)

print("\nClassification Report:")
print(report)

# ============================================
# CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\n✅ Confusion matrix saved as confusion_matrix.png")