import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# CONFIG
# ============================================
IMG_SIZE = 224
BATCH_SIZE = 32

VALID_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Valid"
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
# DATA GENERATOR (NO SHUFFLE)
# ============================================
valid_aug = ImageDataGenerator(rescale=1./255)

valid_gen = valid_aug.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

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

print("✅ Custom Model 2 Loaded")

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
# CLASSIFICATION REPORT
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

plt.figure(figsize=(14, 12))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix - Custom Model 2")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_custom_model2.png")
plt.show()

print("\n✅ Confusion matrix saved as confusion_matrix_custom_model2.png")