import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ============================================
# CONFIG
# ============================================
MODEL_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\epoch_030.weights.h5"
CLASS_MAPPING_PATH = r"D:\Nethmyy__Research\Research_YOLO_model\Weight\class_mapping.json"
IMG_SIZE = 224

# ============================================
# LOAD CLASS MAPPING
# ============================================
with open(CLASS_MAPPING_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
NUM_CLASSES = len(class_indices)

print("Detected classes:", NUM_CLASSES)

# ============================================
# BUILD MODEL (same architecture as training)
# ============================================
def build_model(num_classes):

    model = keras.models.Sequential([

        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',
                            input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.GlobalAveragePooling2D(),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# ============================================
# LOAD MODEL
# ============================================
model = build_model(NUM_CLASSES)
model.load_weights(MODEL_PATH)
print("✅ Model Loaded Successfully")

# ============================================
# SELECT IMAGE
# ============================================
Tk().withdraw()
image_path = askopenfilename(title="Select Image to Test")

if not image_path:
    raise ValueError("No image selected")

print("Selected:", image_path)

# ============================================
# PREPROCESS IMAGE (FIXED VERSION)
# ============================================
# Use Keras loader (same as training pipeline)
img = keras_image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = keras_image.img_to_array(img)

# Normalize (same as ImageDataGenerator rescale=1./255)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ============================================
# PREDICT
# ============================================
pred = model.predict(img_array)[0]

predicted_index = np.argmax(pred)
confidence = float(pred[predicted_index])
predicted_class = index_to_class[predicted_index]

print("\nPredicted Index:", predicted_index)
print("Predicted Class:", predicted_class)
print("Confidence:", confidence)

# 🔍 Debug: Show Top 5
print("\nTop 5 Predictions:")
top5 = np.argsort(pred)[-5:][::-1]
for i in top5:
    print(index_to_class[i], float(pred[i]))

# ============================================
# DISPLAY RESULT (using OpenCV only for display)
# ============================================
orig = cv2.imread(image_path)

label_text = f"{predicted_class} ({confidence:.2f})"

cv2.putText(orig, label_text, (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,255,0), 2)

cv2.imwrite("prediction_result.jpg", orig)
print("\nSaved result as prediction_result.jpg")