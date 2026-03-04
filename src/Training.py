# import os
# import glob
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# from tqdm.keras import TqdmCallback

# # ============================================
# # CONFIG
# # ============================================
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# CHANNELS = 3
# NUM_CLASSES = 70
# EPOCHS = 30
# BATCH_SIZE = 32
# LR = 0.001

# WEIGHT_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Weight"
# os.makedirs(WEIGHT_DIR, exist_ok=True)

# TRAIN_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"
# VALID_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Valid"

# # ============================================
# # MODEL ARCHITECTURE
# # ============================================
# def build_model():
#     model = keras.models.Sequential([

#         layers.Conv2D(32, (3,3), activation='relu', padding='same',
#                       input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(2,2),

#         layers.Conv2D(64, (3,3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(2,2),

#         layers.Conv2D(128, (3,3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(2,2),

#         layers.Conv2D(256, (3,3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(2,2),

#         layers.GlobalAveragePooling2D(),

#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.5),

#         layers.Dense(NUM_CLASSES, activation='softmax')
#     ])

#     return model

# model = build_model()

# # ============================================
# # OPTIMIZER
# # ============================================
# opt = Adam(learning_rate=LR)
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=opt,
#     metrics=['accuracy']
# )

# # ============================================
# # DATA AUGMENTATION
# # ============================================
# train_aug = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     zoom_range=0.15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.15,
#     horizontal_flip=False,
#     fill_mode="nearest"
# )

# valid_aug = ImageDataGenerator(rescale=1./255)

# train_gen = train_aug.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# valid_gen = valid_aug.flow_from_directory(
#     VALID_DIR,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # ============================================
# # RESUME TRAINING
# # ============================================
# def find_latest_checkpoint():
#     checkpoints = glob.glob(os.path.join(WEIGHT_DIR, "epoch_*.h5"))
#     if not checkpoints:
#         return None, 0

#     latest = max(checkpoints, key=os.path.getctime)
#     epoch_number = int(os.path.basename(latest).split("_")[1].split(".")[0])
#     return latest, epoch_number

# checkpoint_path, start_epoch = find_latest_checkpoint()

# if checkpoint_path:
#     print(f"🔁 Resuming from {checkpoint_path}")
#     model.load_weights(checkpoint_path)
# else:
#     print("🆕 Starting fresh training")
#     start_epoch = 0

# # ============================================
# # CALLBACKS
# # ============================================
# checkpoint_callback = ModelCheckpoint(
#     os.path.join(WEIGHT_DIR, "epoch_{epoch:03d}.weights.h5"),
#     save_weights_only=True,
#     save_freq='epoch'
# )


# csv_logger = CSVLogger(os.path.join(WEIGHT_DIR, "training_log.csv"), append=True)

# # ============================================
# # TRAIN
# # ============================================
# history = model.fit(
#     train_gen,
#     validation_data=valid_gen,
#     epochs=EPOCHS,
#     initial_epoch=start_epoch,
#     callbacks=[
#         checkpoint_callback,
#         csv_logger,
#         TqdmCallback(verbose=1)  # progress bar
#     ]
# )

# print("✅ Training Completed")





import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tqdm.keras import TqdmCallback

# ============================================
# CONFIG
# ============================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
NUM_CLASSES = 70
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

WEIGHT_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Weight"
os.makedirs(WEIGHT_DIR, exist_ok=True)

TRAIN_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"
VALID_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Valid"

# ============================================
# MODEL ARCHITECTURE
# ============================================
def build_model():
    model = keras.models.Sequential([

        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

model = build_model()

# ============================================
# OPTIMIZER
# ============================================
opt = Adam(learning_rate=LR)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# ============================================
# DATA AUGMENTATION
# ============================================
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

valid_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 🔥 PRINT AND SAVE CLASS MAPPING (VERY IMPORTANT)
print("\n==============================")
print("Class Indices Mapping:")
print(train_gen.class_indices)
print("==============================\n")

with open(os.path.join(WEIGHT_DIR, "class_mapping.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

valid_gen = valid_aug.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ============================================
# RESUME TRAINING
# ============================================
def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(WEIGHT_DIR, "epoch_*.weights.h5"))
    if not checkpoints:
        return None, 0

    latest = max(checkpoints, key=os.path.getctime)
    epoch_number = int(os.path.basename(latest).split("_")[1].split(".")[0])
    return latest, epoch_number

checkpoint_path, start_epoch = find_latest_checkpoint()

if checkpoint_path:
    print(f"🔁 Resuming from {checkpoint_path}")
    model.load_weights(checkpoint_path)
else:
    print("🆕 Starting fresh training")
    start_epoch = 0

# ============================================
# CALLBACKS
# ============================================
checkpoint_callback = ModelCheckpoint(
    os.path.join(WEIGHT_DIR, "epoch_{epoch:03d}.weights.h5"),
    save_weights_only=True,
    save_freq='epoch'
)

csv_logger = CSVLogger(
    os.path.join(WEIGHT_DIR, "training_log.csv"),
    append=True
)

# ============================================
# TRAIN
# ============================================
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS,
    initial_epoch=start_epoch,
    callbacks=[
        checkpoint_callback,
        csv_logger,
        TqdmCallback(verbose=1)
    ]
)

print("✅ Training Completed")