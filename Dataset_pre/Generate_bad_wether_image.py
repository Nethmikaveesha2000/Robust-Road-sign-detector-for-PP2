import os
import random
import cv2
import albumentations as A
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
DATASET_ROOT = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"
FOG_COUNT = 20

# fog_aug = A.RandomFog(
#     fog_coef_lower=0.2,
#     fog_coef_upper=0.4,
#     alpha_coef=0.1,
#     p=1.0
# )

# # ==========================
# # PROCESS EACH CLASS
# # ==========================
# for class_name in os.listdir(DATASET_ROOT):

#     class_path = os.path.join(DATASET_ROOT, class_name)
#     if not os.path.isdir(class_path):
#         continue

#     images = [f for f in os.listdir(class_path)
#               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     if len(images) < FOG_COUNT:
#         continue

#     selected = random.sample(images, FOG_COUNT)

#     print(f"\n🌫 Generating fog for class: {class_name}")

#     for img_name in tqdm(selected):
#         img_path = os.path.join(class_path, img_name)

#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         foggy = fog_aug(image=img_rgb)["image"]

#         new_name = img_name.rsplit(".", 1)[0] + "_fog.jpg"
#         save_path = os.path.join(class_path, new_name)

#         cv2.imwrite(save_path, cv2.cvtColor(foggy, cv2.COLOR_RGB2BGR))

# print("\n✅ Fog generation completed.")




# rain_aug = A.RandomRain(
#     slant_lower=-10,
#     slant_upper=10,
#     drop_length=15,
#     drop_width=1,
#     blur_value=3,
#     brightness_coefficient=0.9,
#     p=1.0
# )

# for class_name in os.listdir(DATASET_ROOT):

#     class_path = os.path.join(DATASET_ROOT, class_name)
#     if not os.path.isdir(class_path):
#         continue

#     images = [f for f in os.listdir(class_path)
#               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     if len(images) < 20:
#         continue

#     selected = random.sample(images, 20)

#     print(f"\n🌧 Generating rain for class: {class_name}")

#     for img_name in tqdm(selected):
#         img_path = os.path.join(class_path, img_name)

#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         rainy = rain_aug(image=img_rgb)["image"]

#         new_name = img_name.rsplit(".", 1)[0] + "_rain.jpg"
#         save_path = os.path.join(class_path, new_name)

#         cv2.imwrite(save_path, cv2.cvtColor(rainy, cv2.COLOR_RGB2BGR))

# print("\n✅ Rain generation completed.")



import numpy as np

for class_name in os.listdir(DATASET_ROOT):

    class_path = os.path.join(DATASET_ROOT, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(images) < 20:
        continue

    selected = random.sample(images, 20)

    print(f"\n🌙 Generating night for class: {class_name}")

    for img_name in tqdm(selected):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Reduce brightness
        night = img.astype(np.float32) * 0.4

        # Add slight blue tint
        night[:, :, 0] *= 1.1  # blue channel

        # Clip values
        night = np.clip(night, 0, 255).astype(np.uint8)

        # Optional slight blur
        night = cv2.GaussianBlur(night, (3, 3), 0)

        new_name = img_name.rsplit(".", 1)[0] + "_night.jpg"
        save_path = os.path.join(class_path, new_name)

        cv2.imwrite(save_path, night)

print("\n✅ Night generation completed.")
