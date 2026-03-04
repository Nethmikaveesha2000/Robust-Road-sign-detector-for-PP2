# import cv2
# import numpy as np
# import os

# def motion_blur(image, kernel_size=25):
#     kernel = np.zeros((kernel_size, kernel_size))
#     kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
#     kernel = kernel / kernel_size
#     return cv2.filter2D(image, -1, kernel)

# img = cv2.imread(r"C:\Users\Dilshan\Desktop\Test\Screenshot 2026-03-01 205103.png")

# if img is None:
#     print("❌ Image not found.")
#     exit()

# blurred = motion_blur(img, 25)

# # Save folder
# save_folder = r"C:\Users\Dilshan\Desktop\Test\Blurry_Images"
# os.makedirs(save_folder, exist_ok=True)

# save_path = os.path.join(save_folder, "motion_blur.jpg")
# cv2.imwrite(save_path, blurred)

# print(f"✅ Saved to: {save_path}")



import os
import cv2
import albumentations as A
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
FOG_TIMES_PER_IMAGE = 1  # how many fog versions per image

fog_aug = A.RandomFog(
    fog_coef_lower=0.2,
    fog_coef_upper=0.4,
    alpha_coef=0.1,
    p=1.0
)

# ==========================
# SELECT IMAGES
# ==========================
root = tk.Tk()
root.withdraw()

image_paths = filedialog.askopenfilenames(
    title="Select Images",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_paths:
    print("❌ No images selected.")
    exit()

# ==========================
# SELECT SAVE DIRECTORY
# ==========================
save_dir = filedialog.askdirectory(title="Select Folder to Save Fog Images")

if not save_dir:
    print("❌ No save folder selected.")
    exit()

print(f"\n🌫 Generating fog for {len(image_paths)} images...\n")

# ==========================
# PROCESS IMAGES
# ==========================
for img_path in tqdm(image_paths):

    img = cv2.imread(img_path)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base_name = os.path.basename(img_path).rsplit(".", 1)[0]

    for i in range(FOG_TIMES_PER_IMAGE):
        foggy = fog_aug(image=img_rgb)["image"]

        new_name = f"{base_name}_fog_{i+1}.jpg"
        save_path = os.path.join(save_dir, new_name)

        cv2.imwrite(save_path, cv2.cvtColor(foggy, cv2.COLOR_RGB2BGR))

print("\n✅ Fog generation completed successfully.")