import os
import cv2
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================
INPUT_ROOT = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"
OUTPUT_ROOT = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"

TARGET_SIZE = 224

# ============================================
# RESIZE FUNCTION
# ============================================
def resize_dataset(input_root, output_root, size):
    for root, dirs, files in os.walk(input_root):

        # Create corresponding output folder
        relative_path = os.path.relpath(root, input_root)
        save_dir = os.path.join(output_root, relative_path)

        os.makedirs(save_dir, exist_ok=True)

        for file in tqdm(files, desc=f"Processing {relative_path}"):

            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(root, file)
                output_path = os.path.join(save_dir, file)

                img = cv2.imread(input_path)

                if img is None:
                    print(f"⚠ Skipping unreadable file: {input_path}")
                    continue

                # Resize directly to 224x224
                resized = cv2.resize(img, (size, size))

                cv2.imwrite(output_path, resized)

    print("\n✅ Resizing Completed Successfully!")

# ============================================
# RUN
# ============================================
resize_dataset(INPUT_ROOT, OUTPUT_ROOT, TARGET_SIZE)








# ============================================
# CONFIG
# ============================================
# INPUT_ROOT = r"D:\Nethmyy__Research\sri lankan data set 2\dataset - recreated\Test"
# OUTPUT_ROOT = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Test"

# TARGET_SIZE = 224

# # ============================================
# # RESIZE FUNCTION
# # ============================================
# def resize_dataset(input_root, output_root, size):
#     for root, dirs, files in os.walk(input_root):

#         # Create corresponding output folder
#         relative_path = os.path.relpath(root, input_root)
#         save_dir = os.path.join(output_root, relative_path)

#         os.makedirs(save_dir, exist_ok=True)

#         for file in tqdm(files, desc=f"Processing {relative_path}"):

#             if file.lower().endswith((".png", ".jpg", ".jpeg")):
#                 input_path = os.path.join(root, file)
#                 output_path = os.path.join(save_dir, file)

#                 img = cv2.imread(input_path)

#                 if img is None:
#                     print(f"⚠ Skipping unreadable file: {input_path}")
#                     continue

#                 # Resize directly to 224x224
#                 resized = cv2.resize(img, (size, size))

#                 cv2.imwrite(output_path, resized)

#     print("\n✅ Resizing Completed Successfully!")

# # ============================================
# # RUN
# # ============================================
# resize_dataset(INPUT_ROOT, OUTPUT_ROOT, TARGET_SIZE)

# import os
# import cv2
# from tkinter import Tk
# from tkinter.filedialog import askopenfilenames
# from tqdm import tqdm

# # ============================================
# # CONFIG
# # ============================================
# OUTPUT_FOLDER = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train\Uneven Road Ahead"
# TARGET_SIZE = 224

# # ============================================
# # SELECT MULTIPLE IMAGES
# # ============================================
# Tk().withdraw()  # Hide tkinter root window
# image_paths = askopenfilenames(title="Select Images to Resize")

# if not image_paths:
#     raise ValueError("❌ No images selected.")

# print(f"Selected {len(image_paths)} images")

# # ============================================
# # CREATE OUTPUT FOLDER
# # ============================================
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # ============================================
# # PROCESS IMAGES
# # ============================================
# for image_path in tqdm(image_paths, desc="Resizing Images"):

#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"⚠ Skipping unreadable file: {image_path}")
#         continue

#     resized = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))

#     file_name = os.path.basename(image_path)
#     save_path = os.path.join(OUTPUT_FOLDER, file_name)

#     cv2.imwrite(save_path, resized)

# print("\n✅ All selected images resized successfully!")
