import os
import random
import shutil

# ============================================
# CONFIG
# ============================================
TRAIN_ROOT = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"
VALID_ROOT = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Valid"

SPLIT_RATIO = 0.2   # 20% validation
RANDOM_SEED = 42    # reproducible split

random.seed(RANDOM_SEED)

# ============================================
# CREATE VALID ROOT
# ============================================
os.makedirs(VALID_ROOT, exist_ok=True)

# ============================================
# SPLIT PER CLASS
# ============================================
for class_name in os.listdir(TRAIN_ROOT):

    class_train_path = os.path.join(TRAIN_ROOT, class_name)

    if not os.path.isdir(class_train_path):
        continue

    class_valid_path = os.path.join(VALID_ROOT, class_name)
    os.makedirs(class_valid_path, exist_ok=True)

    images = [f for f in os.listdir(class_train_path)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(images) == 0:
        continue

    num_valid = int(len(images) * SPLIT_RATIO)

    selected = random.sample(images, num_valid)

    print(f"\n📂 Class: {class_name}")
    print(f"Total images : {len(images)}")
    print(f"Moving to Valid : {num_valid}")

    for img_name in selected:
        src_path = os.path.join(class_train_path, img_name)
        dst_path = os.path.join(class_valid_path, img_name)
        shutil.move(src_path, dst_path)

print("\n✅ 80/20 dataset split completed successfully!")
