import os
import matplotlib.pyplot as plt

# ============================================
# CONFIG
# ============================================
train_path = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"

# ============================================
# READ CLASSES
# ============================================
class_names = []
image_counts = []

for class_folder in os.listdir(train_path):

    class_path = os.path.join(train_path, class_folder)

    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    class_names.append(class_folder)
    image_counts.append(len(images))

# ============================================
# SORT BY IMAGE COUNT
# ============================================
sorted_data = sorted(zip(image_counts, class_names))
image_counts, class_names = zip(*sorted_data)

# ============================================
# PRINT SUMMARY
# ============================================
print("\n===== DATASET SUMMARY =====")
for name, count in zip(class_names, image_counts):
    print(f"{name:30s} : {count}")

print("\nTotal Classes :", len(class_names))
print("Total Images  :", sum(image_counts))

# ============================================
# PLOT
# ============================================
plt.figure(figsize=(22, 10))
plt.bar(class_names, image_counts)
plt.xticks(rotation=90)
plt.xlabel("Class Name")
plt.ylabel("Number of Images")
plt.title("Class Distribution - Training Dataset")
plt.tight_layout()
plt.show()

