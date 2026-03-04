import os

TRAIN_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Train"
VALID_DIR = r"D:\Nethmyy__Research\Research_YOLO_model\Dataset\Resized\Valid"

# Get class folder names
train_classes = sorted([
    f for f in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, f))
])

valid_classes = sorted([
    f for f in os.listdir(VALID_DIR)
    if os.path.isdir(os.path.join(VALID_DIR, f))
])

print("Train classes:", len(train_classes))
print("Valid classes:", len(valid_classes))

# Find differences
missing_in_train = set(valid_classes) - set(train_classes)
missing_in_valid = set(train_classes) - set(valid_classes)

print("\n❌ Classes in VALID but missing in TRAIN:")
for c in missing_in_train:
    print("  ", c)

print("\n❌ Classes in TRAIN but missing in VALID:")
for c in missing_in_valid:
    print("  ", c)