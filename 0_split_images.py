import os
import shutil
import random

# Paths to the dataset
data_dir = 'data_raw/'
output_dir = 'data/'

# Create train/val/test directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Split ratio (adjust these values as needed)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create function to make directories
def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Make train/val/test directories for each class
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        make_dirs(os.path.join(train_dir, class_name))
        make_dirs(os.path.join(val_dir, class_name))
        make_dirs(os.path.join(test_dir, class_name))

        # Get all images in the current class folder
        images = os.listdir(class_path)
        random.shuffle(images)

        # Calculate split indices
        train_split_idx = int(len(images) * train_ratio)
        val_split_idx = int(len(images) * (train_ratio + val_ratio))

        # Split images
        train_images = images[:train_split_idx]
        val_images = images[train_split_idx:val_split_idx]
        test_images = images[val_split_idx:]

        # Copy images to their respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("Data has been successfully split into train, val, and test sets!")
