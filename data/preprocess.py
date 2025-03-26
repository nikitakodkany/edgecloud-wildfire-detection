import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DIR = "data/raw/fire-detection-dataset-master/fire_images"
PROC_DIR = "data/processed"
IMG_SIZE = 128
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def load_and_resize_images(path, label, size=128):
    data = []
    for file in tqdm(os.listdir(path)):
        full_path = os.path.join(path, file)
        img = cv2.imread(full_path)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        data.append((img, label))
    return data

def main():
    fire_path = os.path.join(RAW_DIR)
    fire_data = load_and_resize_images(fire_path, label=1)

    # Optional: add non-fire images from another folder
    # no_fire_data = load_and_resize_images(NO_FIRE_PATH, label=0)

    full_data = fire_data  # + no_fire_data
    print(f"Loaded {len(full_data)} images")

    imgs, labels = zip(*full_data)
    imgs = np.array(imgs)
    labels = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(imgs, labels, test_size=VAL_SPLIT + TEST_SPLIT, random_state=42)
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio, random_state=42)

    for split, X, y in zip(['train', 'val', 'test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
        split_dir = os.path.join(PROC_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for i, (img, label) in enumerate(zip(X, y)):
            fname = f"{label}_{i}.jpg"
            cv2.imwrite(os.path.join(split_dir, fname), img)

    print("Preprocessing complete. Data saved to:", PROC_DIR)

if __name__ == "__main__":
    main()
