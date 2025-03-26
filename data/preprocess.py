import os
import rasterio
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define directories
RAW_DATA_DIR = 'CEMS-Wildfire-Dataset/dataOptimal'
PROCESSED_IMAGE_DIR = 'data/processed/images'
PROCESSED_MASK_DIR = 'data/processed/masks'
IMG_SIZE = (128, 128)
TEST_SIZE = 0.2
VAL_SIZE = 0.1

def preprocess_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # Adjust bands as necessary
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, IMG_SIZE)
        image = image / 10000.0  # Normalize to [0, 1] if original range is [0, 10000]
        return image.astype(np.float32)

def preprocess_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)  # Ensure binary mask
        return mask

def main():
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)

    images = []
    masks = []

    for root, _, files in os.walk(RAW_DATA_DIR):
        image_file = next((f for f in files if f.endswith('_S2L2A.tif')), None)
        mask_file = next((f for f in files if f.endswith('_DEL.tif')), None)

        if image_file and mask_file:
            img_path = os.path.join(root, image_file)
            mask_path = os.path.join(root, mask_file)

            image = preprocess_image(img_path)
            mask = preprocess_mask(mask_path)

            images.append(image)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=TEST_SIZE + VAL_SIZE, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), random_state=42)

    np.save(os.path.join(PROCESSED_IMAGE_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_IMAGE_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_IMAGE_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_MASK_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_MASK_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_MASK_DIR, 'y_test.npy'), y_test)

if __name__ == "__main__":
    main()
