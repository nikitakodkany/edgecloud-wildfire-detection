import os
import zipfile
import urllib.request

DATA_URL = "https://github.com/GovTechSG/fire-detection-dataset/archive/refs/heads/master.zip"
DEST_DIR = "data/raw"

def download_and_extract(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    zip_path = os.path.join(dest_folder, "dataset.zip")
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)
    print("Done! Dataset downloaded and extracted.")

if __name__ == "__main__":
    download_and_extract(DATA_URL, DEST_DIR)
