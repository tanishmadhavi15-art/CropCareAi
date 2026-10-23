"""
CropCare AI — Dataset download helper.

Downloads the PlantVillage dataset from a verified public source.
Uses the Kaggle-hosted PlantVillage dataset which is freely accessible.

Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
Original paper: Hughes & Salehe (2015), arXiv:1511.08060
License: The PlantVillage dataset is publicly available for research.
"""

import os
import sys
import zipfile
import urllib.request
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_RAW_DIR, PROJECT_ROOT


def download_plantvillage():
    """
    Download PlantVillage dataset.

    We use the tensorflow_datasets (TFDS) distribution which provides
    the dataset without requiring Kaggle authentication.
    Alternatively, if TFDS is not available, we provide manual instructions.
    """
    plant_village_dir = os.path.join(DATA_RAW_DIR, "PlantVillage")

    # Check if already downloaded
    if os.path.exists(plant_village_dir):
        subdirs = [d for d in os.listdir(plant_village_dir)
                    if os.path.isdir(os.path.join(plant_village_dir, d))]
        if len(subdirs) > 0:
            print(f"[INFO] PlantVillage dataset already exists at {plant_village_dir}")
            print(f"[INFO] Found {len(subdirs)} class directories.")
            return plant_village_dir

    # Try using tensorflow_datasets
    try:
        print("[INFO] Attempting download via tensorflow_datasets...")
        import tensorflow_datasets as tfds
        ds, info = tfds.load('plant_village', with_info=True, split='train')
        print(f"[INFO] TFDS loaded: {info.features}")
        # Save images to disk organized by label
        os.makedirs(plant_village_dir, exist_ok=True)
        label_names = info.features['label'].names
        for label_name in label_names:
            os.makedirs(os.path.join(plant_village_dir, label_name), exist_ok=True)

        count = 0
        for example in ds:
            label_idx = example['label'].numpy()
            label_name = label_names[label_idx]
            image = example['image'].numpy()
            from PIL import Image
            img = Image.fromarray(image)
            img.save(os.path.join(plant_village_dir, label_name, f"{count:06d}.jpg"))
            count += 1
            if count % 1000 == 0:
                print(f"  Saved {count} images...")

        print(f"[INFO] Downloaded {count} images via TFDS.")
        return plant_village_dir

    except ImportError:
        print("[INFO] tensorflow_datasets not available.")
    except Exception as e:
        print(f"[WARN] TFDS download failed: {e}")

    # Try direct download from a known mirror
    # The PlantVillage dataset is widely mirrored. We try the GitHub-hosted version.
    print("[INFO] Attempting direct download from public GitHub mirror...")
    GITHUB_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(DATA_RAW_DIR, "plantvillage_download.zip")
    extract_path = os.path.join(DATA_RAW_DIR, "plantvillage_extract")

    try:
        os.makedirs(DATA_RAW_DIR, exist_ok=True)

        if not os.path.exists(zip_path):
            print(f"[INFO] Downloading from {GITHUB_URL}")
            print("[INFO] This may take several minutes (~1 GB)...")
            urllib.request.urlretrieve(GITHUB_URL, zip_path, _progress_hook)
            print()  # newline after progress
        else:
            print(f"[INFO] Using existing download: {zip_path}")

        print("[INFO] Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_path)

        # The GitHub repo has structure: PlantVillage-Dataset-master/raw/color/
        color_dir = None
        for root, dirs, files in os.walk(extract_path):
            if "color" in dirs:
                color_dir = os.path.join(root, "color")
                break

        if color_dir and os.path.isdir(color_dir):
            # Move the color directory to our target location
            if os.path.exists(plant_village_dir):
                shutil.rmtree(plant_village_dir)
            shutil.copytree(color_dir, plant_village_dir)
            print(f"[INFO] Dataset extracted to {plant_village_dir}")

            # Clean up
            shutil.rmtree(extract_path, ignore_errors=True)
            return plant_village_dir
        else:
            print("[ERROR] Could not find 'color' directory in extracted archive.")
            print(f"[INFO] Extracted contents at: {extract_path}")
            # List what we got
            for root, dirs, files in os.walk(extract_path):
                level = root.replace(extract_path, '').count(os.sep)
                if level < 3:
                    indent = ' ' * 2 * level
                    print(f'{indent}{os.path.basename(root)}/')
            return None

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")

    # If all automated methods fail, provide manual instructions
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print("""
The automated download methods were not successful.
Please download the PlantVillage dataset manually:

Option 1: Kaggle (requires free account)
  1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
  2. Download the dataset ZIP
  3. Extract to: {raw_dir}/PlantVillage/
     Each subdirectory should be a class (e.g., Tomato___healthy/)

Option 2: GitHub
  1. Visit: https://github.com/spMohanty/PlantVillage-Dataset
  2. Download the repository
  3. Copy the 'raw/color/' directory contents to: {raw_dir}/PlantVillage/

After placing the data, re-run this script or the audit script.
""".format(raw_dir=DATA_RAW_DIR))
    return None


def _progress_hook(block_num, block_size, total_size):
    """Download progress callback."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {mb_down:.1f} MB")
    sys.stdout.flush()


if __name__ == "__main__":
    result = download_plantvillage()
    if result:
        print(f"\n[SUCCESS] Dataset ready at: {result}")
    else:
        print("\n[FAILED] Dataset not available. See instructions above.")
        sys.exit(1)
