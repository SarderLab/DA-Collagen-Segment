import cv2
import os
import glob
import shutil
import numpy as np
from multiprocessing import Pool, cpu_count

# -------------------- CONFIGURATION --------------------
base_dir = "/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/Tiles/"
trash_base = "/blue/pinaki.sarder/narra.sr/8_DUET_Data/White_Tiles/"  # destination for white tiles
cutoff = 250         # pixel value threshold for "white"
white_limit = 0.8    # fraction of white pixels above which tile is considered white
extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
max_processes = min(cpu_count(), 16)  # adjust for node size
# -------------------------------------------------------

def is_white_image(img_path):
    """Return True if the image should be moved (too white)."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        white_fraction = np.mean(img > cutoff)
        return white_fraction > white_limit
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        return False

def process_folder(folder):
    """Process all images in one folder and move white ones to trash."""
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]
    if not image_files:
        return (os.path.basename(folder), 0, 0)

    # Analyze images in parallel
    with Pool(processes=max_processes) as pool:
        results = pool.map(is_white_image, image_files)

    # Collect white files
    move_list = [f for f, should_move in zip(image_files, results) if should_move]

    # Create folder in Trash with same name as source folder
    folder_name = os.path.basename(folder)
    dest_folder = os.path.join(trash_base, folder_name)
    os.makedirs(dest_folder, exist_ok=True)

    # Move files safely
    moved_count = 0
    for f in move_list:
        try:
            dest_path = os.path.join(dest_folder, os.path.basename(f))
            shutil.move(f, dest_path)
            moved_count += 1
        except PermissionError:
            print(f"⚠️ No permission to move: {f}")
        except Exception as e:
            print(f"⚠️ Error moving {f}: {e}")

    return (folder_name, moved_count, len(image_files))

if __name__ == "__main__":
    folders = [
        f for f in glob.glob(os.path.join(base_dir, "B_*"))
        if os.path.isdir(f)
    ]
    print(f"Found {len(folders)} folders")

    for folder in folders:
        folder_name, moved, total = process_folder(folder)
        print(f"📦 Folder {folder_name}: moved {moved}/{total} white images")
