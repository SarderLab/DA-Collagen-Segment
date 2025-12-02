import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = "/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/Tiles"
num_threads = 64  # Adjust based on CPU cores

def get_images_set(folder_path):
    """Return a set of filenames in a folder."""
    return set(os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.*")))

def delete_image(img_path):
    """Delete image safely and return status."""
    try:
        os.remove(img_path)
        return f"Deleted {img_path}"
    except Exception as e:
        return f"Failed to delete {img_path}: {e}"

def sync_f_with_b(b_folder_path, f_folder_path):
    """Delete images in F that are not in B using multi-threading."""
    b_images = get_images_set(b_folder_path)
    f_images = get_images_set(f_folder_path)

    to_delete = [os.path.join(f_folder_path, img) for img in f_images - b_images]
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(delete_image, img) for img in to_delete]
        for future in as_completed(futures):
            results.append(future.result())
    return results

if __name__ == "__main__":
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("B_"):
            b_folder_path = os.path.join(base_dir, folder_name)
            val = folder_name[2:]
            f_folder_name = f"F_{val}"
            f_folder_path = os.path.join(base_dir, f_folder_name)

            if not os.path.isdir(f_folder_path):
                print(f"No corresponding F folder for {folder_name}, skipping.")
                continue

            print(f"Syncing {f_folder_path} with {b_folder_path}")
            deleted_messages = sync_f_with_b(b_folder_path, f_folder_path)
            for msg in deleted_messages:
                print(msg)
