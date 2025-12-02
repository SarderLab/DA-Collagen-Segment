import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Source and destination
SRC = "/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/Tiles"
DEST = "/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/year-one-eGFR"

os.makedirs(DEST, exist_ok=True)

# List of folders in source
folders = [f for f in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, f))]

# To store summary info
summary = {}

def copy_folder_rsync(folder_name):
    """Copy a folder into DEST/key using rsync."""
    try:
        # Extract key between _ and -
        if '_' in folder_name and '-' in folder_name:
            key = folder_name.split('_')[1].split('-')[0]
        else:
            return folder_name, None  # skip unexpected names

        src_path = os.path.join(SRC, folder_name)
        dest_key_path = os.path.join(DEST, key)
        os.makedirs(dest_key_path, exist_ok=True)
        dest_path = os.path.join(dest_key_path, folder_name)

        # rsync: -a archive, -H preserve hard links, -q quiet
        subprocess.run(["rsync", "-aH", src_path + "/", dest_path], check=True)

        # Count number of files copied
        num_files = sum(len(files) for _, _, files in os.walk(dest_path))

        return folder_name, key, num_files
    except Exception as e:
        print(f"Error copying {folder_name}: {e}")
        return folder_name, None, 0

# Use ThreadPoolExecutor for parallel copying
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(copy_folder_rsync, f) for f in folders]
    for future in as_completed(futures):
        result = future.result()
        if result[1]:  # key exists
            folder_name, key, num_files = result
            summary.setdefault(key, {"folders": 0, "files": 0})
            summary[key]["folders"] += 1
            summary[key]["files"] += num_files
            print(f"Copied {folder_name} → {key} ({num_files} files)")

# Print final summary
print("\nSummary:")
for key, info in summary.items():
    print(f"{key} → {info['folders']} folders, {info['files']} files total")
