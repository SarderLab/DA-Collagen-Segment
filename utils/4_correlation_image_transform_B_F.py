from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------
# Folder paths
# ----------------------
folder_B = Path("/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/Prediction_Input/Brighfield")
folder_F = Path("/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/Prediction_Input/Fluo")

output_root = Path("/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F Set 2/Enhanced_Pred_Input")
output_B = output_root / "Brighfield_v2"
output_F = output_root / "Fluo"

# Create output folders if they don't exist
output_B.mkdir(parents=True, exist_ok=True)
output_F.mkdir(parents=True, exist_ok=True)

# ----------------------
# Edge-aware enhancement function
# ----------------------
def enhance_edges(img, contrast_factor=2.5, gamma=1.5, sharpness_factor=2.5):
    """
    Enhance edges using contrast, unsharp mask, sharpness, and gamma correction.
    """
    # 1. Increase contrast
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    
    # 2. Apply unsharp mask (enhances edges)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # 3. Further sharpening
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
    
    # 4. Gamma correction
    img_np = np.array(img) / 255.0
    img_np = np.power(img_np, gamma)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

# ----------------------
# Process a single image
# ----------------------
def process_image(img_file, output_file, contrast_factor, gamma, sharpness_factor):
    try:
        with Image.open(img_file).convert("RGB") as img:
            enhanced_img = enhance_edges(img, contrast_factor, gamma, sharpness_factor)
            enhanced_img.save(output_file)
    except Exception as e:
        print(f"Skipped {img_file}: {e}")

# ----------------------
# Process folder in parallel
# ----------------------
def process_folder(input_folder, output_folder, contrast_factor=2.5, gamma=1.5, sharpness_factor=2.5, show_sample=False, max_workers=8):
    img_files = list(input_folder.glob("*.*"))
    
    # Show random sample
    if show_sample and img_files:
        sample_file = random.choice(img_files)
        with Image.open(sample_file).convert("RGB") as orig_img:
            enhanced_img = enhance_edges(orig_img, contrast_factor, gamma, sharpness_factor)
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(orig_img)
        plt.title("Original")
        plt.axis("off")
        
        plt.subplot(1,2,2)
        plt.imshow(enhanced_img)
        plt.title("Enhanced")
        plt.axis("off")
        plt.show()
    
    # Prepare jobs
    jobs = [(f, output_folder / f.name) for f in img_files]

    # Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, f, out_f, contrast_factor, gamma, sharpness_factor)
            for f, out_f in jobs
        ]
        for _ in as_completed(futures):
            pass  # Can add progress reporting here

# ----------------------
# Apply enhancements
# ----------------------
# Fluo images
#process_folder(folder_F, output_F, contrast_factor=2.5, gamma=1.5, sharpness_factor=2.5, show_sample=True, max_workers=64)
# Brighfield images
process_folder(folder_B, output_B, contrast_factor=2.5, gamma=1.5, sharpness_factor=2.5, show_sample=True, max_workers=32)
