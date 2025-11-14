from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------
# Edge-aware enhancement
# ----------------------
def enhance_edges(img, contrast_factor=2.5, gamma=1.5, sharpness_factor=2.5):
    """
    Enhance edges using contrast, unsharp mask and gamma correction.
    """
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)

    arr = np.array(img) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


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
def process_folder(
    input_folder: Path,
    output_folder: Path,
    contrast_factor=2.5,
    gamma=1.5,
    sharpness_factor=2.5,
    show_sample=False,
    max_workers=8
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Supported extensions
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_files = [f for f in input_folder.iterdir() if f.suffix.lower() in extensions]

    print(f"Found {len(img_files)} images in {input_folder}")

    # Show preview
    if show_sample and img_files:
        sample_file = random.choice(img_files)
        with Image.open(sample_file).convert("RGB") as orig_img:
            enhanced_img = enhance_edges(orig_img, contrast_factor, gamma, sharpness_factor)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_img)
        plt.title("Enhanced")
        plt.axis("off")

        plt.show()

    # Prepare job list
    jobs = [(f, output_folder / f.name) for f in img_files]

    # Run parallel processing
    print(f"Processing images with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, f, out_f, contrast_factor, gamma, sharpness_factor)
            for f, out_f in jobs
        ]
        for _ in as_completed(futures):
            pass

    print(f"Done. Enhanced images saved to: {output_folder}")


# ----------------------
# Example main
# ----------------------
def main():
    # Example usage — replace these paths anywhere you want
    folder_B = Path("INPUT_FOLDER_PATH_1")
    output_B = Path("OUTPUT_FOLDER_PATH_1")

    folder_F = Path("INPUT_FOLDER_PATH_2")
    output_F = Path("OUTPUT_FOLDER_PATH_2")

    # Brightfield
    process_folder(
        input_folder=folder_B,
        output_folder=output_B,
        contrast_factor=2.5,
        gamma=1.5,
        sharpness_factor=2.5,
        show_sample=True,
        max_workers=32
    )

    # Fluorescence
    process_folder(
        input_folder=folder_F,
        output_folder=output_F,
        contrast_factor=2.5,
        gamma=1.5,
        sharpness_factor=2.5,
        show_sample=True,
        max_workers=64
    )


if __name__ == "__main__":
    main()
