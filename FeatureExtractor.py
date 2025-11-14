"""
Batch Collagen Feature Extraction
---------------------------------
Automatically detects and processes all B/C/F folder triplets in slide directories.

Folder structure expected:
Root/
 ├── 10H/
 │    ├── B_10H-0/
 │    ├── C_10H-0/
 │    ├── F_10H-0/
 │    ├── B_10H-1/
 │    ├── C_10H-1/
 │    ├── F_10H-1/
 │    └── ...
 ├── 11H/
 │    ├── B_11H-0/
 │    ├── C_11H-0/
 │    ├── F_11H-0/
 │    └── ...

Output:
 ├── Root/10H/Features_10H-0/
 ├── Root/10H/Features_10H-1/
 ├── Root/11H/Features_11H-0/
 └── etc.
"""

import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from skimage.filters import threshold_otsu
import plotly.express as px
import argparse
from typing import Dict
from scipy.ndimage import label

# =====================================================================
#                           FEATURE EXTRACTOR
# =====================================================================
class Feature_Extractor:
    def __init__(self,
                 bf_image_dir: str,
                 f_image_dir: str,
                 mask_dir: str,
                 output_dir: str,
                 threshold: Dict[str,float],
                 use_stitched: bool):

        self.bf_image_dir = Path(bf_image_dir)
        self.f_image_dir = Path(f_image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        # keep threshold in signature for backward compatibility (not used for BF/F/mask thresholds)
        self.threshold = threshold
        self.use_stitched = use_stitched
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect and match all image paths
        self.mask_paths, self.f_image_paths, self.bf_image_paths = self._collect_and_match_paths()

    # ------------------- Collect and match -------------------
    def _collect_and_match_paths(self):
        def collect(folder: Path):
            valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            return {p.stem: p for p in files}

        mask_files = collect(self.mask_dir)
        bf_files = collect(self.bf_image_dir)
        f_files = collect(self.f_image_dir)

        # Normalize stems (preserve original behavior)
        mask_files = {stem.replace("_prediction", "").replace("_LargeGlobalFlatfield", ""): path for stem, path in mask_files.items()}
        f_files = {stem.replace("_LargeGlobalFlatfield", ""): path for stem, path in f_files.items()}
        bf_files = {stem.replace(".sci", ""): path for stem, path in bf_files.items()}

        common_stems = sorted(set(mask_files.keys()) & set(f_files.keys()) & set(bf_files.keys()))
        if not common_stems:
            raise ValueError(f"No matching filenames found in {self.mask_dir}")

        mask_paths = [mask_files[s] for s in common_stems]
        f_image_paths = [f_files[s] for s in common_stems]
        bf_image_paths = [bf_files[s] for s in common_stems]

        print(f"[INFO] Found {len(mask_paths)} matched images in {self.mask_dir.name}")
        return mask_paths, f_image_paths, bf_image_paths

    # ------------------- Main runner -------------------
    def run(self):
        if self.use_stitched:
            self._run_stitched()
        else:
            self._run_patch_based()

    # ------------------- Patch-based processing -------------------
    def _run_patch_based(self):
        if not self.mask_paths:
            print(f"No images found in: {self.mask_dir}")
            return
        
        self.output_file_path = self.output_dir / "Patch_Collagen_Features.csv"

        all_feature_list = []
        for img_idx, img in tqdm(enumerate(self.mask_paths), total=len(self.mask_paths)):
            og_pred_image = np.array(Image.open(img))
            bin_image = self.binarize(og_pred_image)

            if np.sum(bin_image) > 0:
                # BF image: invert and convert to grayscale
                bf_image = 255 - np.array(Image.open(self.bf_image_paths[img_idx])).astype(np.float32)
                if bf_image.ndim == 3:
                    bf_image = np.mean(bf_image, axis=-1)

                # Fluorescence image: keep as grayscale
                f_image = np.array(Image.open(self.f_image_paths[img_idx])).astype(np.float32)
                if f_image.ndim == 3:
                    f_image = np.mean(f_image, axis=-1)

                # Ensure bin_image is 2D
                if bin_image.ndim == 3:
                    bin_image = bin_image[:, :, 0]

                # Multiply bin mask -> these are uint8 as before
                masked_bf_image = np.uint8(bin_image * bf_image)
                masked_f_image = np.uint8(bin_image * f_image)


                glob_features = self.global_features(bin_image)
                dt_features = self.distance_transform_features(bin_image)

                # intensity and texture functions now compute adaptive Otsu per-image internally
                bf_features = {**self.intensity_features(masked_bf_image),
                               **self.texture_features(masked_bf_image)}
                f_features = {**self.intensity_features(masked_f_image),
                              **self.texture_features(masked_f_image)}

                # prefix feature names as original
                bf_features = {f'BF {k}': v for k, v in bf_features.items()}
                f_features = {f'F {k}': v for k, v in f_features.items()}
                
                image_features = {**glob_features, **dt_features, **bf_features, **f_features}
                image_features['Image Names'] = img.name

                all_feature_list.append(image_features)

        if all_feature_list:
            feature_df = pd.DataFrame.from_records(all_feature_list)
            feature_df.to_csv(self.output_file_path, index=False)
            #self.plot_feature(feature_df)

    # ------------------- Stitched processing -------------------
    def _run_stitched(self):                
        if not self.mask_dir.exists():
            print(f"No directory named: {self.mask_dir}")
            return

        og_pred_image, og_bf_image, og_f_image = self.stitch_patches()

        bin_image = self.binarize(og_pred_image)
        if np.sum(bin_image) == 0:
            return

        bf_image = 255 - og_bf_image.astype(np.float32)
        if bf_image.ndim == 3:
            bf_image = np.mean(bf_image, axis=-1)

        # F image: keep as grayscale
        f_image = og_f_image.astype(np.float32)
        if f_image.ndim == 3:
            f_image = np.mean(f_image, axis=-1)

        # Ensure bin_image is 2D
        if bin_image.ndim == 3:
            bin_image = bin_image[:, :, 0]

        # Create masked BF/F images for stitched case (same logic as patch)
        masked_bf_image = np.uint8(bin_image * bf_image)
        masked_f_image = np.uint8(bin_image * f_image)

        glob_features = self.global_features(bin_image)
        dt_features = self.distance_transform_features(bin_image)

        bf_features = {**self.intensity_features(masked_bf_image, image_key='BF'),
                       **self.texture_features(masked_bf_image, image_key='BF')}
        f_features = {**self.intensity_features(masked_f_image, image_key='F'),
                      **self.texture_features(masked_f_image, image_key='F')}

        bf_features = {f'BF {k}': v for k, v in bf_features.items()}
        f_features = {f'F {k}': v for k, v in f_features.items()}

        image_features = {**glob_features, **dt_features, **bf_features, **f_features}
        image_features['Image Names'] = self.output_dir.parts[-1]

        image_features_json = {k: float(v) for k, v in image_features.items() if k != 'Image Names'}
        with open(self.output_dir / "Stitched_Collagen_Features.json", "w") as f:
            json.dump(image_features_json, f)
        
        feature_df = pd.DataFrame.from_dict(image_features, orient='index').reset_index()
        feature_df.to_csv(self.output_dir / 'Stitched_Collagen_Features.csv', index=False)

    # ------------------- Binarization -------------------
    def binarize(self, image):
        """
        Adaptive Otsu binarization for the mask image.
        Keeps function name/signature identical; ignores self.threshold for mask.
        """
        # ensure uint8 grayscale
        if image.ndim == 3:
            img_gray = np.mean(image, axis=-1)
        else:
            img_gray = image.copy().astype(np.float32)

        # scale to 0-255 if image in 0-1
        if img_gray.max() <= 1.0:
            img_gray = (img_gray * 255).astype(np.uint8)
        else:
            img_gray = img_gray.astype(np.uint8)

        try:
            thresh = threshold_otsu(img_gray)
        except Exception:
            # fallback to simple >0 if otsu fails
            return (img_gray > 0).astype(np.uint8)

        bin_img = (img_gray >= thresh).astype(np.uint8)
        return bin_img

    # ------------------- Global features -------------------
    def global_features(self, bin_image):
        """
        Compute global features from a binary collagen mask.
        Returns:
            - Total Patch Area
            - Tissue Area (non-collagen tissue)
            - Empty Area
            - Collagen Area
            - Collagen Area Ratio
        """
        total_patch_area = bin_image.shape[0] * bin_image.shape[1]
        collagen_area = np.sum(bin_image)

        if collagen_area > 0:
            # Fill holes to estimate tissue area including collagen
            filled_area = np.sum(binary_fill_holes(bin_image))

            # Tissue area excluding collagen pixels
            tissue_area = filled_area - collagen_area

            # Empty area = remaining pixels
            empty_area = total_patch_area - (collagen_area + tissue_area)

            # Collagen ratio relative to tissue + collagen
            area_ratio = collagen_area / (collagen_area + tissue_area)
        else:
            # No collagen detected
            tissue_area = 0
            empty_area = total_patch_area
            area_ratio = 0.0

        return {
            'Total Patch Area': total_patch_area,
            'Tissue Area': tissue_area,
            'Empty Area': empty_area,
            'Collagen Area': collagen_area,
            'Collagen Area Ratio': area_ratio
        }


    # ------------------- Distance transform features -------------------
    def distance_transform_features(self, bin_image):
        """
        Computes distance transform–based features from a binarized collagen image.

        Parameters:
            bin_image (np.ndarray): Binarized (uint8) collagen mask image.
                                   Foreground should be >0 (e.g., 255) for collagen regions.

        Returns:
            dict: Dictionary containing distance transform statistics and counts above thresholds.
        """

        feature_dict = {}

        # Handle empty or invalid masks
        if bin_image is None or np.sum(bin_image) == 0:
            # Return zeros for all features if no collagen pixels
            feature_names = [
                'Sum Distance Transform', 'Mean Distance Transform', 'Max Distance Transform',
                'Standard Deviation Distance Transform', 'Median Distance Transform'
            ]
            for name in feature_names:
                feature_dict[name] = 0.0

            pixel_steps = [5 * i for i in range(0, 11)]
            for p in pixel_steps:
                feature_dict[f'Distance Transform Pixel Count > {p}'] = 0
                feature_dict[f'Distance Transform ROI Count > {p}'] = 0

            return feature_dict

        # Compute distance transform using Euclidean (L2) distance
        dist_transform_img = cv2.distanceTransform(bin_image, cv2.DIST_L2, 5)

        # Set background (0 distance) pixels to NaN for statistical calculations
        dist_transform_img[dist_transform_img == 0] = np.nan

        # Basic statistics with safety for NaN handling
        feature_dict['Sum Distance Transform'] = np.nansum(dist_transform_img)
        feature_dict['Mean Distance Transform'] = np.nanmean(dist_transform_img)
        feature_dict['Max Distance Transform'] = np.nanmax(dist_transform_img)
        feature_dict['Standard Deviation Distance Transform'] = np.nanstd(dist_transform_img)
        feature_dict['Median Distance Transform'] = np.nanmedian(dist_transform_img)

        # Pixel & ROI counts for increasing distance thresholds
        pixel_steps = [5 * i for i in range(0, 11)]  # 0, 5, 10, ..., 50
        for p in pixel_steps:
            valid_mask = np.nan_to_num(dist_transform_img) > p
            pixel_count = np.sum(valid_mask)
            _, roi_count = label(valid_mask)

            feature_dict[f'Distance Transform Pixel Count > {p}'] = pixel_count
            feature_dict[f'Distance Transform ROI Count > {p}'] = roi_count

        return feature_dict

    # ------------------- Intensity features -------------------
    def intensity_features(self, gray_image, image_key='BF'):
        """
        Adaptive intensity features:
        - Compute Otsu threshold on the provided gray_image (if possible) and mask-out values below threshold.
        - Returns same-named keys as original ('Mean Intensity', 'Median Intensity', etc.)
        """
        img = gray_image.copy().astype(float)

        # if image is all zeros -> return nans/zeros
        if np.count_nonzero(img) == 0:
            return {
                'Mean Intensity': np.nan,
                'Median Intensity': np.nan,
                'Standard Deviation Intensity': np.nan,
                'Maximum Intensity': np.nan
            }

        # compute per-image Otsu threshold (note: expects uint8)
        try:
            img_for_thresh = img
            if img_for_thresh.max() <= 1.0:
                img_for_thresh = (img_for_thresh * 255).astype(np.uint8)
            else:
                img_for_thresh = np.clip(img_for_thresh, 0, 255).astype(np.uint8)
            otsu_thresh = threshold_otsu(img_for_thresh)
            # apply threshold on float image using otsu_thresh scale 0-255
            img[img < otsu_thresh] = np.nan
        except Exception:
            # fallback: do not threshold if Otsu fails
            pass

        return {
            'Mean Intensity': np.nanmean(img),
            'Median Intensity': np.nanmedian(img),
            'Standard Deviation Intensity': np.nanstd(img),
            'Maximum Intensity': np.nanmax(img)
        }

    # ------------------- Texture features -------------------
    def texture_features(self, gray_image, image_key='BF'):
        """
        Adaptive texture features:
        - Compute Otsu threshold to zero out low-intensity pixels
        - Convert to uint8 and call graycomatrix / graycoprops
        """
        texture_image = gray_image.copy().astype(float)

        # if empty -> zeros for features
        if np.count_nonzero(texture_image) == 0:
            return {
                'Contrast': 0.0,
                'Homogeneity': 0.0,
                'Correlation': 0.0,
                'Energy': 0.0
            }

        try:
            # prepare image for otsu
            img_for_thresh = texture_image
            if img_for_thresh.max() <= 1.0:
                img_for_thresh = (img_for_thresh * 255).astype(np.uint8)
            else:
                img_for_thresh = np.clip(img_for_thresh, 0, 255).astype(np.uint8)
            otsu_thresh = threshold_otsu(img_for_thresh)
            # zero out below threshold
            texture_image[texture_image < otsu_thresh] = 0
        except Exception:
            # if otsu fails, proceed without thresholding
            pass

        # convert to uint8 (graycomatrix requires integer types)
        texture_image = np.clip(texture_image, 0, 255).astype(np.uint8)

        if np.all(texture_image == 0):
            return {
                'Contrast': 0.0,
                'Homogeneity': 0.0,
                'Correlation': 0.0,
                'Energy': 0.0
            }

        # build GLCM and compute properties
        # handle any small images by ensuring levels <=256 (we keep 256)
        glcm = graycomatrix(texture_image, [1], [0], levels=256, symmetric=True, normed=True)
        return {
            'Contrast': graycoprops(glcm, 'contrast')[0, 0],
            'Homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'Correlation': graycoprops(glcm, 'correlation')[0, 0],
            'Energy': graycoprops(glcm, 'energy')[0, 0]
        }

    # ------------------- Violin plots -------------------
    def plot_feature(self, feature_df):
        for feat in feature_df.columns:
            if feat == 'Image Names':
                continue
            fig = px.violin(feature_df, y=feat)
            fig.update_layout(title=f'Violin Plot of {feat}', yaxis_title=feat)
            fig.write_image(self.output_dir / f'{feat}.png')

    # ------------------- Patch coordinates for stitching -------------------
    def get_patch_coordinates(self, patch_name):
        ext = patch_name.split('.')[-1]
        patch_name = patch_name.replace(ext, "")
        parts = patch_name.split("_")[1].split(' ')
        try:
            x_coord = int(parts[-1].split("X")[-1])
        except ValueError:
            x_coord = 0
        try:
            y_coord = int(parts[-2].split("Y")[-1])
        except ValueError:
            y_coord = 0
        return y_coord, x_coord

    # ------------------- Stitch patches -------------------
    def stitch_patches(self, downsample=10, patch_overlap=150):
        checked_mask_paths, checked_f_paths, checked_bf_paths = [], [], []
        patch_shape = None
        for i, mask_path in enumerate(self.mask_paths):
            patch_img = np.array(Image.open(mask_path))
            patch_shape = patch_img.shape
            if np.sum(patch_img) > 0:
                checked_mask_paths.append(mask_path)
                checked_f_paths.append(self.f_image_paths[i])
                checked_bf_paths.append(self.bf_image_paths[i])
        if not checked_mask_paths:
            raise ValueError("No non-empty mask patches found!")

        x_coords, y_coords = [], []
        for mask_path in checked_mask_paths:
            key = Path(mask_path).stem.replace("_prediction", "").replace("_LargeGlobalFlatfield", "").replace("Test_Example_", "")
            y_coord, x_coord = self.get_patch_coordinates(key)
            x_coords.append(x_coord)
            y_coords.append(y_coord)
        y_coords = [i - min(y_coords) for i in y_coords]
        x_coords = [i - min(x_coords) for i in x_coords]

        max_width, max_height = max(x_coords) + 1, max(y_coords) + 1
        pixel_width = (max_width * patch_shape[1]) - ((max_width - 1) * patch_overlap)
        pixel_height = (max_height * patch_shape[0]) - ((max_height - 1) * patch_overlap)

        stitched_downsampled_mask = np.zeros((pixel_height // downsample, pixel_width // downsample), dtype=np.uint8)
        stitched_downsampled_bf = np.zeros((*stitched_downsampled_mask.shape, 3), dtype=np.uint8)
        stitched_downsampled_f = np.zeros_like(stitched_downsampled_bf)
        overlap_downsampled = np.zeros_like(stitched_downsampled_mask)

        with tqdm(zip(y_coords, x_coords, checked_mask_paths, checked_bf_paths, checked_f_paths), total=len(checked_mask_paths)) as pbar:
            for y, x, mask_path, bf_path, f_path in pbar:
                pbar.set_description(f"Working on patch: {Path(mask_path).name}")

                checked_patch = np.array(Image.open(mask_path))[0:-(patch_overlap + 1), 0:-(patch_overlap + 1)]
                resized_patch = resize(checked_patch, (checked_patch.shape[0] // downsample, checked_patch.shape[1] // downsample))

                checked_bf = np.array(Image.open(bf_path))[0:-(patch_overlap + 1), 0:-(patch_overlap + 1)]
                resized_bf = resize(checked_bf, (checked_patch.shape[0] // downsample, checked_patch.shape[1] // downsample, 3))

                checked_f = np.array(Image.open(f_path))[0:-(patch_overlap + 1), 0:-(patch_overlap + 1)]
                resized_f = resize(checked_f, (checked_patch.shape[0] // downsample, checked_patch.shape[1] // downsample, 3))

                y_start, x_start = int(y * resized_patch.shape[0]), int(x * resized_patch.shape[1])

                stitched_downsampled_mask[y_start:y_start+resized_patch.shape[0], x_start:x_start+resized_patch.shape[1]] += 255 * resized_patch
                stitched_downsampled_bf[y_start:y_start+resized_patch.shape[0], x_start:x_start+resized_patch.shape[1], :] += 255 * resized_bf
                stitched_downsampled_f[y_start:y_start+resized_patch.shape[0], x_start:x_start+resized_patch.shape[1], :] += 255 * resized_f
                overlap_downsampled[y_start:y_start+resized_patch.shape[0], x_start:x_start+resized_patch.shape[1]] += 1

        overlap_downsampled = np.maximum(overlap_downsampled, 1)
        stitched_downsampled_mask = (stitched_downsampled_mask / overlap_downsampled).astype(np.uint8)
        stitched_downsampled_bf = (stitched_downsampled_bf / overlap_downsampled[..., None]).astype(np.uint8)
        stitched_downsampled_f = (stitched_downsampled_f / overlap_downsampled[..., None]).astype(np.uint8)

        slide_name = self.output_dir.parts[-1]
        Image.fromarray(stitched_downsampled_mask).save(self.output_dir / f"{slide_name}_Stitched_Output.tif")
        Image.fromarray(stitched_downsampled_bf).save(self.output_dir / f"{slide_name}_Stitched_BF.tif")
        Image.fromarray(stitched_downsampled_f).save(self.output_dir / f"{slide_name}_Stitched_F.tif")

        return stitched_downsampled_mask, stitched_downsampled_bf, stitched_downsampled_f

# =====================================================================
#                    BATCH PROCESSING LOGIC
# =====================================================================
def find_bcf_sets(slide_dir: Path):
    subdirs = [d for d in slide_dir.iterdir() if d.is_dir()]
    b_dirs = {d.name.split("_")[1]: d for d in subdirs if d.name.startswith("B_")}
    c_dirs = {d.name.split("_")[1]: d for d in subdirs if d.name.startswith("C_")}
    f_dirs = {d.name.split("_")[1]: d for d in subdirs if d.name.startswith("F_")}
    common_keys = sorted(set(b_dirs.keys()) & set(c_dirs.keys()) & set(f_dirs.keys()))
    sets = []
    for key in common_keys:
        sets.append({"key": key, "B": b_dirs[key], "C": c_dirs[key], "F": f_dirs[key]})
    return sets

def main(args):
    root_dir = Path(args.root_dir)
    slide_folders = [d for d in root_dir.iterdir() if d.is_dir()]

    for slide_dir in slide_folders:
        print(f"\n================ Processing Slide: {slide_dir.name} ================")
        bcf_sets = find_bcf_sets(slide_dir)
        if not bcf_sets:
            print(f"[WARNING] No valid B/C/F sets found in {slide_dir}")
            continue
        for s in bcf_sets:
            print(f"\n--- Processing Set: {s['key']} ---")
            output_dir = slide_dir / f"Features_{s['key']}"
            output_dir.mkdir(parents=True, exist_ok=True)
            threshold = getattr(args, "threshold", {"BF": None, "F": None})
            feat_extractor = Feature_Extractor(
                bf_image_dir=s["B"],
                f_image_dir=s["F"],
                mask_dir=s["C"],
                output_dir=output_dir,
                threshold=threshold,
                use_stitched=args.use_stitched
            )
            start_time = time.time()
            feat_extractor.run()
            elapsed = time.time() - start_time
            print(f"[INFO] Finished {s['key']} in {int(elapsed//60)}:{int(elapsed%60)} min")

# =====================================================================
#                            ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch Collagen Feature Extraction")
    parser.add_argument('--config', type=str, default="./Config/feat_extract.json", help='Path to config JSON')
    args = parser.parse_args()
    with open(args.config) as f:
        data = json.load(f)
    args = argparse.Namespace(**data)
    main(args)
