"""

Extracting collagen features in each predicted patch

Features:
    - Binarized:
        - Total collagen area(thresholded (thresheld?))
        - Total not-collagen area (same threshold)
        - Area ratio collagen
        - Distance Transform:
            - Mean
            - Median
            - Standard Deviation
            - Maximum
            - Minimum
    - Continuous:
        - Channel-wise? Grayscale?
            - Mean intensity
            - Median intensity
            - Std. Dev intensity
            - Max intensity
            - Texture

"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import argparse
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import binary_fill_holes
from PIL import Image
from tqdm import tqdm
import plotly.express as px
import json
from skimage.transform import resize
from typing import Dict
import time

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
        self.threshold = threshold
        self.use_stitched = use_stitched
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect and match all image paths
        self.mask_paths, self.f_image_paths, self.bf_image_paths = self._collect_and_match_paths()

    def _collect_and_match_paths(self):
        """
        Collects all images from mask, bf_image, and f_image folders,
        matches them by stem (filename without extension).
        """       

        def collect(folder: Path):
            valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            print(f'Collecting images from: {folder}')
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            # print(f'Found {len(files)} files in {folder}')
            return {p.stem: p for p in files}


        mask_files = collect(self.mask_dir)
        bf_files = collect(self.bf_image_dir)
        f_files = collect(self.f_image_dir)
        # print(f'Found {len(mask_files)} mask files, {len(f_files)} fluorescence files, and {len(bf_files)} brightfield files.')
        # Adjust stems by removing "Test_Example_" prefix
        mask_files = {
            stem.replace("Test_Example_", "").replace("_prediction", "").replace("_LargeGlobalFlatfield.tif", ""): path
            for stem, path in mask_files.items()
            if any(sub in stem for sub in ["Test_Example_", "_prediction", "_LargeGlobalFlatfield.tif"])
        }
        # Adjust stems by removing "_LargeGlobalFlatfield.tif" inside
        f_files = {
            stem.replace("_LargeGlobalFlatfield.tif", ""): path
            for stem, path in f_files.items()
            if "_LargeGlobalFlatfield.tif" in stem
        }                
        # Adjust stems by removing ".sci" inside
        bf_files = {
            stem.replace(".sci", ""): path
            for stem, path in bf_files.items()
            if ".sci" in stem
        }
        # print(f'Found {len(mask_files)} mask files, {len(f_files)} fluorescence files, and {len(bf_files)} brightfield files.')
        # Find common filenames across all three sets
        common_stems = set(mask_files.keys()) & set(f_files.keys()) & set(bf_files.keys())

        if not common_stems:
            raise ValueError("No matching filenames found across mask, fluorescence, and brightfield image dirs")

        # Sort consistently by stem
        sorted_stems = sorted(common_stems)

        mask_paths = [mask_files[s] for s in sorted_stems]
        f_image_paths = [f_files[s] for s in sorted_stems]
        bf_image_paths = [bf_files[s] for s in sorted_stems]
        
        print(f'--------------On: {self.mask_dir.parts[-3]} -------------------------')
        print(f'------------------Found: {len(mask_paths)} Images!---------------------')

        return mask_paths, f_image_paths, bf_image_paths

    def run(self):
        """
        Main pipeline runner. Uses patch-based or stitched strategy.
        """
        if self.use_stitched:
            self._run_stitched()
        else:
            self._run_patch_based()

    def _run_patch_based(self):
        if not self.mask_paths:
            print(f"No images found in: {self.mask_dir}")
            return
        
        self.output_file_path = self.output_dir / "Patch_Collagen_Features.csv"

        all_feature_list = []
        for img_idx, img in tqdm(enumerate(self.mask_paths), total=len(self.mask_paths)):
            # Reading the image:
            og_pred_image = np.array(Image.open(img))
            bin_image = self.binarize(og_pred_image)

            if np.sum(bin_image) > 0:
                bf_image = np.mean(255 - np.array(Image.open(self.bf_image_paths[img_idx])), axis=-1)
                f_image = np.mean(np.array(Image.open(self.f_image_paths[img_idx])), axis=-1)

                masked_bf_image = np.uint8(bin_image * bf_image)
                masked_f_image = np.uint8(bin_image * f_image)

                glob_features = self.global_features(bin_image)
                dt_features = self.distance_transform_features(bin_image)

                bf_features = {**self.intensity_features(masked_bf_image),
                               **self.texture_features(masked_bf_image)}
                f_features = {**self.intensity_features(masked_f_image),
                              **self.texture_features(masked_f_image)}

                # Prefix keys: Renaming keys in each set of dictionaries
                bf_features = {f'BF {k}': v for k, v in bf_features.items()}
                f_features = {f'F {k}': v for k, v in f_features.items()}
                
                # Merging dictionaries (Python ≥3.5)
                image_features = {**glob_features, **dt_features, **bf_features, **f_features}
                image_features['Image Names'] = img.name

                all_feature_list.append(image_features)

        # Save combined feature dataframe
        feature_df = pd.DataFrame.from_records(all_feature_list)
        feature_df.to_csv(self.output_file_path, index=False)

        self.plot_feature(feature_df)

    def _run_stitched(self):                
        
        if not self.mask_dir.exists():
            print(f"No directory named: {self.mask_dir}")
            return

        og_pred_image, og_bf_image, og_f_image = self.stitch_patches()

        bin_image = self.binarize(og_pred_image)
        if np.sum(bin_image) == 0:
            return

        bf_image = np.mean(255 - og_bf_image, axis=-1)
        f_image = np.mean(og_f_image, axis=-1)

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

        # Merging dictionaries
        image_features = {**glob_features, **dt_features, **bf_features, **f_features}
        image_features['Image Names'] = self.output_dir.parts[-3]

        # JSON-serializable
        image_features_json = {k: float(v) for k, v in image_features.items() if k != 'Image Names'}

        with open(self.output_dir / "Stitched_Collagen_Features.json", "w") as f:
            json.dump(image_features_json, f)
                
        # Saving combined feature dataframe
        feature_df = pd.DataFrame.from_dict(image_features, orient='index').reset_index()
        feature_df.to_csv(self.output_dir / 'Stitched_Collagen_Features.csv', index=False)
    
    def binarize(self, image):
        # Take in a continuous image prediction (2D) and return binarized image

        bin_img = image.copy()
        bin_img = (bin_img >= (255 * self.threshold["F"])).astype(np.uint8)

        return bin_img

    def global_features(self, bin_image):

        # Takes as input the binary image and returns global features
        total_patch_area = np.shape(bin_image)[0] * np.shape(bin_image)[1]
        collagen_area = np.sum(bin_image)
        if collagen_area > 0:
            tissue_area = np.sum(binary_fill_holes(bin_image))
            other_area = total_patch_area - tissue_area

            area_ratio = collagen_area / (tissue_area + collagen_area)

        else:
            tissue_area = total_patch_area
            area_ratio = 0.0

        feature_dict = {
            'Total Patch Area':total_patch_area,
            'Tissue Area': tissue_area,
            'Empty Area': other_area,
            'Collagen Area':collagen_area,
            'Collagen Area Ratio':area_ratio
        }

        return feature_dict

    def distance_transform_features(self, bin_image):
        
        # Takes as input the binarized collagen image and returns distance transform features
        # Should already be uint8 type
        # Using L2 for distance, mask_size = 5 
        dist_transform_img = cv2.distanceTransform(bin_image, cv2.DIST_L2,5)
        # Setting 0 values to nan for statistics
        dist_transform_img[dist_transform_img == 0] = np.nan

        sum_distance = np.nansum(dist_transform_img)
        mean_distance = np.nanmean(dist_transform_img)
        max_distance = np.nanmax(dist_transform_img)
        std_distance = np.nanstd(dist_transform_img)
        med_distance = np.nanmedian(dist_transform_img)

        feature_dict = {
            'Sum Distance Transform': sum_distance,
            'Mean Distance Transform': mean_distance,
            'Max Distance Transform': max_distance,
            'Standard Deviation Distance Transform': std_distance,
            'Median Distance Transform': med_distance
        }

        return feature_dict
    
    def intensity_features(self, gray_image, image_key='BF'):

        # Take as input the grayscale image and return a dictionary of intensity features
        intensity_image = gray_image.copy().astype(float)
        # Ignoring non-collagen regions (defined using threshold)
        intensity_image[intensity_image < (255 * self.threshold[image_key])] = np.nan

        mean_int = np.nanmean(intensity_image)
        med_int = np.nanmedian(intensity_image)
        std_int = np.nanstd(intensity_image)
        max_int = np.nanmax(intensity_image)

        feature_dict = {
            'Mean Intensity': mean_int,
            'Median Intensity': med_int,
            'Standard Deviation Intensity': std_int,
            'Maximum Intensity': max_int
        }

        return feature_dict

    def texture_features(self, gray_image, image_key='BF'):

        # Taking the grayscale collagen image and returning dictionary of texture features
        texture_image = gray_image.copy()
        # Ignoring non-collagen regions (using self.threshold)
        texture_image[texture_image < (255 * self.threshold[image_key])] = 0

        texture_matrix = graycomatrix(texture_image, [1] ,[0] , levels=256, symmetric=True, normed=True)
        
        texture_feature_names = ['Contrast','Homogeneity','Correlation','Energy']
        feature_dict = {}
        for i,t_name in enumerate(texture_feature_names):
            t_feat_value = graycoprops(texture_matrix, t_name.lower())
            feature_dict[t_name] = t_feat_value[0][0]

        return feature_dict

    def plot_feature(self,feature_df):

        for feat in feature_df.columns.tolist():

            if not feat=='Image Names':

                feat_bar = px.violin(
                    data_frame=feature_df,
                    y = feat
                )

                feat_bar.update_layout(
                    yaxis = {
                        'title':{
                            'text':f'<b>{feat}</b>'
                        }
                    },
                    title = {
                        'text': f'<b>Violin plot of {feat}<br>across predicted image patches'
                    }
                )

                feat_bar.write_image(self.output_dir / f'{feat}.png')

    def get_patch_coordinates(self,patch_name):
        ext = patch_name.split('.')[-1]
        patch_name = patch_name.replace(ext, "")
        parts = patch_name.split("_")[1].split(' ')
        try:
            x_coord = int(parts[-1].split("X")[-1])  # Extract X index
        except ValueError:         
            x_coord = 0
        try:
            y_coord = int(parts[-2].split("Y")[-1])  # Extract Y index
        except ValueError:
            y_coord = 0

        return y_coord, x_coord
        
    def stitch_patches(self, downsample=10, patch_overlap=150):
        """
        Stitch and downsample predictions for whole slide-level fibrosis quantification
        """
        ""
        checked_mask_paths, checked_f_paths, checked_bf_paths = [], [], []
        patch_shape = None

        # Keep only non-empty mask patches and align indices with f/bf
        for i, mask_path in enumerate(self.mask_paths):
            patch_img = np.array(Image.open(mask_path))
            patch_shape = patch_img.shape

            if np.sum(patch_img) > 0:
                checked_mask_paths.append(mask_path)
                checked_f_paths.append(self.f_image_paths[i])
                checked_bf_paths.append(self.bf_image_paths[i])

        if not checked_mask_paths:
            raise ValueError("No non-empty mask patches found!")
        
        # Extract coordinates based on mask filenames
        x_coords, y_coords = [], []
        for mask_path in checked_mask_paths:
            # key = Path(mask_path).stem.replace("Test_Example_", "")
            key = Path(mask_path).stem.replace("_prediction", "").replace("_LargeGlobalFlatfield", "").replace("Test_Example_", "")
            y_coord, x_coord = self.get_patch_coordinates(key)
            x_coords.append(x_coord)
            y_coords.append(y_coord)

        # Normalize coordinates
        y_coords = [i - min(y_coords) for i in y_coords]
        x_coords = [i - min(x_coords) for i in x_coords]
        
        # Stitched canvas size
        max_width, max_height = max(x_coords) + 1, max(y_coords) + 1
        pixel_width = (max_width * patch_shape[1]) - ((max_width - 1) * patch_overlap)
        pixel_height = (max_height * patch_shape[0]) - ((max_height - 1) * patch_overlap)

        stitched_downsampled_mask = np.zeros(
            (pixel_height // downsample, pixel_width // downsample), dtype=np.uint8
        )
        stitched_downsampled_bf = np.zeros((*stitched_downsampled_mask.shape, 3), dtype=np.uint8)
        stitched_downsampled_f = np.zeros_like(stitched_downsampled_bf)       
        
        # Weight map for overlaps
        overlap_downsampled = np.zeros(
            (pixel_height // downsample, pixel_width // downsample), dtype=np.uint8
        )

        print(f"Assembling patches: {len(checked_mask_paths)}")       
        
        with tqdm(zip(y_coords, x_coords, checked_mask_paths, checked_bf_paths, checked_f_paths),
                total=len(checked_mask_paths)) as pbar:
            for y, x, mask_path, bf_path, f_path in pbar:
                pbar.set_description(f"Working on patch: {Path(mask_path).name}")

                # --- Load mask patch ---
                checked_patch = np.array(Image.open(mask_path))[
                    0 : -(patch_overlap + 1), 0 : -(patch_overlap + 1)
                ]
                resized_patch = resize(
                    checked_patch,
                    output_shape=[
                        checked_patch.shape[0] // downsample,
                        checked_patch.shape[1] // downsample,
                    ])
                
                # --- Load bf patch ---
                checked_bf = np.array(Image.open(bf_path))[
                    0 : -(patch_overlap + 1), 0 : -(patch_overlap + 1)
                ]
                resized_bf = resize(
                    checked_bf,
                    output_shape=[
                        checked_patch.shape[0] // downsample,
                        checked_patch.shape[1] // downsample,
                        3,
                    ])

                # --- Load f patch ---
                checked_f = np.array(Image.open(f_path))[
                    0 : -(patch_overlap + 1), 0 : -(patch_overlap + 1)
                ]
                resized_f = resize(
                    checked_f,
                    output_shape=[
                        checked_patch.shape[0] // downsample,
                        checked_patch.shape[1] // downsample,
                        3,
                    ])                
                
                # Where should this patch go?
                y_start, x_start = int(y * resized_patch.shape[0]), int(x * resized_patch.shape[1])
                
                #  --- Accumulate (float32) ---
                stitched_downsampled_mask[y_start:y_start+resized_patch.shape[0],
                                        x_start:x_start+resized_patch.shape[1]] += 255 * resized_patch

                stitched_downsampled_bf[y_start:y_start+resized_patch.shape[0],
                                        x_start:x_start+resized_patch.shape[1], :] += 255 * resized_bf

                stitched_downsampled_f[y_start:y_start+resized_patch.shape[0],
                                    x_start:x_start+resized_patch.shape[1], :] += 255 * resized_f

                # --- Update overlap counter ---
                overlap_downsampled[y_start:y_start+resized_patch.shape[0],
                                    x_start:x_start+resized_patch.shape[1]] += 1
                pbar.update(1)
                
        # --- Normalize by overlaps ---
        overlap_downsampled = np.maximum(overlap_downsampled, 1)  # prevent division by 0
        stitched_downsampled_mask = (stitched_downsampled_mask / overlap_downsampled).astype(np.uint8)
        stitched_downsampled_bf   = (stitched_downsampled_bf / overlap_downsampled[..., None]).astype(np.uint8)
        stitched_downsampled_f    = (stitched_downsampled_f / overlap_downsampled[..., None]).astype(np.uint8)
        
        # Save outputs
        slide_name = self.output_dir.parts[-3]
        Image.fromarray(stitched_downsampled_mask).save(self.output_dir / f"{slide_name}_Stitched_Output.tif")
        Image.fromarray(stitched_downsampled_bf).save(self.output_dir / f"{slide_name}_Stitched_BF.tif")
        Image.fromarray(stitched_downsampled_f).save(self.output_dir / f"{slide_name}_Stitched_F.tif")

        return stitched_downsampled_mask, stitched_downsampled_bf, stitched_downsampled_f
        

def main(args):

    feature_extractor = Feature_Extractor(
        mask_dir = args.mask_dir,
        bf_image_dir = args.bf_image_dir,
        f_image_dir = args.f_image_dir,        
        output_dir = args.output_dir,
        threshold = args.threshold,
        use_stitched = args.use_stitched
    )
    feature_extractor.run()


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description = 'Collagen feature extraction using predicted masks'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default="./Config/feat_extract.json",
        help='Path to the config file'
    )
    args = parser.parse_args()
    
    # Load config from JSON
    with open(args.config) as json_file:
        data = json.load(json_file)

    # Overwrite args with config
    args = argparse.Namespace(**data)
    
    if not getattr(args, 'root_dir', None):
        main(args)
    else:
        args.root_dir = Path(args.root_dir)

        for slide_path in args.root_dir.iterdir():
            if not slide_path.is_dir():
                continue

            # Test only one slide for debugging
            if '2H' not in slide_path.name:
                print(f"Skipping {slide_path.name} as it does not contain 2H")
                continue
            
            # Save current root_dir
            cur_root_dir = args.root_dir
            
            args.root_dir = slide_path
            print(args.root_dir)
            slide_name = slide_path.name
            
            #  --- Automatically detect directories ---
            subdirs = [d for d in slide_path.iterdir() if d.is_dir() and slide_name in d.name]

            args.bf_image_dir = next((d for d in subdirs if 'B' in d.name and not 'C' in d.name and not 'F' in d.name), None)
            args.f_image_dir  = next((d for d in subdirs if 'F' in d.name and not 'B' in d.name and not 'C' in d.name), None)
            args.mask_dir     = next((d for d in subdirs if 'C' in d.name and 'Testing_Output' not in d.name), None)

            # Handle Testing_Output inside C directory if it exists
            if args.mask_dir and (args.mask_dir / 'Testing_Output').exists():
                args.mask_dir = args.mask_dir / 'Testing_Output'
            
            # Check required directories
            if not args.bf_image_dir or not args.f_image_dir or not args.mask_dir:
                print(f"\n[WARNING] Skipping {slide_path.name}: Missing required subdirectories.")
                print(f"  Found BF:   {args.bf_image_dir}")
                print(f"  Found F:    {args.f_image_dir}")
                print(f"  Found Mask: {args.mask_dir}")
                print("--------------------------------------------------")
                continue
            
            # Set output directory            
            args.output_dir = slide_path / "Features"
            # print(args)
            print('-------------------------------------------')
            start_time = time.time()
            main(args)
            
            time_diff = time.time() - start_time
            print(f'Time taken to extract features for {slide_name}: {int(time_diff // 3600)}:{int((time_diff % 3600) // 60)}:{int(time_diff % 60)}')
            print('-------------------------------------------')
            
            # Retrive current root_dir
            args.root_dir = cur_root_dir

