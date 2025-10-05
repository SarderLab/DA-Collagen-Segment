# -*- coding: utf-8 -*-
"""
K-Fold Testing Script: Save predictions per fold & target
"""

import os
import sys
import json
import torch
import gc
from glob import glob
from sklearn.model_selection import KFold

from Input_Pipeline import make_training_set, AttrSubset
from InferenceDA import Test_Network


def main():
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Load parameters
    parameters_file = sys.argv[1]
    parameters = json.load(open(parameters_file))
    input_parameters = parameters['input_parameters']

    # Dummy Neptune (optional)
    class FakeNeptune:
        def assign(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def upload(self, *args, **kwargs): pass

    nept_run = FakeNeptune()

    # Load image paths
    image_paths_base = []
    for key, path in input_parameters['image_dir'].items():
        if os.path.isdir(path):
            image_paths_base.append(sorted(glob(path + '/*')))
        elif os.path.isfile(path):
            import pandas as pd
            image_paths_base.append(sorted(pd.read_csv(path)['Paths'].tolist()))

    # Combine inputs (assumes two modalities like F and B)
    image_paths = [[f, b] for f, b in zip(image_paths_base[0], image_paths_base[1])]

    # Create dataset (test only)
    _, dataset_test = make_training_set('test', None, None, image_paths, [], input_parameters)
    dataset_test = AttrSubset(dataset_test, list(range(len(image_paths))))
    dataset_test.testing_metrics = False  # disable metrics

    # K-Fold setup
    k_folds = int(parameters.get('train_parameters', {}).get('train_test_split', {}).get('k-fold', 5))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    model_dir = input_parameters['model_dir']
    output_dir_base = input_parameters['output_dir']

    for fold_count, (_, test_idx) in enumerate(kf.split(image_paths), start=1):
        print(f"\n--- Fold {fold_count}/{k_folds} ---")

        # Select model for this fold
        fold_model_candidates = sorted(glob(os.path.join(model_dir, f'Collagen_Seg_Model_Latest_{fold_count}.pth')))
        if not fold_model_candidates:
            print(f"No model found for fold {fold_count}")
            continue
        fold_model_path = fold_model_candidates[0]

        # Subset dataset for this fold
        dataset_valid_fold = AttrSubset(dataset_test, test_idx)
        dataset_valid_fold.testing_metrics = False

        # Loop over target datasets
        for target_idx, target_path in enumerate(input_parameters['target_dir']['DUET']):
            target_name = os.path.basename(target_path.rstrip('/'))
            output_target_dir = os.path.join(output_dir_base, f'fold_{fold_count}', target_name)
            os.makedirs(output_target_dir, exist_ok=True)

            print(f"Predicting for target dataset: {target_name} (Fold {fold_count})")

            # Inject target-specific folder
            input_parameters['current_target_output_dir'] = output_target_dir
            input_parameters['current_target_idx'] = target_idx

            # Run predictions
            Test_Network(fold_model_path, dataset_valid_fold, nept_run, input_parameters)

        # Clear memory
        del dataset_valid_fold
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
