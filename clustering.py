# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import pandas as pd
import torch
from tqdm import tqdm

from Input_Pipeline import make_training_set, AttrSubset
from Cluster import Clusterer


class FakeNeptune:
    def assign(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def upload(self, *args, **kwargs): pass


def load_image_paths(input_parameters, mode):
    input_image_type = list(input_parameters['image_dir'].keys())
    image_paths, t_image_paths = [], None

    if input_parameters['type'] == 'multi':
        image_paths_base = []
        for inp_type in input_image_type:
            path = input_parameters['image_dir'][inp_type]
            if os.path.isdir(path):
                image_paths_base.append(sorted(glob.glob(path + '/*')))
            elif os.path.isfile(path):
                image_paths_base.append(sorted(pd.read_csv(path)['Paths'].tolist()))
            else:
                raise ValueError(f"Invalid path for {inp_type}: {path}")

        image_paths = [[f, b] for f, b in zip(image_paths_base[0], image_paths_base[1])]

        if mode == "ADDA":
            t_image_paths_base = []
            for inp_type in input_image_type:
                t_paths = input_parameters['target_dir'][inp_type]
                if isinstance(t_paths, str):
                    t_image_paths_base.append(sorted(glob.glob(t_paths + '/*')))
                else:
                    t_image_paths_base.append([sorted(glob.glob(tp + '/*')) for tp in t_paths])

            if isinstance(input_parameters['target_dir'][input_image_type[0]], str):
                t_image_paths = [[f, b] for f, b in zip(t_image_paths_base[0], t_image_paths_base[1])]
            else:
                t_image_paths = []
                for t in range(len(input_parameters['target_dir'][input_image_type[0]])):
                    F_paths, B_paths = t_image_paths_base[0][t], t_image_paths_base[1][t]
                    t_image_paths.append([[f, b] for f, b in zip(F_paths, B_paths)])
    else:
        inp_type = input_image_type[0]
        path = input_parameters['image_dir'][inp_type]
        if os.path.isdir(path):
            image_paths = sorted(glob.glob(path + '/*'))
        elif os.path.isfile(path):
            image_paths = sorted(pd.read_csv(path)['Paths'].tolist())

        if mode == "ADDA":
            t_paths_list = input_parameters['target_dir'][inp_type]
            if isinstance(t_paths_list, str):
                t_image_paths = sorted(glob.glob(t_paths_list + '/*'))
            else:
                t_image_paths = []
                for t_path in t_paths_list:
                    t_image_paths.append(sorted(glob.glob(t_path + '/*')))

    return image_paths, t_image_paths


def load_model_and_metadata(input_parameters, nept_run=None):
    model_file = input_parameters['model_file']
    preprocessing = input_parameters['preprocessing']
    model_details = input_parameters['model_details']
    return model_file, preprocessing, model_details


def main():
    torch.cuda.empty_cache()
    parameters_file = sys.argv[1]
    parameters = json.load(open(parameters_file))
    input_parameters = parameters['input_parameters']
    mode = parameters["mode"]

    nept_run = FakeNeptune()
    os.makedirs(input_parameters['output_dir'], exist_ok=True)

    # Load images
    image_paths, t_image_paths = load_image_paths(input_parameters, mode)

    # Load model & metadata
    model_file, preprocessing, model_details = load_model_and_metadata(input_parameters, nept_run)
    input_parameters['model_details'] = model_details
    input_parameters['preprocessing'] = preprocessing

    # Build datasets
    print("Creating dataset for source...")
    _, dataset_test = make_training_set('test', None, None, image_paths, [], input_parameters)
    dataset_test = AttrSubset(dataset_test, list(range(len(image_paths))))

    datasets = [dataset_test]
    if mode == "ADDA" and t_image_paths:
        if isinstance(t_image_paths[0], list) and isinstance(t_image_paths[0][0], list):
            for i, t_paths in enumerate(t_image_paths):
                print(f"Creating dataset for target {i+1}")
                _, dataset_test_t = make_training_set('test', None, None, t_paths, [], input_parameters)
                dataset_test_t = AttrSubset(dataset_test_t, list(range(len(t_paths))))
                datasets.append(dataset_test_t)
        else:
            print("Creating dataset for target")
            _, dataset_test_t = make_training_set('test', None, None, t_image_paths, [], input_parameters)
            dataset_test_t = AttrSubset(dataset_test_t, list(range(len(t_image_paths))))
            datasets.append(dataset_test_t)

    # Sites
    sites = input_parameters.get("dataset_names", ["CGPL", "Emory", "UCD_PATH", "BIDMC"])

    # Run clustering
    cluster_object = Clusterer(input_parameters, plot_labels=sites)
    cluster_object.run_umap_multiple_datasets_before_after(
        datasets,
        dataset_names=sites,
        trained_model_file=model_file
    )


if __name__ == '__main__':
    main()
