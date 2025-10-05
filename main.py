# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:39:49 2024

@author: fafsari

Organizing inputs (JSON file) and running either training or testing

Currently works for image patches (training) and either image patches or larger (not WSI) testing

"""

import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from math import floor
import json
import joblib
from math import ceil
from sklearn.model_selection import KFold
import torch
import time
import neptune
import gc

from Input_Pipeline import *
from Train import Training_Loop
from TrainmultiDA import TrainingDA_Loop
from TestmultiDA import Test_Network
from Cluster import Clusterer


# Class to use if no neptune information is provided without having to add logic every time nept_run is called
class FakeNeptune:
    def __init__(self):
        pass
    def assign(self):
        pass
    def log(self):
        pass
    def upload(self):
        pass

def check_image_bytes(image_path_list,lower_threshold = None, upper_threshold = None):
    # Checking if a path contains an image with greater than a set threshold of bytes
    # In this case it looks like all the images with less than ~40kb just contain background

    passed_list = []
    if lower_threshold is None and upper_threshold is None:
        return image_path_list

    for i in image_path_list:
        file_size = os.path.getsize(i)
        if not lower_threshold is None:
            if file_size>=lower_threshold:
                if not upper_threshold is None:
                    if file_size<=upper_threshold:
                        passed_list.append(i)
                else:
                    passed_list.append(i)
        else:
            if not upper_threshold is None:
                if file_size<=upper_threshold:
                    passed_list.append(i)
            else:
                passed_list.append(i)

    return passed_list

def check_duplicate(image_path_list,output_path):

    passed_list = []
    for i in image_path_list:
        image_name = i.split(os.sep)[-1]
        if not os.path.exists(output_path+f"Testing_Output/{image_name.replace('.jpg','_prediction.tif')}"):
            passed_list.append(i)
    
    return passed_list


def main():
    torch.cuda.empty_cache()
    # Changing up from sys.argv to reading a specific set of input parameters
    parameters_file = sys.argv[1]

    parameters = json.load(open(parameters_file))   

    # Getting input parameters and neptune-specific parameters (if specified)
    input_parameters = parameters['input_parameters']
    
    # Get mode
    mode = parameters["mode"]

    if 'neptune' in input_parameters:
        nept_params = input_parameters['neptune']

        nept_api_token = os.environ.get('NEPTUNE_API_TOKEN')
        nept_run = neptune.init_run(
            project = nept_params['project'],
            source_files = nept_params['source_files'],
            api_token = nept_api_token,
            tags = nept_params['tags']
            )

        nept_run['input_parameters'] = input_parameters
    else:
        nept_run = FakeNeptune()

    if input_parameters['phase'] == 'train':

        training_parameters = parameters['train_parameters']
        # Path to folder containing ground truth images    
        if os.path.isdir(input_parameters['label_dir']):
            label_paths = sorted(glob(input_parameters['label_dir']+'/*'))
            training_parameters['supervision'] = 'full'

        elif os.path.isfile(input_parameters['label_dir']):
            label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())
            training_parameters['supervision'] = 'full'
        else:
            print('Initial training of a model requires labeled targets')
        
        if mode == "ADDA":
            
            if isinstance(input_parameters['target_label_dir'], str):
                if os.path.isdir(input_parameters['target_label_dir']):
                    t_label_paths = sorted(glob(input_parameters['target_label_dir']+'/*'))
                    
                else:
                    print('Initial training of a model requires labeled targets')
            elif isinstance(input_parameters['target_label_dir'], list):
                t_label_paths = []
                for j in input_parameters['target_label_dir']:
                    if os.path.isdir(j):
                        t_label_paths.append(sorted(glob(j+'/*')))
                    elif os.path.isfile(j):
                        t_label_paths.append(sorted(pd.read_csv(j)['Paths'].tolist()))
                    else:
                        print('Initial training of a model requires labeled targets')
                        
                for t in range(len(t_label_paths)):
                    print(f"Target {t+1} label paths: {len(t_label_paths[t])}")

        input_image_type = list(input_parameters['image_dir'].keys())

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])
        
        training_parameters['output_dir'] = input_parameters['output_dir']

        if input_parameters['type'] == 'multi':
            
            # Getting multi-input image paths
            image_paths_base = []
            if mode == "ADDA":
                t_image_paths_base = []
                t_image_paths = [] 
                
            for inp_type in input_image_type:
                # print(inp_type,input_parameters['image_dir'][inp_type])
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'/*')))
                if mode == "ADDA":
                    if isinstance(input_parameters['target_dir'][inp_type], str):
                        if os.path.isdir(input_parameters['target_dir'][inp_type]):
                            t_image_paths_base.append(sorted(glob(input_parameters['target_dir'][inp_type]+'/*')))
                    elif isinstance(input_parameters['target_dir'][inp_type], list):
                        
                        inp_type_paths = []
                        
                        for t_path in input_parameters['target_dir'][inp_type]:           
                            if os.path.isdir(t_path):
                                inp_type_paths.append(sorted(glob(t_path+'/*')))
                            elif os.path.isfile(t_path):
                                inp_type_paths.append(sorted(pd.read_csv(t_path)['Paths'].tolist()))
                            else:
                                print('Initial training of a model requires labeled targets')
                        t_image_paths_base.append(inp_type_paths)
                    
            # Getting the image paths as lists of lists (each input type)
            image_paths = []
            for i , inp_type in enumerate(input_image_type):
                print(f"Input type: {inp_type}, Number of Source images: {len(image_paths_base[i])}")
                
            # for i in range(len(image_paths_base[0])):
            image_paths = [[f, b] for f, b in zip(image_paths_base[0], image_paths_base[1])]            
                
            if mode == "ADDA":
                # t_image_paths = []
                if isinstance(input_parameters['target_dir'][input_image_type[0]], str):
                    if os.path.isdir(input_parameters['target_dir'][input_image_type[0]]):                        
                        F_paths = t_image_paths_base[0]
                        B_paths = t_image_paths_base[1]
                        t_image_paths = [[f, b] for f, b in zip(F_paths, B_paths)]
                        
                    elif os.path.isfile(input_parameters['target_dir'][input_image_type[0]]):
                        image_paths_base.append(sorted(pd.read_csv(input_parameters['target_dir'][input_image_type[0]])['Paths'].tolist()))
                    print(f"Target: {len(t_image_paths)} paired images")
                elif isinstance(input_parameters['target_dir'][input_image_type[0]], list):
                    for t in range(len(input_parameters['target_dir'][input_image_type[0]])):
                        F_paths = t_image_paths_base[0][t]
                        B_paths = t_image_paths_base[1][t]
                        image_paths_t = [[f, b] for f, b in zip(F_paths, B_paths)]        
                        t_image_paths.append(image_paths_t)                
        
                    for i, paths in enumerate(t_image_paths):
                        print(f"Target {i+1}: {len(paths)} paired images")

        elif input_parameters['type']=='single':

            # Getting either DUET or Brightfield images
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'/*'))
            elif os.path.isfile(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())
        
        # Preload the entire dataset once
        dataset_train_all, dataset_valid_all = make_training_set(
            phase='train',
            train_img_paths=image_paths,
            train_tar=label_paths,
            valid_img_paths=image_paths,   # For full access, same as train
            valid_tar=label_paths,
            parameters=training_parameters
        )

        if mode == "ADDA":
            if isinstance(input_parameters['target_dir'][input_image_type[0]], str):
                dataset_train_all_t, dataset_valid_all_t = make_training_set(
                    phase='train',
                    train_img_paths=t_image_paths,
                    train_tar=t_label_paths,
                    valid_img_paths=t_image_paths,
                    valid_tar=t_label_paths,
                    parameters=training_parameters,
                    ema=True
                )
            elif isinstance(input_parameters['target_dir'][input_image_type[0]], list):
                dataset_train_all_t = []
                dataset_valid_all_t = []
                
                for t in range(len(t_image_paths)):
                    print(f"Creating dataset for target {t+1}")
                    train_all_t, valid_all_t = make_training_set(
                        phase='train',
                        train_img_paths=t_image_paths[t],
                        train_tar=t_label_paths[t],
                        valid_img_paths=t_image_paths[t],
                        valid_tar=t_label_paths[t],
                        parameters=training_parameters,
                        ema=True
                    )
                    dataset_train_all_t.append(train_all_t)
                    dataset_valid_all_t.append(valid_all_t)

        # Now determining which images are used for training and which are used for testing

        # --- K-Fold Split ---
        if 'k-fold' in training_parameters['train_test_split']:

            kf = KFold(
                n_splits=int(training_parameters['train_test_split']['k-fold']),
                shuffle=True,
                random_state=42
            )

            k_count = 0

            if mode == "ADDA":
                # Precompute source splits
                source_splits = list(kf.split(image_paths))

                # Precompute target splits
                if isinstance(t_image_paths, list) and isinstance(t_image_paths[0], list):
                    target_splits_list = [list(kf.split(t_paths)) for t_paths in t_image_paths]
                else:
                    target_splits_list = [list(kf.split(t_image_paths))]

                # Iterate over folds
                for k_count, source_fold in enumerate(source_splits, 1):
                    print("-"*50, f'On k-fold: {k_count}', "-"*50)
                    training_parameters["current_k_fold"] = k_count

                    # Source train/test indices
                    train_idx, test_idx = source_fold

                    # Source datasets (memory-efficient view)
                    dataset_train = AttrSubset(dataset_train_all, train_idx)
                    dataset_valid = AttrSubset(dataset_valid_all, test_idx)

                    # Neptune logging: indices only
                    nept_run.assign({f'{k_count}_Training_Idx': train_idx.tolist()})
                    nept_run.assign({f'{k_count}_Testing_Idx': test_idx.tolist()})

                    # --- Targets ---
                    dataset_train_t_list = []
                    dataset_valid_t_list = []

                    for t_idx, t_splits in enumerate(target_splits_list):
                        t_train_idx, t_test_idx = t_splits[k_count - 1]

                        # Neptune logging: indices only
                        nept_run.assign({f'{k_count}_Target{t_idx+1}_Training_Idx': t_train_idx.tolist()})
                        nept_run.assign({f'{k_count}_Target{t_idx+1}_Testing_Idx': t_test_idx.tolist()})

                        # Build datasets
                        dataset_train_t = AttrSubset(dataset_train_all_t[t_idx], t_train_idx)
                        dataset_valid_t = AttrSubset(dataset_valid_all_t[t_idx], t_test_idx)
                        dataset_valid_t.testing_metrics = True

                        dataset_train_t_list.append(dataset_train_t)
                        dataset_valid_t_list.append(dataset_valid_t)

                    # --- Train the ADDA models for this fold ---
                    modelsDA = TrainingDA_Loop(
                        input_parameters["model_file"], 
                        dataset_train_s=dataset_train, 
                        dataset_valid_s=dataset_valid,
                        dataset_train_t=dataset_train_t_list, 
                        dataset_valid_t=dataset_valid_t_list,
                        train_parameters=training_parameters,
                        nept_run=nept_run
                    )

                    # --- Test Source and Target models ---
                    training_parameters["model_details"]["target_type"] = "binary"
                    training_parameters["mode"] += "_Source"
                    print("-"*50, "Testing Source Model", "-"*50)
                    Test_Network(modelsDA["Source Model"], dataset_valid_t_list, nept_run, training_parameters)

                    training_parameters["mode"] = training_parameters["mode"].replace("_Source", "_Target")
                    print("-"*50, "Testing Target Model", "-"*50)
                    Test_Network(modelsDA["Target Model"], dataset_valid_t_list, nept_run, training_parameters)

                    # --- Cleanup to free memory ---
                    del dataset_train, dataset_valid
                    del dataset_train_t_list, dataset_valid_t_list
                    del modelsDA
                    gc.collect()

                    # Reset mode and target_type for next fold
                    training_parameters["mode"] = "ADDA"
                    training_parameters["model_details"]["target_type"] = "nonbinary"



            else:
                # Standard k-fold for source only
                for train_idx, test_idx in kf.split(image_paths):
                    k_count += 1
                    print("-"*50, f'On k-fold: {k_count}', "-"*50)
                    training_parameters["current_k_fold"] = k_count

                    train_idx = list(train_idx.astype(int))
                    test_idx  = list(test_idx.astype(int))

                    nept_run.assign({f'{k_count}_Training_Set': [image_paths[i] for i in train_idx]})
                    nept_run.assign({f'{k_count}_Testing_Set':  [image_paths[i] for i in test_idx]})

                    dataset_train = AttrSubset(dataset_train_all, train_idx)
                    dataset_valid = AttrSubset(dataset_valid_all, test_idx)
                    
                    # Training the model                                        
                    model = Training_Loop(
                        dataset_train=dataset_train,
                        dataset_valid=dataset_valid,
                        train_parameters=training_parameters,
                        nept_run=nept_run
                    )
                    # Testing the model
                    Test_Network(model, dataset_valid, nept_run, training_parameters)

        elif 'split' in training_parameters['train_test_split']:
            
            # Set the seed once to ensure reproducibility
            np.random.seed(42)
            shuffle_idx = np.random.permutation(len(image_paths))

            split_pct = training_parameters['train_test_split']['split']
            train_idx = shuffle_idx[0 : floor(split_pct * len(image_paths))]
            val_idx   = shuffle_idx[floor(split_pct * len(image_paths)) : len(image_paths)]
            
            dataset_train = AttrSubset(dataset_train_all, train_idx)
            dataset_valid = AttrSubset(dataset_valid_all, val_idx)               
            
            if mode == "None":
                model = Training_Loop(dataset_train, dataset_valid, training_parameters,nept_run)
                Test_Network(model, dataset_valid, nept_run, training_parameters)
                
            elif mode == "ADDA":
                
                # Set the seed once to ensure reproducibility
                np.random.seed(42)
                # Random train/test split with 'split' proportion assigned to training data
                shuffle_idx = np.random.permutation(len(t_image_paths))

                split_pct = training_parameters['train_test_split']['split']
                train_idx = shuffle_idx[0 : floor(split_pct * len(t_image_paths))]
                val_idx   = shuffle_idx[floor(split_pct * len(t_image_paths)) : len(t_image_paths)]
                
                dataset_train_t = AttrSubset(dataset_train_all_t, train_idx)                
                dataset_valid_t = AttrSubset(dataset_valid_all_t, val_idx)                
                
                modelDA = TrainingDA_Loop(input_parameters["model_file"], dataset_train, dataset_valid,
                                              dataset_train_t, dataset_valid_t, 
                                              training_parameters, nept_run)
                Test_Network(modelDA, dataset_valid_t, nept_run, training_parameters)

        elif 'training' in training_parameters['train_test_split']:
            
            # Passing a file containing the names of files in the training and testing sets
            # Adding the image_dir afterwards (dependent on whether it's a multi or single input)
            train_list = pd.read_csv(training_parameters['train_test_split']['training'])['Image_Names'].tolist()
            test_list = pd.read_csv(training_parameters['train_test_split']['testing'])['Image_Names'].tolist()

            if input_parameters['type']=='multi':
                train_images = []
                for tr in train_list:
                    # Getting each type of input
                    tr_list = []
                    for in_type in input_image_type:
                        tr_list.append(input_parameters['image_dir'][in_type]+tr)
                    # Adding lists of inputs to main train images list
                    train_images.append(tr_list)
                
                test_images = []
                for te in test_list:
                    te_list = []
                    for in_type in input_image_type:
                        te_list.append(input_parameters['image_dir'][in_type]+te)
                    
                    test_images.append(te_list)

            elif input_parameters['type']=='single':
                train_images = [input_parameters['image_dir'][input_image_type[0]]+i for i in train_list]
                test_images = [input_parameters['image_dir'][input_image_type[0]]+i for i in test_list]

            train_labels = [input_parameters['label_dir']+i for i in train_list]
            test_labels = [input_parameters['label_dir']+i for i in test_list]

            dataset_train, dataset_valid = make_training_set(
                'train',
                train_img_paths = train_images,
                train_tar = train_labels,
                valid_img_paths=test_images,
                valid_tar=test_labels,
                parameters= training_parameters
            )
            start_time = time.time()
            model = Training_Loop(dataset_train,dataset_valid,training_parameters,nept_run)
        
            end_time = time.time()
            spent_time = end_time - start_time
            print("Time spent to test is:", int(spent_time // 60), "minutes and", int(spent_time % 60), "seconds")
        
            Test_Network(model,dataset_valid,nept_run,training_parameters)

    elif input_parameters['phase']=='test':

        input_image_type = list(input_parameters['image_dir'].keys())

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])

        # Whether to test with labels (calculate metrics) or not
        if 'label_dir' in input_parameters:
            if os.path.isdir(input_parameters['label_dir']):
                label_paths = sorted(glob(input_parameters['label_dir']+'/*'))
            elif os.path.isfile(input_parameters['label_dir']):
                label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())
        else:
            label_paths = []
        
        if input_parameters['type'] == 'multi':
            
            
            image_paths_base = []
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'/*')))                    
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()))

            # Hack for only predicting on images which contain minimal background
            # Defined threshold (40kb) for brightfield images
            passed_images = check_image_bytes(image_paths_base[1],lower_threshold = None, upper_threshold = None)
            print(f'{len(passed_images)} passed the bytes check')

            if 'skip_duplicates' in input_parameters:
                if input_parameters['skip_duplicates']:
                    passed_images = check_duplicate(passed_images,input_parameters['output_dir'])
                    print(f'{len(passed_images)} passed the duplicates check!')

            passed_image_idxes = [image_paths_base[1].index(i) for i in passed_images]
            image_paths_base = [
                [image_paths_base[0][i] for i in passed_image_idxes],
                passed_images
            ]

            # Getting the image paths as lists of lists (each input type)
            image_paths = []
            for i in range(len(image_paths_base[0])):
                image_paths.append([image_paths_base[0][i],image_paths_base[1][i]])

        elif input_parameters['type']=='single':
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'/*'))
            elif os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())

        # Loading model and data
        if 'neptune' in input_parameters:
            model_version = neptune.init_model(
                project = nept_params['project'],
                # with_id = input_parameters['model'],
                mode = 'async',
                api_token = nept_api_token
            )

            if 'model_file' not in input_parameters:
                # Download the model from the model version
                if not os.path.exists(input_parameters['output_dir']+'/model/'):
                    os.makedirs(input_parameters['output_dir']+'/model/')

                model_version['model_file'].download(input_parameters['output_dir']+'/model')
                model_file = os.listdir(input_parameters['output_dir']+'/model/')[0]
            else:
                model_file = input_parameters['model_file']

            all_model_metadata = model_version.get_structure()
            prep_keys = all_model_metadata['preprocessing'].keys()
            preprocessing = {}
            for p in prep_keys:
                preprocessing[p] = model_version[f'preprocessing/{p}'].fetch()

            model_deets_keys = all_model_metadata['model_details'].keys()
            model_details = {}
            for m in model_deets_keys:
                model_details[m] = model_version[f'model_details/{m}'].fetch()

            # Grabbing clustering scaler and training set coordinates
            try:
                model_version['UMAP_Scaler/scaler_means'].download(input_parameters['output_dir']+'/scaler_means.npy')
                model_version['UMAP_Scaler/scaler_var'].download(input_parameters['output_dir']+'/scaler_var.npy')
                model_version['UMAP_Scaler/umap_reducer'].download(input_parameters['output_dir']+'/umap_reducer.sav')

                model_details['scaler_means'] = np.load(input_parameters['output_dir']+'/scaler_means.npy')
                model_details['scaler_var'] = np.load(input_parameters['output_dir']+'/scaler_var.npy')
                model_details['umap_reducer'] = joblib.load(input_parameters['output_dir']+'/umap_reducer.sav')
            except:
                print('no umap info for this model')
        else:

            model_file = input_parameters['model_file']
            preprocessing = input_parameters['preprocessing']
            model_details = input_parameters['model_details']

            model_details['scaler_means'] = None
            model_details['scaler_var'] = None

        input_parameters['model_details'] = model_details
        input_parameters['preprocessing'] = preprocessing

        # This is a hack for running on large sets of large images
        # image_set_size = 3 #len(image_paths)   #20
        image_set_size = len(image_paths)
        run_throughs = ceil(len(image_paths)/image_set_size)
        start_time = time.time()
        for run in range(run_throughs):
            
            if not int((run+1)*image_set_size)>=len(image_paths):
                print(f'Running on images: {int(run*image_set_size)} to {int((run+1)*image_set_size)} of {len(image_paths)} from folder {input_parameters["image_dir"][input_image_type[0]]}')
                run_paths = image_paths[int(run*image_set_size):int((run+1)*image_set_size)]
            else:
                print(f'Running on images: {int(run*image_set_size)} to {len(image_paths)} of {len(image_paths)} from folder {input_parameters["image_dir"][input_image_type[0]]}')
                run_paths = image_paths[int(run*image_set_size):len(image_paths)]
            
            # print(run_paths)
            if os.path.exists(input_parameters['label_dir']):    
                if not int((run+1)*image_set_size)>=len(image_paths):   
                    label_paths_run = label_paths[int(run*image_set_size):int((run+1)*image_set_size)]
                else:
                    label_paths_run = label_paths[int(run*image_set_size):len(image_paths)]
                    
                _, dataset_test = make_training_set(
                    'test',
                    None,
                    None,
                    run_paths,
                    label_paths_run,
                    input_parameters
                )
                dataset_test = AttrSubset(dataset_test, list(range(len(run_paths))))
                dataset_test.testing_metrics = True                
            else:
                _, dataset_test = make_training_set(
                    'test',
                    None,
                    None,
                    run_paths,
                    [],
                    input_parameters
                )            
                dataset_test = AttrSubset(dataset_test, list(range(len(run_paths))))
            Test_Network(model_file, dataset_test, nept_run, input_parameters)
            
            # Free up memory for efficiency
            del run_paths, dataset_test            
            gc.collect()
            torch.cuda.empty_cache()  
            
        end_time = time.time()
        spent_time = end_time - start_time
        print("Time spent to test is:", int(spent_time // 60), "minutes and", int(spent_time % 60), "seconds")

    elif input_parameters['phase'] == 'cluster':

        # Inputs are the same as testing but instead of predicting, just generating some clustering/relative clustering of latent features
        input_image_type = list(input_parameters['image_dir'].keys())
        label_paths = []
        if mode == "ADDA":
            t_label_paths = []

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])
        
        if input_parameters['type'] == 'multi':

            image_paths_base = []
            if mode == "ADDA":
                t_image_paths_base = []
                t_image_paths = [] 
                
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'/*')))
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()))
                else:
                    print(input_parameters['image_dir'][inp_type])
                    image_paths_base = []
                
                if mode == "ADDA":
                    if isinstance(input_parameters['target_dir'][inp_type], str):
                        if os.path.isdir(input_parameters['target_dir'][inp_type]):
                            t_image_paths_base.append(sorted(glob(input_parameters['target_dir'][inp_type]+'/*')))
                    elif isinstance(input_parameters['target_dir'][inp_type], list):
                        
                        inp_type_paths = []
                        
                        for t_path in input_parameters['target_dir'][inp_type]:           
                            if os.path.isdir(t_path):
                                inp_type_paths.append(sorted(glob(t_path+'/*')))
                            elif os.path.isfile(t_path):
                                inp_type_paths.append(sorted(pd.read_csv(t_path)['Paths'].tolist()))
                            else:
                                print('No target images!')
                        t_image_paths_base.append(inp_type_paths)

            for i , inp_type in enumerate(input_image_type):
                print(f"Input type: {inp_type}, Number of Source images: {len(image_paths_base[i])}")
            # image_paths = [list(i) for i in zip(*image_paths_base)]
            image_paths = [[f, b] for f, b in zip(image_paths_base[0], image_paths_base[1])]

            if mode == "ADDA":                
                if isinstance(input_parameters['target_dir'][input_image_type[0]], str):
                    F_paths = t_image_paths_base[0]
                    B_paths = t_image_paths_base[1]
                    t_image_paths = [[f, b] for f, b in zip(F_paths, B_paths)]        
                    # t_image_paths.append(image_paths_t)
                    print(f"Target: {len(t_image_paths)} paired images")
                    
                elif isinstance(input_parameters['target_dir'][input_image_type[0]], list):
                    t_image_paths = []
                    for t in range(len(input_parameters['target_dir'][input_image_type[0]])):
                        F_paths = t_image_paths_base[0][t]
                        B_paths = t_image_paths_base[1][t]
                        image_paths_t = [[f, b] for f, b in zip(F_paths, B_paths)]        
                        t_image_paths.append(image_paths_t)                
        
                    for i, paths in enumerate(t_image_paths):
                        print(f"Target {i+1}: {len(paths)} paired images")

        elif input_parameters['type']=='single':
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'/*'))
            elif os.path.isfile(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())
            else:
                print(input_parameters['image_dir'][input_image_type[0]])
                image_paths = []
        
        # Loading model and data
        if 'neptune' in input_parameters:
            model_version = neptune.init_model(
                project = nept_params['project'],
                with_id = input_parameters['model'],
                mode = 'async',
                api_token = nept_api_token
            )

            if 'model_file' not in input_parameters:
                # Download the model from the model version
                if not os.path.exists(input_parameters['output_dir']+'/model/'):
                    os.makedirs(input_parameters['output_dir']+'/model/')

                model_version['model_file'].download(input_parameters['output_dir']+'/model')
                model_file = os.listdir(input_parameters['output_dir']+'/model/')[0]
            else:
                model_file = input_parameters['model_file']

            all_model_metadata = model_version.get_structure()
            prep_keys = all_model_metadata['preprocessing'].keys()
            preprocessing = {}
            for p in prep_keys:
                preprocessing[p] = model_version[f'preprocessing/{p}'].fetch()

            model_deets_keys = all_model_metadata['model_details'].keys()
            model_details = {}
            for m in model_deets_keys:
                model_details[m] = model_version[f'model_details/{m}'].fetch()

        else:

            model_file = input_parameters['model_file']
            preprocessing = input_parameters['preprocessing']
            model_details = input_parameters['model_details']

        input_parameters['model_details'] = model_details
        input_parameters['preprocessing'] = preprocessing
        print(f"Creating dataset for the source")
        _, dataset_test = make_training_set(
            'test',
            None,
            None,
            image_paths,
            label_paths,
            input_parameters
        )
        dataset_test = AttrSubset(dataset_test, list(range(len(image_paths))))
        if mode == "ADDA":
            dataset_test = [dataset_test]
            if isinstance(input_parameters['target_dir'][input_image_type[0]], str):
                print(f"Creating dataset for the target")
                _, dataset_test_t = make_training_set(
                    'test',
                    None,
                    None,
                    t_image_paths,
                    t_label_paths,
                    input_parameters
                )
                dataset_test_t = AttrSubset(dataset_test_t, list(range(len(t_image_paths))))
                dataset_test.append(dataset_test_t)
            elif isinstance(input_parameters['target_dir'][input_image_type[0]], list):                                               
                for t in range(len(t_image_paths)):
                    print(f"Creating dataset for target {t+1}")
                    _, dataset_test_t = make_training_set(
                        'test',
                        None,
                        None,
                        t_image_paths[t],
                        t_label_paths,
                        input_parameters
                    )
                    dataset_test_t = AttrSubset(dataset_test_t, list(range(len(t_image_paths[t]))))
                    dataset_test.append(dataset_test_t)
        
        cluster_object = Clusterer(input_parameters)
        # scatter_plot_before, scatter_plot_after = cluster_object.run_umap_multiple_datasets_before_after(model_file,dataset_test)
        cluster_object.run_umap_multiple_datasets_before_after(
            dataset_test, 
            dataset_names=input_parameters["dataset_names"] if "dataset_names" in input_parameters else None,
            trained_model_file=model_file)
        
        # Saving plots
        # scatter_plot_before.savefig(input_parameters['output_dir']+'/UMAP_Plots/Before_Training.png',dpi=300)
        # scatter_plot_after.savefig(input_parameters['output_dir']+'/UMAP_Plots/After_Training.png',dpi=300)
        
        # # Getting labels from model_version
        # model_version['Merged_Results_Table'].download(input_parameters['output_dir']+'/Merged_Results_Table.csv')
        # plot_labels = pd.read_csv(input_parameters['output_dir']+'Merged_Results_Table.csv',header = 0)

        # cluster_object = Clusterer(input_parameters,plot_labels = plot_labels)
        # cluster_object.run_clustering_iterator(model_file,dataset_test)

        # # Uploading data to model_version metadata (if specified)
        # upload_to_model_version = True
        # if upload_to_model_version:
        #     model_version['UMAP_coordinates'].upload(input_parameters['output_dir']+'Extracted_UMAP_Coordinates.csv')
            
        #     umap_plots = os.listdir(input_parameters['output_dir']+'UMAP_Plots/')
        #     for u in umap_plots:
        #         model_version[f'UMAP_Plots/{u}'].upload(input_parameters['output_dir']+f'UMAP_Plots/{u}')

        # model_version['UMAP_Scaler/scaler_means'].upload(input_parameters['output_dir']+f'scaler_means.npy')
        # model_version['UMAP_Scaler/scaler_var'].upload(input_parameters['output_dir']+f'scaler_var.npy')
        # model_version['UMAP_Scaler/umap_reducer'].upload(input_parameters['output_dir']+f'umap_reducer.sav')

if __name__=='__main__':
    main()