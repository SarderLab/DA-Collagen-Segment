import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import torch
from torch.utils.data import DataLoader
from torch import nn

import plotly.express as px

from sklearn.preprocessing import StandardScaler
import umap

from Models import create_model  # your model loader


class Clusterer:
    def __init__(self,
                 parameters,
                 plot_labels=None,
                 save_scaler_properties=True,
                 save_latent_features=False,
                 save_umap_coordinates=True):

        self.plot_labels = plot_labels
        self.output_folder = parameters['output_dir']
        self.parameters = parameters

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # Post-processing features (spatial averaging and flattening)
        self.feature_post_extract = nn.Sequential(
            nn.AvgPool2d(kernel_size=15),
            nn.Flatten(start_dim=1)
        )
        self.feature_post_extract.to(self.device)

        self.save_latent_features = save_latent_features
        self.save_umap_coordinates = save_umap_coordinates
        self.save_scaler_properties = save_scaler_properties

    # ------------------ Load Model ------------------ #
    def load_model(self, model_file):
        model_details = self.parameters['model_details']
        target_type = model_details['target_type']
        ann_classes = model_details['ann_classes']

        n_classes = len(ann_classes) if target_type == 'binary' else 1
        in_channels = int(self.parameters['preprocessing']['image_size'].split(',')[-1])

        self.model = create_model(model_details, n_classes)

        if model_file is not None:
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    # ------------------ Feature Extraction ------------------ #
    def extract_feature_loop(self, dataset):
        labels = []
        all_features = None

        with torch.no_grad():
            with tqdm(range(len(dataset)), desc='Extracting Features') as pbar:
                for i in range(len(dataset)):
                    sample = dataset[i]

                    if getattr(dataset, 'patch_batch', False):
                        # Large image → patch lists
                        image_list, _, _ = sample
                        for image in image_list:
                            image = image[None, :, :, :]  # add batch dimension
                            feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                            features = self.feature_post_extract(feature_maps)
                            features_np = features.cpu().numpy()

                            if all_features is None:
                                all_features = features_np
                            else:
                                all_features = np.concatenate((all_features, features_np), axis=0)

                            labels.extend([f'image-{i+1}'] * features_np.shape[0])
                            pbar.update(1)
                    else:
                        image, _, _ = sample
                        image = image[None, :, :, :]
                        feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                        features = self.feature_post_extract(feature_maps)
                        features_np = features.cpu().numpy()

                        if all_features is None:
                            all_features = features_np
                        else:
                            all_features = np.concatenate((all_features, features_np), axis=0)

                        labels.extend([f'image-{i+1}'] * features_np.shape[0])
                        pbar.update(1)

        print(f"Shape of features: {all_features.shape}, Total labels: {len(labels)}")
        return all_features, labels

    # ------------------ Save Features ------------------ #
    def save_features(self, features, col_labels, row_labels, file_name):
        df = pd.DataFrame(data=features, columns=col_labels, index=row_labels)
        df.to_csv(os.path.join(self.output_folder, file_name))

    # ------------------ Reduce Features ------------------ #
    def reduce_features(self, features, phase=None):
        scaler = StandardScaler().fit(features)
        scaled_data = scaler.transform(features)

        if self.save_scaler_properties:
            np.save(os.path.join(self.output_folder, f'scaler_means{phase}.npy'), scaler.mean_)
            np.save(os.path.join(self.output_folder, f'scaler_var{phase}.npy'), scaler.var_)

        umap_reducer = umap.UMAP()
        embedding = umap_reducer.fit_transform(scaled_data)

        if self.save_scaler_properties:
            joblib.dump(umap_reducer, os.path.join(self.output_folder, f'umap_reducer{phase}.sav'))

        print(f"UMAP embedding shape: {embedding.shape}")
        return embedding

    # ------------------ Generate Plot ------------------ #
    def gen_plot(self, umap_features, labels=None, title="UMAP of latent features", phase=None):
        df = pd.DataFrame(umap_features, columns=['umap1', 'umap2'])
        if labels is not None:
            df['labels'] = labels

        scatter_plot = px.scatter(df,
                                  x='umap1',
                                  y='umap2',
                                  color='labels' if labels is not None else None,
                                  title=title)

        output_dir = os.path.join(self.output_folder, 'UMAP_Plots')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'UMAP_Plot{phase}.png')
        scatter_plot.write_image(file_path, width=1000, height=1000, scale=2)
        print(f"Saved UMAP plot to: {file_path}")
        return scatter_plot

    # ------------------ Full Clustering Pipeline ------------------ #
    def run_clustering_iterator(self, model_file, dataset):
        self.load_model(model_file)
        features, labels = self.extract_feature_loop(dataset)
        print("Features extracted.")

        embedding = self.reduce_features(features)
        print("UMAP reduced.")

        plot = self.gen_plot(embedding, labels=labels)
        print("Plot generated.")
        return plot

    # ------------------ UMAP Multiple Datasets ------------------ #
    def run_umap_multiple_datasets_before_after(self, datasets, dataset_names=None, trained_model_file=None):
        if dataset_names is None:
            dataset_names = ["Source"] + [f"Target-{i}" for i in range(1, len(datasets))]

        all_labels = []

        # BEFORE training
        print('='*30, "UMAP Before Training", '='*30)
        self.load_model(None)
        all_features_before = []

        for i, dataset in enumerate(datasets):
            features, _ = self.extract_feature_loop(dataset)
            all_features_before.append(features)
            all_labels.extend([dataset_names[i]] * features.shape[0])

        all_features_before = np.concatenate(all_features_before, axis=0)
        embedding_before = self.reduce_features(all_features_before, phase='_before')
        scatter_before = self.gen_plot(embedding_before, labels=all_labels, title="UMAP Before Training", phase='_before')

        # AFTER training
        print('='*30, "UMAP After Training", '='*30)
        self.load_model(trained_model_file)
        all_features_after = []

        for dataset in datasets:
            features, _ = self.extract_feature_loop(dataset)
            all_features_after.append(features)

        all_features_after = np.concatenate(all_features_after, axis=0)
        embedding_after = self.reduce_features(all_features_after, phase='_after')
        scatter_after = self.gen_plot(embedding_after, labels=all_labels, title="UMAP After Training", phase='_after')

        return scatter_before, scatter_after
