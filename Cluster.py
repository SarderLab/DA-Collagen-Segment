import os
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

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

        # Post-processing features (spatial averaging + flattening)
        self.feature_post_extract = nn.Sequential(
            nn.AvgPool2d(kernel_size=15),
            nn.Flatten(start_dim=1)
        ).to(self.device)

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
    def extract_feature_loop(self, dataset, site_name=None):
        labels = []
        all_features = None

        with torch.no_grad():
            pbar = tqdm(range(len(dataset)), desc=f'Extracting Features ({site_name})', ncols=100)
            for i in pbar:
                sample = dataset[i]
                if getattr(dataset, 'patch_batch', False):
                    image_list, _, _ = sample
                    for j, image in enumerate(image_list):
                        image = image[None, :, :, :]
                        feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                        features = self.feature_post_extract(feature_maps)
                        features_np = features.cpu().numpy()
                        all_features = features_np if all_features is None else np.concatenate((all_features, features_np), axis=0)
                        labels.extend([site_name or f'image-{i+1}_patch-{j+1}'] * features_np.shape[0])
                else:
                    image, _, _ = sample
                    image = image[None, :, :, :]
                    feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                    features = self.feature_post_extract(feature_maps)
                    features_np = features.cpu().numpy()
                    all_features = features_np if all_features is None else np.concatenate((all_features, features_np), axis=0)
                    labels.extend([site_name or f'image-{i+1}'] * features_np.shape[0])
        print(f"\nShape of features: {all_features.shape}, Total labels: {len(labels)}")
        return all_features, labels

    # ------------------ Unsupervised UMAP ------------------ #
    def reduce_features_umap(self, features_list, labels_list, phase=None, separate=False):
        scaled_features = []
        scaled_labels = []

        # Scale per site
        for feats, site in zip(features_list, labels_list):
            scaler = StandardScaler().fit(feats)
            feats_scaled = scaler.transform(feats)
            scaled_features.append(feats_scaled)
            scaled_labels.extend([site] * feats.shape[0])

            if self.save_scaler_properties:
                np.save(os.path.join(self.output_folder, f'scaler_means_{site}{phase}.npy'), scaler.mean_)
                np.save(os.path.join(self.output_folder, f'scaler_var_{site}{phase}.npy'), scaler.var_)

        X_all = np.concatenate(scaled_features, axis=0)
        y_all = np.array(scaled_labels)

        # UMAP for natural separation
        umap_reducer = umap.UMAP(
            n_neighbors=50,   # larger for more global structure
            min_dist=0.3,     # controls how tightly points cluster
            metric='euclidean',
            random_state=42
        )
        embedding = umap_reducer.fit_transform(X_all)

        # No artificial separation
        if separate:
            pass

        if self.save_scaler_properties:
            joblib.dump(umap_reducer, os.path.join(self.output_folder, f'umap_reducer{phase}.sav'))

        print(f"UMAP embedding shape: {embedding.shape}")
        return embedding, y_all

    # ------------------ Generate Plot ------------------ #
    def gen_plot(self, umap_features, labels=None, title="UMAP of latent features", phase=None):
        site_colors = {
            "BIDMC": "#FF7F0E",
            "CGPL": "#D62728",
            "Emory": "#E377C2",
            "UCD_PATH": "#1F77B4"
        }

        plt.figure(figsize=(12, 9))
        labels_np = np.array(labels) if labels is not None else np.array([''] * len(umap_features))

        for site in np.unique(labels_np):
            pts = umap_features[labels_np == site]
            if pts.size > 0:
                plt.scatter(
                    pts[:, 0], pts[:, 1],
                    s=60,
                    color=site_colors.get(site, 'gray'),
                    alpha=0.6,  # semi-transparent points
                    edgecolor='k',
                    linewidth=0.5,
                    label=site
                )

        plt.title(title, fontsize=20, fontweight='bold')
        plt.xlabel("UMAP 1", fontsize=16, fontweight='bold')
        plt.ylabel("UMAP 2", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Legend with matching alpha/transparency
        handles, labels_unique = plt.gca().get_legend_handles_labels()
        for handle in handles:
            handle.set_alpha(0.6)
        plt.legend(handles, labels_unique, loc='best', fontsize=14, frameon=True, handlelength=1.5, borderpad=1.2)

        plt.tight_layout()
        output_dir = os.path.join(self.output_folder, 'UMAP_Plots')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'UMAP_Plot{phase}.png')
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Saved UMAP plot to: {file_path}")

    # ------------------ Run full UMAP pipeline ------------------ #
    def run_umap_multiple_datasets_before_after(self, datasets, dataset_names=None, trained_model_file=None):
        if dataset_names is None:
            dataset_names = ["Source"] + [f"Target-{i}" for i in range(1, len(datasets))]

        # BEFORE training
        print('='*30, "UMAP Before Training", '='*30)
        self.load_model(None)
        features_list_before = []
        labels_list_before = []

        for i, dataset in enumerate(datasets):
            features, _ = self.extract_feature_loop(dataset, site_name=dataset_names[i])
            features_list_before.append(features)
            labels_list_before.append(dataset_names[i])

        embedding_before, site_labels_before = self.reduce_features_umap(
            features_list_before, labels_list_before, phase='_before', separate=False
        )
        self.gen_plot(embedding_before, labels=site_labels_before, title="UMAP Before Training", phase='_before')

        # AFTER training
        print('='*30, "UMAP After Training", '='*30)
        self.load_model(trained_model_file)
        features_list_after = []
        labels_list_after = []

        for i, dataset in enumerate(datasets):
            features, _ = self.extract_feature_loop(dataset, site_name=dataset_names[i])
            features_list_after.append(features)
            labels_list_after.append(dataset_names[i])

        embedding_after, site_labels_after = self.reduce_features_umap(
            features_list_after, labels_list_after, phase='_after', separate=False
        )
        self.gen_plot(embedding_after, labels=site_labels_after, title="UMAP After Training", phase='_after')
