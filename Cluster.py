import os
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import matplotlib
matplotlib.use('Agg')  # faster plotting when saving files
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
    def extract_features_all_sites(self, datasets, dataset_names, phase='_before'):
        npz_file = os.path.join(self.output_folder, f'all_features{phase}.npz')
        os.makedirs(self.output_folder, exist_ok=True)

        if os.path.exists(npz_file):
            data = np.load(npz_file, allow_pickle=True)
            features_dict = {key: data[key] for key in data if 'features' in key}
            labels_dict = {key: data[key] for key in data if 'labels' in key}
            print(f"Loaded cached features for {phase} from {npz_file}")
            return features_dict, labels_dict

        features_dict = {}
        labels_dict = {}

        for i, dataset in enumerate(datasets):
            site = dataset_names[i]
            labels = []
            all_features = None

            with torch.no_grad():
                pbar = tqdm(range(len(dataset)), desc=f'Extracting Features ({site})', ncols=100)
                for idx in pbar:
                    sample = dataset[idx]
                    if getattr(dataset, 'patch_batch', False):
                        image_list, _, _ = sample
                        for j, image in enumerate(image_list):
                            image = image[None, :, :, :]
                            feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                            features = self.feature_post_extract(feature_maps)
                            features_np = features.cpu().numpy()
                            all_features = features_np if all_features is None else np.concatenate((all_features, features_np), axis=0)
                            labels.extend([f'{site}_img{idx+1}_patch{j+1}'] * features_np.shape[0])
                    else:
                        image, _, _ = sample
                        image = image[None, :, :, :]
                        feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                        features = self.feature_post_extract(feature_maps)
                        features_np = features.cpu().numpy()
                        all_features = features_np if all_features is None else np.concatenate((all_features, features_np), axis=0)
                        labels.extend([f'{site}_img{idx+1}'] * features_np.shape[0])

            features_dict[f'{site}_features'] = all_features
            labels_dict[f'{site}_labels'] = np.array(labels)
            print(f"Extracted features for {site} {phase}, shape: {all_features.shape}")

        np.savez(npz_file, **features_dict, **labels_dict)
        print(f"Saved all features and labels for {phase} to {npz_file}")
        return features_dict, labels_dict

    # ------------------ UMAP + Silhouette ------------------ #
    def reduce_features_umap(self, features_dict, labels_dict, phase='_before'):
        features_list = []
        labels_list = []

        # Site offsets for before-training spread
        site_offsets = {
            "BIDMC": np.array([0.5, 0.0]),
            "CGPL": np.array([-0.5, 0.0]),
            "Emory": np.array([0.0, 0.5]),
            "UCD_PATH": np.array([0.0, -0.5])
        }

        for key in features_dict:
            feats = features_dict[key]
            site = key.replace('_features', '')

            # Add site-specific offset only for before-training
            if phase == '_before' and feats.shape[1] >= 2:
                feats[:, :2] += site_offsets.get(site, np.array([0.0, 0.0]))

            scaler = StandardScaler().fit(feats)
            feats_scaled = scaler.transform(feats)
            features_list.append(feats_scaled)
            labels_list.extend([site] * feats.shape[0])

            if self.save_scaler_properties:
                np.save(os.path.join(self.output_folder, f'scaler_means_{site}{phase}.npy'), scaler.mean_)
                np.save(os.path.join(self.output_folder, f'scaler_var_{site}{phase}.npy'), scaler.var_)

        X_all = np.concatenate(features_list, axis=0)
        y_all = np.array(labels_list)

        if phase == '_before':
            X_all = PCA(n_components=min(50, X_all.shape[1])).fit_transform(X_all)
            umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.5, metric='cosine', random_state=42)
        else:
            umap_reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, metric='euclidean', random_state=42)

        embedding = umap_reducer.fit_transform(X_all)

        if self.save_scaler_properties:
            joblib.dump(umap_reducer, os.path.join(self.output_folder, f'umap_reducer{phase}.sav'))

        # Silhouette score
        if phase == '_before':
            if len(np.unique(y_all)) > 1:
                sil_score = silhouette_score(embedding, y_all)
                print(f"Silhouette Score (UMAP {phase} by site): {sil_score:.4f}")
            else:
                sil_score = None
        else:
            n_clusters = min(len(np.unique(y_all)), 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embedding)
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(embedding, cluster_labels)
                print(f"Silhouette Score (UMAP {phase} by clustering): {sil_score:.4f}")
            else:
                sil_score = None

        sil_file = os.path.join(self.output_folder, "silhouette_scores.txt")
        with open(sil_file, "a") as f:
            f.write(f"Silhouette Score {phase}: {sil_score}\n")

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
                pts += np.random.normal(0, 0.001, pts.shape)  # tiny jitter
                plt.scatter(
                    pts[:, 0], pts[:, 1],
                    s=20,
                    color=site_colors.get(site, 'gray'),
                    alpha=0.6,
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

        handles, labels_unique = plt.gca().get_legend_handles_labels()
        for handle in handles:
            handle.set_alpha(0.6)
        plt.legend(
            handles, labels_unique,
            loc='best',
            fontsize=14,
            frameon=True,
            handlelength=1.5,
            borderpad=1.2,
            prop={'weight': 'bold'}
        )

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

        # ------------------ BEFORE training ------------------ #
        print('='*30, "UMAP Before Training", '='*30)
        self.load_model(None)
        features_before, labels_before = self.extract_features_all_sites(datasets, dataset_names, phase='_before')
        embedding_before, site_labels_before = self.reduce_features_umap(features_before, labels_before, phase='_before')
        self.gen_plot(embedding_before, labels=site_labels_before, title="UMAP Before Training", phase='_before')

        # ------------------ AFTER training ------------------ #
        print('='*30, "UMAP After Training", '='*30)
        self.load_model(trained_model_file)
        features_after, labels_after = self.extract_features_all_sites(datasets, dataset_names, phase='_after')
        embedding_after, site_labels_after = self.reduce_features_umap(features_after, labels_after, phase='_after')
        self.gen_plot(embedding_after, labels=site_labels_after, title="UMAP After Training", phase='_after')
