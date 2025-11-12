"""

Performing dimensional reduction from latent features extracted from image patches 


"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import torch
from torch import nn

import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
from Models import create_model


class Clusterer:
    def __init__(self,
                 parameters,
                 plot_labels = None,
                 save_scaler_properties = True,
                 save_latent_features = False,
                 save_umap_coordinates = True):
        
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

    def run_clustering_iterator(self,model_file,dataset):
        
        self.load_model(model_file)

        # extracting features and labels
        extracted_features, labels = self.extract_feature_loop(dataset)
        print('Done extracting Features!')

        if self.save_latent_features:
            self.save_features(extracted_features,labels,None,file_name = 'Extracted_Latent_Features.csv')
            print('Done Saving Features!')
        
        # UMAP reduction
        reduced_features, sil_score = self.reduce_features(extracted_features, labels)
        print('Done Reducing Features')

        if self.save_umap_coordinates:
            self.save_features(reduced_features,['UMAP1','UMAP2'], labels, file_name = 'Extracted_UMAP_Coordinates.csv')
            print('Done saving UMAP coordinates')

        # Make some plot from the reduced features and labels
        if self.plot_labels is None:
            plot = self.gen_plot(reduced_features)
            print('Done Generating Plot!')

        else:
            plot = self.gen_plot(reduced_features, labels)

    def load_model(self, model_file):

        model_details = self.parameters['model_details']
        target_type = model_details['target_type']
        ann_classes = model_details['ann_classes']
        
        if target_type=='binary':
            n_classes = len(ann_classes)
        else:
            n_classes = 1

        in_channels = int(self.parameters['preprocessing']['image_size'].split(',')[-1])
        
        self.model = create_model(model_details, n_classes)
        
        if model_file is not None:
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def extract_feature_loop(self, dataset):
        
        labels = []
        all_features = None
        
        with torch.no_grad():                
            with tqdm(range(len(dataset)), desc=f'Extracting Features ') as pbar:
                for i in range(len(dataset)):
                    sample = dataset[i]
                    
                    # ---------------- Feature Extraction Pipeline for large images ----------------
                    if dataset.patch_batch:  #isinstance(sample[0], list):  
                        # Large image → patch lists
                        image_list, _, _ = sample
                        
                        for image in image_list:
                            image = image[None, :, :, :]  # add batch dimension
                            feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                            features = self.feature_post_extract(feature_maps)

                            if all_features is None:
                                all_features = features.cpu().numpy()
                            else:
                                features = features.cpu().numpy()
                                all_features = np.concatenate((all_features, features), axis=0)

                            labels.append(f'image-{i+1}')
                            pbar.update(1)
                    else:
                        image, _, _ = sample
                        image = image[None, :, :, :]  # add batch dimension
                        feature_maps = self.model.get_embeddings(image.to(self.device))[-1]
                        features = self.feature_post_extract(feature_maps)

                        if all_features is None:
                            all_features = features.cpu().numpy()
                        else:
                            features = features.cpu().numpy()
                            all_features = np.concatenate((all_features, features), axis=0)

                        labels.append(f'image-{i+1}')
                        pbar.update(1)
                  
            print(f'shape of features: {np.shape(all_features)}')
                            
        return all_features, labels

    def save_features(self,features, col_labels, row_labels, file_name):

        features_df = pd.DataFrame(data = features, columns=col_labels, index=row_labels)
        features_df.to_csv(os.path.join(self.output_folder, file_name))
        

    def reduce_features(self, features, labels=None, phase=None):

        scaler = StandardScaler().fit(features)
        # Saving the scaler information to the output directory
        scaler_means = scaler.mean_
        scaler_vars = scaler.var_

        if self.save_scaler_properties:
            output_mean_file = os.path.join(self.output_folder, f'scaler_means{phase}.npy')
            with open(output_mean_file, 'wb') as f:
                np.save(f, scaler_means)
            
            output_var_file = os.path.join(self.output_folder, f'scaler_var{phase}.npy')
            with open(output_var_file,'wb') as f:
                np.save(f, scaler_vars)

        scaled_data = scaler.transform(features)

        umap_reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            metric='euclidean', 
            random_state=42, 
            force_approximation_algorithm=True
        )
        embedding = umap_reducer.fit_transform(scaled_data)
        
        # Compute silhouette score (only if labels are given and >1 unique class)
        sil_score = None
        if labels is not None:
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                sil_score = silhouette_score(embedding, labels)
            else:
                sil_score = np.nan  # meaningless if only one label

        # Saving umap_reducer to be used later
        if self.save_scaler_properties:
            output_reducer_file = os.path.join(self.output_folder, f'umap_reducer{phase}.sav')
            joblib.dump(umap_reducer, output_reducer_file)

        return embedding, sil_score

    def gen_plot(self,umap_features, labels=None, title="UMAP of latent features", phase=None):

        if self.plot_labels is None:
            if labels is None:
                scatter_plot = px.scatter(
                    data_frame = pd.DataFrame(data=umap_features,columns = ['umap1','umap2'],index=labels),
                    x = 'umap1',
                    y='umap2',
                    title = title
                )
            else:
                scatter_plot = px.scatter(
                    data_frame = pd.DataFrame(data=umap_features,columns = ['umap1','umap2'],index=labels),
                    x = 'umap1',
                    y='umap2',
                    color = labels,
                    title = title
                )
            # Update font styles
            scatter_plot.update_layout(
                title=dict(text=title, font=dict(size=22, family="Arial", color="black")),
                legend=dict(                    
                    title=dict(text="", font=dict(size=6, family="Arial", color="black")),
                    font=dict(size=14, family="Arial", color="black"),
                    x=0.95,          # horizontal position (1 = far right)
                    y=0.95,          # vertical position (1 = top)
                    xanchor='right', # align right side of legend at x
                    yanchor='top',   # align top of legend at y
                    bgcolor='rgba(255,255,255,0.5)', # semi-transparent background
                    bordercolor='black',
                    borderwidth=1
                ),
                xaxis=dict(
                    title=dict(text="<b>UMAP1</b>", font=dict(size=18)),
                    tickfont=dict(size=14, family="Arial", color="black")
                ),
                yaxis=dict(
                    title=dict(text="<b>UMAP2</b>", font=dict(size=18)),
                    tickfont=dict(size=14, family="Arial", color="black")
                )
            )

            output_umap_plots = os.path.join(self.output_folder, 'UMAP_Plots')
            if not os.path.exists(output_umap_plots):
                os.makedirs(output_umap_plots)

            self.save_plot(scatter_plot,os.path.join(output_umap_plots,f'UMAP_Plot{phase}.png')) 
            
        else:
            # self.plot_labels is a pandas DataFrame containing an "Image_Names" column that aligns with the "labels" passed to this function
            column_labels = [i for i in self.plot_labels.columns.tolist() if not i=='Image_Names']
            umap_df = pd.DataFrame(data=umap_features,columns=['umap1','umap2'])
            umap_df['Image_Names'] = labels

            merged_df = pd.merge(umap_df,self.plot_labels,on='Image_Names')
            print(f'shape of merged: {merged_df.shape}')

            # Making output directory
            output_umap_plots = os.path.join(self.output_folder, 'UMAP_Plots')
            if not os.path.exists(output_umap_plots):
                os.makedirs(output_umap_plots)

            for f in column_labels:
                
                scatter_plot = px.scatter(
                    data_frame = merged_df,
                    x='umap1',
                    y='umap2',
                    color=f,
                    title = f'UMAP of latent features labeled with: {f}'
                )

                self.save_plot(scatter_plot, os.path.join(output_umap_plots, f'UMAP_Plot_{f}.png'))

        return scatter_plot

    def save_plot(self, plot, filename):

        plot.write_image(
            os.path.join(self.output_folder, filename),
            width=2000,   # width in pixels
            height=1500,  # height in pixels
            scale=1       # scaling factor to increase DPI
        )        

    def cluster_in_loop(self,model,image):

        # Extract latent features given an image, return latent features
        feature_maps = model.encoder(image.to(self.device))[-1]
        features = self.feature_post_extract(feature_maps)

        return features
    
    def run_umap_multiple_datasets_before_after(self, datasets, dataset_names=None, trained_model_file=None):
        """
        Extract embeddings for multiple datasets and generate UMAPs
        before and after training. Each dataset gets its own color.
        """
        
        if dataset_names is None:
            dataset_names = ["Source"]
            for i in range(1, len(datasets)):
                dataset_names.append(f"Target-{i}")
        
        all_labels = []
        
        # ---------- BEFORE TRAINING ----------
        print('='*30, "UMAP Before Training", '='*30)
        self.load_model(None)  # untrained
        all_features_before = []
        
        for i, dataset in enumerate(datasets):
            print('-'*20, f"Extracting features for dataset: {dataset_names[i]}", '-'*20)
            features, _ = self.extract_feature_loop(dataset)            
            print(f"Feature Extraction Done for {dataset_names[i]} with shape: {features.shape}")
            all_features_before.append(features)
            all_labels.extend([dataset_names[i]] * features.shape[0])
        
        all_features_before = np.concatenate(all_features_before, axis=0)
        embedding_before, sil_score = self.reduce_features(all_features_before, labels=all_labels, phase='_before')
        print(f"UMAP reduction before training done with Silhouette Score: {np.round(sil_score, 2)}")
        scatter_before = self.gen_plot(embedding_before, labels=all_labels, title=f"UMAP Before Training (Silhouette Score: {np.round(sil_score, 2)})", phase='_before')
        
        # ---------- AFTER TRAINING ----------
        print('='*30, "UMAP After Training", '='*30)
        self.load_model(trained_model_file)
        all_features_after = []
        
        for i, dataset in enumerate(datasets):
            features, _ = self.extract_feature_loop(dataset)
            all_features_after.append(features)
        
        all_features_after = np.concatenate(all_features_after, axis=0)
        embedding_after, sil_score = self.reduce_features(all_features_after, labels=all_labels, phase='_after')
        print(f"UMAP reduction after training done with Silhouette Score: {np.round(sil_score, 2)}")
        scatter_after = self.gen_plot(embedding_after, labels=all_labels, title=f"UMAP After Training (Silhouette Score: {np.round(sil_score, 2)})", phase='_after')
        
        return scatter_before, scatter_after
