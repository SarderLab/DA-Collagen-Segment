import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, KFold
import umap
import random
import seaborn as sns



class Fibrosis_Analysis:
    """
    A class for analyzing fibrosis severity using handcrafted collagen features 
    combined with EHR (Electronic Health Record) data.
    
    Steps:
      1. Create dataset from collagen CSVs + EHR Excel file.
      2. Apply a list of regressors/classifiers.
      3. Compare and plot model performances.
    """

    def __init__(self, root_folder: str, ehr_excel_path: str, task_type: str = "regression"):
        """
        Initialize the Fibrosis_Analysis class.

        Parameters
        ----------
        root_folder : str
            Path to the folder containing CSV files with collagen features.
        ehr_excel_path : str
            Path to the Excel file containing EHR data.
        task_type : str, optional
            Type of ML task — 'regression' or 'classification'. Default is 'regression'.
        """
        self.root_folder = root_folder
        self.ehr_excel_path = ehr_excel_path
        self.task_type = task_type.lower()

        # Placeholders        
        self.slide_paths = [s for s in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, s))]
        self.dataset = None
        self.models = {}
        self.results = {}
        self.X_train = self.X_test = self.y_train = self.y_test = None

        # Input checks
        if not os.path.exists(root_folder):
            raise FileNotFoundError(f"Root folder not found: {root_folder}")
        if not os.path.exists(ehr_excel_path):
            raise FileNotFoundError(f"EHR Excel file not found: {ehr_excel_path}")
        if self.task_type not in ["regression", "classification"]:
            raise ValueError("task_type must be either 'regression' or 'classification'")

        def create_dataset(self):
            """
            Create a combined dataset by merging collagen feature CSVs with EHR data.
            
            Assumes folder structure:
                root_folder/
                    1H/
                        B_1H-0
                        F_1H-0
                        C_1H-0
                        Features_1H-0/
                    2H/
                    ...
            """

            # Step 1: Collect feature CSV paths
            feature_paths = []
            for slide_dir in self.slide_paths:
                slide_full_path = os.path.join(self.root_folder, slide_dir)
                for subfolder in os.listdir(slide_full_path):
                    subfolder_path = os.path.join(slide_full_path, subfolder)
                    if os.path.isdir(subfolder_path) and "Features" in subfolder:
                        # Collect all CSV files inside this Features folder
                        csv_files = [os.path.join(subfolder_path, f) 
                                    for f in os.listdir(subfolder_path) if f.endswith(".csv")]
                        feature_paths.extend(csv_files)

            if not feature_paths:
                raise FileNotFoundError("No feature CSV files found in the provided root folder.")

            # Step 2: Read and concatenate all collagen feature CSVs
            all_features = []
            for feat_path in feature_paths:
                feat_df = pd.read_csv(feat_path)
                all_features.append(feat_df)
            collagen_df = pd.concat(all_features, ignore_index=True)

            # Step 3: Read EHR Excel file
            ehr_df = pd.read_excel(self.ehr_excel_path)

            # Step 4: Merge collagen features with EHR data on 'PatientID'
            if 'Image Names' not in collagen_df.columns or 'Image Names' not in ehr_df.columns:
                raise KeyError("Both collagen CSVs and EHR Excel must have a 'Image Names' column.")
            
            self.dataset = pd.merge(collagen_df, ehr_df, on='Image Names')
            print(f"Dataset created successfully with {self.dataset.shape[0]} samples and {self.dataset.shape[1]} features.")

        
        def apply_models(self):            

            SEED = 42
            np.random.seed(SEED)
            random.seed(SEED)

            # Define regressors
            regressors = {
                'SVM': SVR(),  # SVR is deterministic unless inputs are randomized
                'KNN': KNeighborsRegressor(),  # KNN is deterministic
                'XGBoost': XGBRegressor(verbosity=0, random_state=SEED),
                'ANN': MLPRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1, random_state=SEED),
                # 'ANN': MLPRegressor(max_iter=1000, random_state=SEED),  # Original ANN with less iterations
                'LinearRegression': LinearRegression()  # deterministic
            }

            # Groups: patient identifier for each row in X
            groups = self.dataset['SlideName'].values

            # Define feature set 
            
            collagen_features = self.dataset.columns[:31]
            BF_features = self.dataset.columns[31:39]
            F_features = self.dataset.columns[39:46]
            outcomes = ['1yr SCr', '1yr eGFR', '1yr SCr', '2yr SCr', '2yr eGFR', 'eGFR Slope', 'Pathologist Fibrosis Score']
            # only collagen features + BF features
            # exclude_cols = outcomes + ['SlideName'] + F_features

            # X = self.dataset[[col for col in self.dataset.columns if col not in exclude_cols]]
            # X = self.dataset[collagen_features.tolist() + BF_features.tolist()]
            # X = self.dataset[BF_features.tolist()]
            X = self.dataset[BF_features.tolist() + F_features.tolist()]

            # Store results
            results = []
            label_results = []

            # Create KFold object
            # gkf = GroupKFold(n_splits=5)
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            np.save(self.out_dir+'kf.npy', kf)  # Save the KFold object for reproducibility

            # # Save the KFold object for reproducibility in output_path
            # np.save(output_path+'gkf.npy', gkf)

            # Main loop
            for outcome in outcomes:
                y = self.dataset[outcome].values
                
                # for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
                for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    Npts = len(X_train)  # Number of features (columns) in the training set
                    
                    # print(f"len(X_train): {len(X_train)}, len(X_test): {len(X_test)}, {Npts}")

                    # Scale features
                    # scaler = StandardScaler()
                    scaler = MinMaxScaler()  # Using MinMaxScaler for better performance with UMAP
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Use top-k UMAP components (say 5)
                    if Npts < 10:
                        top_k_list = [2, 3, 4]
                    else:
                        top_k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]  
                    
                    # UMAP dimensionality reduction
                    # reducer = umap.UMAP(n_components=top_k_list[-1], random_state=SEED)
                    # If speed is more important than reproducibility, remove random_state and optionally set n_jobs=-1 to use all cores:            
                    reducer = umap.UMAP(n_components=top_k_list[-1], n_jobs=-1)
                    X_train_umap = reducer.fit_transform(X_train_scaled)
                    X_test_umap = reducer.transform(X_test_scaled)

                    for k in top_k_list:
                        X_train_k = X_train_umap[:, :k]
                        X_test_k = X_test_umap[:, :k]

                        # Apply regressors
                        for name, model in regressors.items():
                            model.fit(X_train_k, y_train)
                            y_pred = model.predict(X_test_k)
                            r2 = r2_score(y_test, y_pred)
                            results.append({
                                'fold': fold,
                                'Outcome': outcome,
                                'Regressor': name,
                                'R2': r2,
                                'UMAP_Components': k                    ,
                            # })
                            # label_results.append({
                                'y-test': y_test.tolist(),
                                'y-pred': y_pred.tolist(),
                                'SlideName': groups[test_idx].tolist() # Include SlideName for reference
                            })
                            # print(f"[{outcome}] {name}: R2 = {r2:.3f}")

            # Convert to DataFrame for easy viewing
            self.results_df = pd.DataFrame(results)
            # label_results_df = pd.DataFrame(label_results)

            # results_df.to_csv(output_path+'UMAP_Regression_Results.csv', index=False)
            # results_df.to_csv(output_path+'UMAP_Regression_Results_BF.csv', index=False)
            # results_df.to_csv(output_path+'UMAP_Regression_Results_justBF.csv', index=False)
            self.results_df.to_csv(self.out_dir+'UMAP_Regression_Results_BF_F.csv', index=False)
            # label_results_df.to_csv(output_path+'UMAP_Regression_Results.csv', index=False)
        
        def plot_results(self):

            # Compute mean R2 for each regressor, outcome, and num components
            grouped = (
                self.results_df.groupby(['Outcome', 'Regressor', 'UMAP_Components'])['R2']
                .mean()
                .reset_index()
                #.pivot_table(index='UMAP_Components', columns='Regressor', values='R2')  # Optional for wide form
            )

            # For each outcome and regressor, find UMAP_Components (k) with highest average R2
            best_k_df = (
                grouped.loc[grouped.groupby(['Outcome', 'Regressor'])['R2'].idxmax()]
                .reset_index(drop=True)
            )
            # print(best_k_df)

            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=best_k_df,
                x='Outcome',         # This will be your main grouping
                y='R2',
                hue='Regressor',     # Each color/group is a regressor
                edgecolor='black'
            )

            plt.title('Best mean R2 (and best UMAP k) per Outcome', weight='bold')#, grouped by Regressor')
            plt.ylabel('Mean R2 (best UMAP components per regressor)', weight='bold')
            plt.xlabel('Outcome', weight='bold')
            # plt.ylim(0, 1.1)  # Or set a range to accommodate your R2 scores
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Regressor')

            # Annotate bars with k values
            try:
                for i, row in best_k_df.iterrows():
                    xpos = (
                        list(best_k_df['Outcome'].unique()).index(row['Outcome']) +
                        (list(best_k_df['Regressor'].unique()).index(row['Regressor'])
                        - len(best_k_df['Regressor'].unique()) / 2) * 0.18 + 0.1
                    )
                    plt.text(
                        x=xpos,
                        y=min(row['R2'] + 0.02, 1.05),  # Limit so text doesn’t go above plot
                        s=f"k={row['UMAP_Components']}",
                        ha='center', va='bottom', rotation=90, fontsize=9, color='black'
                    )
            except Exception as e:
                print(f"Annotation failed: {e}")

            plt.tight_layout()
            plt.show()

            # Final save step
            try:
                plt.tight_layout()
                # plt.savefig(output_path + '/bar_plot_all.png', dpi=300, bbox_inches='tight')
                plt.savefig(self.out_dir + '/bar_plot_BF.png', dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Saving figure failed: {e}")
            finally:
                plt.close()


# ---------- MAIN FUNCTION ----------
def main(args):
    """
    Main entry point for fibrosis analysis pipeline.
    """
    print(f"\nRunning Fibrosis Analysis ({args.task_type})")
    print(f"Root folder: {args.root_folder}")
    print(f"EHR Excel: {args.ehr_excel_path}")

    # Initialize pipeline
    analysis = Fibrosis_Analysis(
        root_folder=args.root_folder,
        ehr_excel_path=args.ehr_excel_path,
        task_type=args.task_type
    )

    # Step 1: Create dataset
    print("Creating dataset...")
    analysis.create_dataset()

    # Step 2: Apply models
    print("Training models...")
    analysis.apply_models()

    # Step 3: Plot results
    print("Plotting results...")
    analysis.plot_results()

    print("\nFibrosis analysis completed successfully!")
    return analysis


# ---------- EXECUTION BLOCK ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fibrosis Analysis Pipeline - Disease Severity Prediction Using Collagen Features',
    )
    parser.add_argument(
        '--config',
        type=str,
        default="./Config/fibrosis.json",
        help='Path to the configuration JSON file'
    )
    args = parser.parse_args()

    # Load config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as json_file:
        data = json.load(json_file)

    # Overwrite argparse defaults with config values
    args = argparse.Namespace(**data)

    # Run main pipeline
    main(args)