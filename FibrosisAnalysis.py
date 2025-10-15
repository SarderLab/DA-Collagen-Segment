import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import umap
import random
import seaborn as sns
import matplotlib.pyplot as plt

class Fibrosis_Analysis:
    """
    Analyze fibrosis severity using C + BF + F collagen features and EHR data.
    """

    def __init__(self, root_folder: str, ehr_excel_path: str, task_type: str = "regression"):
        self.root_folder = root_folder
        self.ehr_excel_path = ehr_excel_path
        self.task_type = task_type.lower()
        self.dataset = None
        self.results_df = None
        self.out_dir = os.path.join(root_folder, "Results/")
        os.makedirs(self.out_dir, exist_ok=True)

        # List slide folders
        self.slide_paths = [s for s in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, s))]

        # Input checks
        if not os.path.exists(root_folder):
            raise FileNotFoundError(f"Root folder not found: {root_folder}")
        if not os.path.exists(ehr_excel_path):
            raise FileNotFoundError(f"EHR Excel file not found: {ehr_excel_path}")
        if self.task_type not in ["regression", "classification"]:
            raise ValueError("task_type must be either 'regression' or 'classification'")

    def create_dataset(self):
        """
        Merge C + BF + F collagen feature CSVs with EHR data.
        """

        feature_paths = []
        for slide_dir in self.slide_paths:
            slide_full_path = os.path.join(self.root_folder, slide_dir)
            for subfolder in os.listdir(slide_full_path):
                subfolder_path = os.path.join(slide_full_path, subfolder)
                if os.path.isdir(subfolder_path) and "Features" in subfolder:
                    csv_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(".csv")]
                    feature_paths.extend(csv_files)

        if not feature_paths:
            raise FileNotFoundError("No feature CSV files found in the provided root folder.")

        # Read and concatenate all CSVs
        all_features = [pd.read_csv(f) for f in feature_paths]
        full_df = pd.concat(all_features, ignore_index=True)

        print("\n--- Full concatenated feature dataframe ---")
        print(full_df.head(5))

        # Ensure Image Names exist
        if 'Image Names' not in full_df.columns:
            raise KeyError("Feature CSVs must have 'Image Names' column.")

        # Aggregate by SlideName
        full_df['SlideName'] = full_df['Image Names'].apply(lambda x: str(x).split('_')[0])

        # Select only numeric columns for mean aggregation
        numeric_cols = full_df.select_dtypes(include=[np.number]).columns
        agg_df = full_df.groupby('SlideName')[numeric_cols].mean().reset_index()

        print("\n--- Aggregated slide-level feature dataframe ---")
        print(agg_df.head(5))

        # Read EHR Excel
        ehr_df = pd.read_excel(self.ehr_excel_path)
        if 'SlideName' not in ehr_df.columns:
            if 'Image Names' in ehr_df.columns:
                ehr_df = ehr_df.rename(columns={'Image Names': 'SlideName'})
            else:
                raise KeyError("EHR Excel must have 'SlideName' or 'Image Names' column.")

        print("\n--- EHR dataframe ---")
        print(ehr_df.head(5))

        # Merge aggregated features with EHR
        self.dataset = pd.merge(agg_df, ehr_df, on='SlideName', how='inner')

        print("\n--- Final merged dataset ---")
        print(self.dataset.head(5))
        print(f"Dataset created successfully with {self.dataset.shape[0]} samples and {self.dataset.shape[1]} features.")

    def apply_models(self):
        """
        Train regressors using GroupKFold CV with UMAP, hyperparameter tuning, and NaN/inf handling.
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")

        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)

        # Define regressors and hyperparameter grids
        regressors = {
            'XGBoost': {
                'model': XGBRegressor(verbosity=0, random_state=SEED),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.7, 1.0],
                    'colsample_bytree': [0.7, 1.0]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1], 'kernel': ['rbf', 'linear']}
            },
            'KNN': {
                'model': KNeighborsRegressor(),
                'params': {'n_neighbors': [3, 5, 7]}
            },
            'ANN': {
                'model': MLPRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1, random_state=SEED),
                'params': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}
            },
            'Ridge': {'model': Ridge(random_state=SEED), 'params': {'alpha': [0.1, 1.0]}},
            'Lasso': {'model': Lasso(random_state=SEED), 'params': {'alpha': [0.01, 0.1]}}
        }

        groups = self.dataset['SlideName'].values

        # Select C + BF + F features
        feature_cols = self.dataset.columns.drop(['SlideName', '1yr eGFR', 'Age', 'Gender', '1yr SCr'])
        X = self.dataset[feature_cols]
        y = self.dataset['1yr eGFR'].values

        # Handle NaN / inf
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)

        results = []

        # GroupKFold
        gkf = GroupKFold(n_splits=5)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"\n--- Fold {fold} ---")
            print("Train slides:", groups[train_idx])
            print("Test slides:", groups[test_idx])

            # Handle NaN / inf in splits
            X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_train.fillna(X_train.mean(), inplace=True)
            X_test.fillna(X_train.mean(), inplace=True)

            # Scale
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # UMAP
            n_features = X_train_scaled.shape[1]
            top_k_list = [2, 3, 4] if n_features < 10 else list(range(2, min(11, n_features+1)))
            reducer = umap.UMAP(n_components=top_k_list[-1], n_jobs=-1)
            X_train_umap = reducer.fit_transform(X_train_scaled)
            X_test_umap = reducer.transform(X_test_scaled)

            for k in top_k_list:
                X_train_k = X_train_umap[:, :k]
                X_test_k = X_test_umap[:, :k]

                for name, info in regressors.items():
                    model = info['model']
                    params = info['params']

                    # GridSearchCV
                    grid = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
                    grid.fit(X_train_k, y_train)
                    best_model = grid.best_estimator_
                    y_pred = best_model.predict(X_test_k)
                    r2 = r2_score(y_test, y_pred)

                    results.append({
                        'fold': fold,
                        'Regressor': name,
                        'R2': r2,
                        'UMAP_Components': k,
                        'best_params': grid.best_params_,
                        'y-test': y_test.tolist(),
                        'y-pred': y_pred.tolist(),
                        'SlideName': groups[test_idx].tolist()
                    })

                    print(f"[Fold {fold}] {name} | R2={r2:.3f} | Best params: {grid.best_params_}")
                    print("First 5 true vs predicted:", list(zip(y_test[:5], y_pred[:5])))

        self.results_df = pd.DataFrame(results)
        print("\n--- Results dataframe ---")
        print(self.results_df.head(5))

        self.results_df.to_csv(os.path.join(self.out_dir, 'UMAP_Regression_Results_C_BF_F.csv'), index=False)
        print(f"Model results saved to {self.out_dir}")

    def plot_results(self):
        """
        Plot best R2 per regressor.
        """
        if self.results_df is None:
            raise ValueError("No results found. Run apply_models() first.")

        grouped = (
            self.results_df.groupby(['Regressor', 'UMAP_Components'])['R2']
            .mean().reset_index()
        )
        best_k_df = grouped.loc[grouped.groupby('Regressor')['R2'].idxmax()].reset_index(drop=True)

        plt.figure(figsize=(10,6))
        sns.barplot(data=best_k_df, x='Regressor', y='R2', edgecolor='black', palette='Set2')
        plt.title('Best mean R2 per Regressor', weight='bold')
        plt.ylabel('Mean R2', weight='bold')
        plt.xlabel('Regressor', weight='bold')
        plt.xticks(rotation=45, ha='right')
        for i, row in best_k_df.iterrows():
            plt.text(i, row['R2']+0.02, f"k={row['UMAP_Components']}", ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'bar_plot_C_BF_F.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


# ---------- MAIN FUNCTION ----------
def main(args):
    print(f"\nRunning Fibrosis Analysis ({args.task_type})")
    print(f"Root folder: {args.root_folder}")
    print(f"EHR Excel: {args.ehr_excel_path}")

    analysis = Fibrosis_Analysis(
        root_folder=args.root_folder,
        ehr_excel_path=args.ehr_excel_path,
        task_type=args.task_type
    )

    analysis.create_dataset()
    analysis.apply_models()
    analysis.plot_results()
    print("\nFibrosis analysis completed successfully!")
    return analysis


# ---------- EXECUTION BLOCK ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fibrosis Analysis Pipeline')
    parser.add_argument('--config', type=str, default="./Config/fibrosis.json",
                        help='Path to the configuration JSON file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as json_file:
        data = json.load(json_file)

    args = argparse.Namespace(**data)
    main(args)
