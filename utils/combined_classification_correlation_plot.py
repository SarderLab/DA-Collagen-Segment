import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
base_dir = "/blue/pinaki.sarder/narra.sr/8_DUET_Data/Data_Correlation/T0-T1 UCD Kidney Biopsies/F1+F2/correlation_classification_updated_features"

folders = [
    "Collagen+Brightfield",
    "Collagen+Fluorescence",
    "Brightfield+Collagen+Fluorescence"
]

umap_k = 4  # UMAP components to filter

folder_colors = {
    "Collagen+Brightfield": "#1f77b4",
    "Collagen+Fluorescence": "#ff7f0e",
    "Brightfield+Collagen+Fluorescence": "#2ca02c"
}
# ==========================================

all_results = []

# ---------------- Load CSVs ----------------
for folder in folders:
    csv_path = os.path.join(base_dir, folder, "UMAP_Classification_Results_patch.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {folder} (CSV not found)")
        continue
    df = pd.read_csv(csv_path)
    
    # Required columns
    required_cols = ['UMAP_Components','Classifier','f1_weighted']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Skipping {folder} (missing required columns)")
        continue
    
    # Filter by UMAP components
    df = df[df['UMAP_Components'] == umap_k].copy()
    df["Folder"] = folder
    all_results.append(df)

if not all_results:
    raise ValueError("No valid CSVs found. Check folder names or CSVs.")

data = pd.concat(all_results, ignore_index=True)

# ==========================================
# ----------- Define Classifier Order ------
# ==========================================
# Custom order for classifiers in both plots
classifier_order = sorted(data['Classifier'].unique())
# Or define manually if you want: 
# classifier_order = ['LogisticRegression', 'SVM', 'KNN', 'MLP', 'XGB']

# ==========================================
# ----------- Boxplot ----------------------
# ==========================================
plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.set_context("talk")

# Boxplot
box = sns.boxplot(
    x="Classifier", y="f1_weighted", hue="Folder",
    data=data, palette=folder_colors,
    linewidth=2.0, fliersize=4, dodge=True,
    saturation=1.0, hue_order=folders,
    order=classifier_order
)

# Swarmplot with folder colors + black edges
sns.swarmplot(
    x="Classifier", y="f1_weighted", hue="Folder", data=data,
    palette=folder_colors,
    dodge=True, size=5, edgecolor="black", linewidth=0.8, alpha=0.9,
    hue_order=folders,
    order=classifier_order
)

# Keep only one legend and make text bold
handles, labels = box.get_legend_handles_labels()
plt.legend(handles[:len(folders)], [f"{lab}" for lab in labels[:len(folders)]],
           title="Features", loc="upper right", fontsize=12, title_fontsize=13, frameon=True)
plt.setp(plt.gca().get_legend().get_texts(), fontweight='bold')
plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

plt.title(f"F1-weighted Distribution per Classifier", fontsize=16, weight="bold")
plt.xlabel("Classifier", fontsize=14, weight="bold")
plt.ylabel("F1-weighted", fontsize=14, weight="bold")
plt.xticks(fontsize=12, weight="bold")
plt.yticks(fontsize=12, weight="bold")
plt.ylim(0, 1)

plt.tight_layout()
boxplot_path = os.path.join(base_dir, f"boxplot_F1_UMAP{umap_k}_publication.png")
plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"✅ Boxplot saved at: {boxplot_path}")

# ==========================================
# ----------- Barplot (Mean F1) -----------
# ==========================================
mean_df = data.groupby(["Folder", "Classifier"])["f1_weighted"].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(
    x="Classifier", y="f1_weighted", hue="Folder",
    data=mean_df, palette=folder_colors,
    edgecolor="black", dodge=True, width=0.7,
    hue_order=folders,
    order=classifier_order
)

plt.title(f"Mean F1-weighted per Classifier", fontsize=16, weight="bold")
plt.xlabel("Classifier", fontsize=14, weight="bold")
plt.ylabel("Mean F1-weighted", fontsize=14, weight="bold")
plt.xticks(fontsize=12, weight="bold")
plt.yticks(fontsize=12, weight="bold")
plt.ylim(0, 1)

# Legend inside upper right and bold
plt.legend(title="Features", fontsize=12, title_fontsize=13,
           loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
plt.setp(plt.gca().get_legend().get_texts(), fontweight='bold')
plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

plt.tight_layout()
barplot_path = os.path.join(base_dir, f"barplot_mean_F1_UMAP{umap_k}_publication.png")
plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"✅ Barplot saved at: {barplot_path}")
