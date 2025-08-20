# DA-Collagen-Segment

This repository contains code for collagen segmentation using domain
adaptation.

## Getting Started

To get started, clone the repository and install the necessary
dependencies.

``` bash
git clone https://github.com/SarderLab/DA-Collagen-Segment.git
cd DA-Collagen-Segment
pip install -r requirements.txt
```

You can then run `main.py` with the appropriate arguments to train and
test the models.

``` bash
sbatch scripts/run_train.sh
```

## Repository Structure

-   **Config:** Contains configuration files for different experiments.
-   **scripts:** Contains shell scripts to submit the job using Slurm.
-   **Segmentation_Metrics_Pytorch:** Contains PyTorch-based
    segmentation evaluation metrics.
-   **Augmentation_Functions.py:** Defines functions for data
    augmentation.
-   **Evaluate.py:** Script for evaluating trained models.
-   **Loss.py:** Defines the loss functions used for training.
-   **Models.py:** Contains the model architectures.
-   **Train.py:** Script for training the source only models.
-   **TestmultiDA.py:** Script for testing source only or single/multiple domain adaptation
    models.
-   **TrainmultiDA.py:** Script for training multiple domain adaptation
    models.
