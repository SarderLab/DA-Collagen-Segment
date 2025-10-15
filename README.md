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

## Configuration File Overview
Each experiment in this repository is defined by a JSON configuration file. This file controls dataset paths, model architecture, preprocessing, and training parameters.

### General Settings
| Parameter | Description                                                                                                                                                                                                                             |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`    | Specifies the training strategy. Common options include:<br>• `source_only` — train only on the source domain.<br>• `ADDA` — perform adversarial domain adaptation.<br>• `multiDA` — train with multiple target domains simultaneously. |


### input_parameters
These define dataset locations, training phase, and logging configuration.

| Parameter          | Description                                                                                                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `phase`            | Training mode — usually `"train"` or `"test"`.                                                                                                                                      |
| `type`             | Indicates if the training is single-domain (`"single"`) or multi-domain (`"multi"`).                                                                                                |
| `overlapRate`      | Amount of overlap between image patches during data generation.                                                                                                                     |
| `image_dir`        | Dictionary containing source domain paths:<br>• `DUET`: path to fluorescent (DUET) source images.<br>• `Brightfield`: path to brightfield source images.                            |
| `target_dir`       | Dictionary of target domain paths used for domain adaptation.<br>Each entry lists one or more folders for new target datasets.                                                      |
| `label_dir`        | Path to ground-truth segmentation masks for the source domain.                                                                                                                      |
| `target_label_dir` | List of target domain label paths (optional — may be empty for unsupervised adaptation).                                                                                            |
| `model_file`       | Path to a pretrained or previously saved model checkpoint (optional).                                                                                                               |
| `output_dir`       | Directory where training results, logs, and model checkpoints are saved.                                                                                                            |
| `neptune`          | Optional Neptune.ai tracking setup:<br>• `project`: project name for logging.<br>• `source_files`: file patterns to include in tracking.<br>• `tags`: experiment tags for tracking. |

### train_parameters
These parameters define how the model is trained, including architecture, loss functions, and optimization.

| Parameter          | Description                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`             | Repeats the main training mode (e.g., `ADDA`) for clarity.                                                                                           |
| `model_details`    | Nested parameters describing the model architecture. See below.                                                                                      |
| `train_test_split` | Defines how the dataset is split:<br>• `k-fold`: number of folds for cross-validation.<br>• `split`: train/test split ratio (e.g., 0.8 = 80% train). |
| `loss`             | Specifies the loss function used for evaluation (e.g., `"MSE"`, `"Dice"`, `"BCE"`).                                                                  |
| `train_loss`       | Defines the main training loss (e.g., `"structural_loss"`).                                                                                          |
| `lr`               | Learning rate used for optimization.                                                                                                                 |
| `batch_size`       | Number of samples per batch.                                                                                                                         |
| `step_num`         | Total number of training steps or iterations.                                                                                                        |
| `save_step`        | Frequency (in epochs or steps) for saving model checkpoints.                                                                                         |
| `k_disc`           | Number of discriminator training steps per generator step (used in adversarial training).                                                            |
| `k_gen`            | Number of generator steps per discriminator step.                                                                                                    |

### model_details

| Parameter         | Description                                                                             |
| ----------------- | --------------------------------------------------------------------------------------- |
| `architecture`    | Specifies the model backbone or composite design (e.g., `"ensemble"`, `"unet"`).        |
| `encoder`         | Encoder backbone network (e.g., `"resnet34"`, `"efficientnet-b0"`).                     |
| `encoder_weights` | Pretraining weights for the encoder (`"imagenet"` or `"None"`).                         |
| `dropout_rate`    | Dropout rate applied to prevent overfitting.                                            |
| `pretrained`      | Set to `1` to load pretrained encoder weights.                                          |
| `active`          | Activation function used for the final output layer (e.g., `"sigmoid"` or `"softmax"`). |
| `target_type`     | `"binary"` or `"nonbinary"` segmentation output.                                        |
| `in_channels`     | Number of input channels (e.g., 2 for dual-input DUET/Brightfield).                     |
| `ann_classes`     | Comma-separated class names (e.g., `"background,collagen"`).                            |

### preprocessing
| Parameter         | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `image_size`      | Size of the input image patches (e.g., `"512,512,2"`).                           |
| `mask_size`       | Size of the segmentation mask patches (e.g., `"512,512,1"`).                     |
| `color_transform` | Defines preprocessing or channel selection method (e.g., `"multi_input_green"`). |

### Consistency and Adaptation Parameters
| Parameter            | Description                                                                    |
| -------------------- | ------------------------------------------------------------------------------ |
| `cons_weight`        | Weight for the consistency loss term (used in semi-supervised or DA training). |
| `consistency_rampup` | Number of iterations to gradually increase consistency weight.                 |
| `target_alpha`       | Momentum update rate for exponential moving average (EMA) in target encoder.   |
| `target_alpha_late`  | Late-stage EMA rate for more stable adaptation.                                |
| `target_late_epoch`  | Epoch after which `target_alpha_late` takes effect.                            |

### Training
Once your configuration file (e.g., Config/Example_ADDA.json) is created, you can start training with:
``` bash
python TrainmultiDA.py --config Config/Example_ADDA.json
```

Or submit via SLURM:
``` bash
sbatch scripts/run_train.sh
```

### Predicting
``` bash
python main.py /Config_path
```







