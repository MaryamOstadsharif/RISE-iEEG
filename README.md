[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2408.14477v1)
# RISE-iEEG: Robust to Inter-Subject Electrodes Implantation Variability iEEG Classifier
RISE-iEEG, a novel decoder model specifically designed to tackle the challenges posed by electrode implantation variability across subjects.
## Table of Contents
* [General Information](#general-information)
* [Getting Started](#getting-started)
* [Settings Configuration](#settings-configuration)
* [Prepare Data](#prepare-data)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
* [Contributing](#contributing)
* [License](#license)
<br/>

## General Information
By effectively addressing the challenges of electrode implantation variability, RISE-iEEG enhances classification accuracy without compromising on interpretability or generalization. Its architecture eliminates the need for precise electrode coordinate data, broadening its applicability in various neural decoding tasks and potentially advancing the field of iEEG-based research and applications.
## Requirements

- Python 3.x
- CUDA-enabled GPU
- Necessary libraries (see `requirements.txt`)

## Getting Started

1. Clone this repository to your local machine.

2. Install the required dependencies. `pip install -r requirements.txt`

3. Prepare your dataset

4. Create the `./configs/settings.yaml` according to `./cinfigs/settings_sample.yaml`

5. Create the `./configs/device_path.yaml` according to `./cinfigs/device_path_sample.yaml`

6. Run the `main.py` script to execute the model.

## Prepare Data
1. Download the data
   - Audio Visual : [link]
   - Upper Limb movement: [link]
2. To load data, you need two folder:
   - raw data folder : Copy the downloaded files into folling folder:
        - aa
        - bb
        - cc
   - preprocessed data folder : the model save preprocessed data to this folder in following subfolders:
     - Speech_Music
     - Singing_Music
     - Question_Answer
     - move_rest
3. Set the path to this folder in `device_path.yaml` as `raw_dataset_path` and `preprocessed_dataset_path`

## Settings Configuration

The `settings.yaml` file is used to configure the various parameters for the iEEG decoding model. Below is an explanation of each setting:

### Data Set Configuration

- **`task`**: Specifies the task to be performed. Options include:
  - `Question_Answer`
  - `Singing_Music`
  - `Speech_Music`
  - `move_rest`
  - Example: `task: 'Singing_Music'`
  
- **`load_preprocessed_data`**: Boolean flag indicating whether to load preprocessed data.
  - Example: `load_preprocessed_data: true`
  
- **`st_num_patient`**: Index of the first patient in the dataset.
  - Example: `st_num_patient: 0`
  
- **`num_patient`**: The number of patients in the dataset.
  - Example: `num_patient: 29`
  
- **`n_channels_all`**: Maximum number of channels.
  - Example: `n_channels_all: 250`

### Model Settings

- **`hyper_param`**: Dictionary containing hyperparameters for the model:
  - **`F1`**: Filter size for the first convolutional layer. Example: `5`
  - **`dropoutRate`**: Dropout rate to prevent overfitting. Example: `0.542`
  - **`kernLength`**: Kernel length for the first convolutional layer. Example: `60`
  - **`kernLength_sep`**: Kernel length for the separable convolutional layer. Example: `88`
  - **`dropoutType`**: Type of dropout used. Options: `Dropout`, `AlphaDropout`, `SpatialDropout2D`. Example: `'Dropout'`
  - **`D`**: Depth multiplier for separable convolutions. Example: `2`
  
- **`n_ROI`**: Number of Regions of Interest (ROIs) considered in the model.
  - Example: `n_ROI: 20`
  
- **`coef_reg`**: Regularization coefficient to prevent overfitting.
  - Example: `coef_reg: 0.01`

### Evaluation Settings

- **`n_folds`**: Number of folds for cross-validation. Use `1` for one patient out cross-validation.
  - Example: `n_folds: 10`
  
- **`one_patient_out`**: Boolean flag to indicate whether to use one patient out cross-validation.
  - Example: `one_patient_out: False`
  
- **`Unseen_patient`**: Boolean flag to indicate whether to use the 'Unseen_patient' scenario.
  - Example: `Unseen_patient: False`

- **`type_balancing`**: Type of data balancing applied.
  - Options: `no_balancing`, `over_sampling`
  - Example: `type_balancing: 'over_sampling'`

### Training Options

- **`epochs`**: Number of training epochs.
  - Example: `epochs: 300`
  
- **`patience`**: Number of epochs with no improvement before stopping early.
  - Example: `patience: 20`
  
- **`early_stop_monitor`**: Metric to monitor for early stopping. Options: `val_loss`, `val_accuracy`.
  - Example: `early_stop_monitor: 'val_accuracy'`
  
- **`optimizer`**: Optimizer used for training. Options: `adam`, `sgd`, `rmsprop`.
  - Example: `optimizer: 'adam'`
  
- **`loss`**: Loss function used for training. Options: `categorical_crossentropy`, `binary_crossentropy`, `mse`.
  - Example: `loss: 'categorical_crossentropy'`

### Additional Settings

- **`del_temporal_lobe`**: Boolean flag to indicate whether to exclude the temporal lobe from the analysis.
  - Example: `del_temporal_lobe: False`

## Repository Structure
This repository is organized as follows:

- `/main.py`: The main script to run.

- `/results/`: This folder is generated when you first run the model. The results are saved with time stamp if the `debug_mode: true` is in `settings.yaml`

- `/src/`: All codes are located in this folder

- `/src/data`: Contains scripts for data loading (`data_loader.py`, `data_preprocessor.py`) to load the data, analyze it and preprocess it.

- `/src/experiments/`: Contains the `explainatory_data_analysis` script for checking the EDA and play with data! This folder contains scripts for different experiments

- `/src/model/`: Contains the functions required to build, train and evaluate models.

- `/src/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `/src/utils`: Contains utility script `utils.py` for some helper function that are used in code.

- `/visualization`: Contains the `vizualize_utils.py` script for data and result visualization.
<br/>



## Citations
The code contained in this repository for BTsC is companion to the paper:  

```
@InProceedings{Maryam2024,
  title = {RISE-iEEG: Robust to Inter-Subject Electrodes Implantation Variability iEEG Classifier},
  author = {Maryam Ostadsharif Memar, Navid Ziaei, Behzad Nazari, Ali Yousefi},
  url = {https://arxiv.org/abs/xxx},
  year = {2024},
}
```
which should be cited for academic use of this code.  
<br/>

## Contributing

We encourage you to contribute to RISEiEEG! 

## License

This project is licensed under the terms of the MIT license.
