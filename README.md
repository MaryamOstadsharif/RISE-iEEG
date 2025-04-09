[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2408.14477v1)
# RISE-iEEG: Robust to Inter-Subject Electrodes Implantation Variability iEEG Classifier
RISE-iEEG, a novel decoder model specifically designed to tackle the challenges posed by electrode implantation variability across subjects.

## Table of Contents
* [General Information](#general-information)
* [Reference](#reference)
* [Getting Started](#getting-started)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
<br/>


## General Information
By effectively addressing the challenges of electrode implantation variability, RISE-iEEG enhances classification accuracy without compromising on interpretability or generalization. Its architecture eliminates the need for precise electrode coordinate data, broadening its applicability in various neural decoding tasks and potentially advancing the field of iEEG-based research and applications.

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2024](https://arxiv.org/abs/2408.14477v1). Cite this paper using its [DOI](https://arxiv.org/abs/2408.14477v1).

## Getting Started

1. Clone this repository to your local machine.

2. Install the required dependencies. `pip install -r requirements.txt`

3. Prepare your dataset

4. Create the `./configs/settings.yaml` according to `./cinfigs/settings_sample.yaml`

5. Create the `./configs/device_path.yaml` according to `./cinfigs/device_path_sample.yaml`

6. Run the `main.py` script to execute the model.

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
The code contained in this repository for RISE-iEEG is companion to the paper:  

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

We encourage you to contribute to RISE-iEEG! 

## License

This project is licensed under the terms of the MIT license.
