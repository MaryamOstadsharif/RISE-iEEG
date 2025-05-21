[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2408.14477v1)
# RISE-iEEG: Robust to Inter-Subject Electrodes Implantation Variability iEEG Classifier

## Table of Contents
* [General Information](#general-information)
* [Reference](#reference)
* [Getting Started](#getting-started)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
<br/>

## General Information
RISE-iEEG is a robust neural decoding model designed to overcome inter-subject variability in electrode placement within intracranial EEG (iEEG) data. Traditional models struggle to generalize across patients due to differences in electrode implantation, limiting their practical use. RISE-iEEG addresses this by using a patient-specific projection network that maps neural data into a shared low-dimensional space, allowing a shared classifier to decode signals consistently across subjects—without needing electrode location data. This makes RISE-iEEG well-suited for brain-computer interface and clinical applications, enabling accurate, cross-patient decoding in diverse cognitive and motor tasks.

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2024](https://arxiv.org/abs/2408.14477v1). Cite this paper using its [DOI](https://arxiv.org/abs/2408.14477v1).

## Getting Started
1. Clone the Repository 
`git clone https://github.com/MaryamOstadsharif/RISE-iEEG.git`

2. Install the required dependencies. `pip install -r requirements.txt`

3. Prepare your dataset

4. Create the `./configs/settings.yaml` according to `./cinfigs/settings_sample.yaml`

5. Create the `./configs/device_path.yaml` according to `./cinfigs/device_path_sample.yaml`

6. Run the `main.py` script to execute the model.

## Repository Structure
This repository is organized as follows:

- `/main.py`: The main script to run.

- `/src/preprocessing`: Contains scripts for data loading and preprocessing.

- `/src/experiments/`: Contains scripts for different experiments.
  
- `/src/model/`: Contains the functions required to build, train and evaluate models.

- `/src/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `/src/utils`: Contains utility script `utils.py` for some helper function that are used in code.

- `/src/interpretation`: Contains script to run the Integrated Gradients method for model interpretation.
  
- `src/visualization`: Contains the `vizualize_utils.py` script for data and result visualization.
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
