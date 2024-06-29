# MultiPatient Model Training

This repository contains the code for training a MultiPatient model with customizable settings and environment configurations.

## Requirements

- Python 3.x
- CUDA-enabled GPU
- Necessary libraries (see `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multipatient_model.git
    cd multipatient_model
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Setting up Environment Variables

Specify which GPU to use by setting environment variables:

```python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use
