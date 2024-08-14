import os
import json
from pathlib import Path
import datetime
import random
import numpy as np
import tensorflow as tf

def configure_environment():
    # Set environment variables to specify which GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use

    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

