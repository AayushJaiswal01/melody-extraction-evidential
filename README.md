# Evidential Deep Learning for Melody Extraction

This repository contains the official source code for the paper: "[Your Paper Title Here]". We provide implementations for the proposed models (M3 Evidential Regression, M2 Confidence Model) and the baseline model (M1 Beta-NLL).

## Abstract

[Paste the abstract of your paper here. This gives readers immediate context about your work.]

## Repository Structure

-   `training/`: Contains the main training scripts for each model.
    -   `M1_beta_nll.py`: The β-NLL baseline model.
    -   `M2_confidence_model.py`: The two-stage baseline classification and confidence model.
    -   `M3_evidential_regression.py`: The proposed single-head evidential regression model.
-   `requirements.txt`: A list of required Python packages to run the code.

## Setup

### 1. Environment

We recommend using a conda environment.

```bash
# Create and activate a new conda environment
conda create -n melody python=3.10
conda activate melody

# Install required packages
pip install -r requirements.txt
```

### 2. Data Preparation

This code expects pre-processed audio and pitch data in `.npy` format. The data should be structured as follows:

```
/path/to/your/dataset/
├── train/
│   ├── audio/
│   │   ├── file1.npy
│   │   └── ...
│   └── pitch/
│       ├── file1.npy
│       └── ...
├── val/
│   ├── audio/
│   └── pitch/
└── test/
    ├── audio/
    └── pitch/
```

The audio `.npy` files should contain spectrograms of shape `(100, 1025)`. The pitch `.npy` files should contain ground truth frequency values in Hz of shape `(100,)`.

*Note: The script for pre-processing raw `.wav` files into this format can be found in the `data_preprocessing/` directory.*

## Training the Models

To train a model, navigate to the `training/` directory and run the desired script. You must edit the script to point to your dataset directory.

**Before running, open the script and modify the `DATA_DIR` variable:**

```python
# --- PATHS ---
# !!! IMPORTANT: CHANGE THIS PATH !!!
DATA_DIR = '/path/to/your/dataset/'
```

**To run training for the M3 model:**

```bash
cd training/
python M3_evidential_regression.py
```

Model weights will be saved automatically in a `saved_models_*` directory created within the `training/` folder.

## Citation

If you use this code in your research, please consider citing our paper:

```bibtex
@article{YourLastNameYEAR,
  title={[Your Paper Title Here]},
  author={[Your Name] and [Co-author Names]},
  journal={[Journal or Conference Name]},
  year={[Year]}
}
```
