# Master thesis - Deep Learning-based Scale Estimation of Local Image Features

## Overview

This repository contains the implementation of a Deep Learning-based approach for scale estimation of local image features, developed using Python. It includes scripts for dataset generation, custom data generators, a modified ResNet34 architecture, training configurations, and evaluation notebooks.

## Prerequisites
Before starting, ensure the following libraries are installed:

OpenCV: Required for the SIFT, SURF algorithm. Note that older versions of OpenCV are needed for compatibility.
NumPy
Pandas
Tensorflow
Scikit-learn


## Repository Structure
### Notebooks

dataset-generation.ipynb: Located in notebooks/, this notebook generates training and test datasets using SIFT and SURF algorithms, outputting them in .JSON format.
### Scripts

data.py: Found in scripts/, this script includes a custom data generator for preprocessing and batching data for training.
model.py: This script, located in scripts/, contains the implementation of the modified ResNet34 architecture, named Mod-ResNet34.
train.py: Also in scripts/, this script is used for training configurations. It allows modifications to the loss function, optimizer, metrics, and hyperparameters.

### Evaluation
evaluation.ipynb: Available in notebooks/, this notebook includes the evaluation process and necessary code.

### Saved Models
Two trained models, each for SIFT and SURF algorithms using the Mod-ResNet34 architecture, are stored in saved-model/.

## Implementation

### Install the following libraries:

This work is developed in **Python**, and the following libraries need to be installed:

- OpenCV (to use the SURF algorithm, old OpenCV libraries need to be installed)
- NumPy
- Pandas
- Tensorflow
- Scikit-learn

## Usage
Change file paths as needed for your setup.


### Example Command for Training

```bash
python -m scripts.train from PATH
```
Replace PATH with the appropriate directory path for your files.




