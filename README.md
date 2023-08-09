# Master thesis - Deep Learning-based Scale Estimation of Local Image Features

## Implementation

### Install the following libraries:

This work is developed in **Python**, and the following libraries need to be installed:

- OpenCV (to use the SURF algorithm, old OpenCV libraries need to be installed)
- NumPy
- Pandas
- Tensorflow
- Scikit-learn

Further, dataset generation and evaluation were carried out with a Jupyter Notebook.

### Code

The dataset generation is provided in `dataset-generation.ipynb` in the directory `notebooks/`. This notebook creates the training and test datasets using SIFT and SURF algorithms in .JSON format. The custom data generator used in this thesis is available in `data.py` in the directory `scripts/`. This custom data generator outputs pre-processed batches of data used to train different network architectures. The modified ResNet34 architecture implemented in this work is available in `model.py` with the **Mod-ResNet34** function in the directory `scripts/`. Further, the training configuration and the training script is available in `train.py` in the directory `scripts/`. The loss function, optimizer, metrics, and hyperparameters can be modified here. The evaluation process and the necessary code are provided in `evaluation.ipynb` in `notebooks/`. Two trained models with weights are provided in `saved-model/`, each trained with SIFT and SURF with Mod-ResNet34 architecture, respectively.


Change the path of the files accordingly.


Example call to the training script from the terminal: 
```python -m scripts.train from PATH```





