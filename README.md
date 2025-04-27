# Neural Network Hyperparameter Tuning for CIFAR-10 Classification

## Introduction
This project focuses on hyperparameter tuning for a neural network to classify images from the CIFAR-10 dataset. It includes a comprehensive pipeline from data loading, model training, hyperparameter search, to result visualization and model evaluation.

## Features
- **Data Loading**: Loads and preprocesses the CIFAR-10 dataset, including flattening, normalization, and one-hot encoding of labels.
- **Model Definition**: Defines a neural network model with customizable layer sizes, activation functions, and weight initialization methods.
- **Hyperparameter Tuning**: Implements a grid search approach to find the best hyperparameters for the model.
- **Training and Early Stopping**: Trains the model with early stopping based on validation accuracy to prevent overfitting.
- **Visualization**: Plots learning curves and combined results for all experiments to facilitate analysis.
- **Model Evaluation**: Evaluates the best model on the test set and archives the best model for future use.

## Directory Structure
```
project_root/
├── experiments/         # Directory to store experiment results
├── model/               # Directory to store model files
│   └── best_model/      # Directory to archive the best model
├── config.py            # Configuration file for hyperparameters and directories
├── hyperparameter_tuning.py  # Hyperparameter tuning module
├── model.py             # Neural network model definition
├── trainer.py           # Training module
├── utils.py             # Utility functions for data loading and model saving
├── visualization.py     # Visualization module
├── shiyan_main.py       # Main script to run the entire pipeline
└── test.py              # Script to evaluate the model on the test set
```

## Installation
To run this project, you need to have the following libraries installed:
- `numpy`
- `matplotlib`
- `sklearn`
- `keras`
- `tensorflow` (required by `keras` to load CIFAR-10 dataset)

You can install these libraries using `pip`:
```bash
pip install numpy matplotlib scikit-learn keras tensorflow
```

## Usage
### 1. Configuration
Modify the `PARAM_GRID` and `TRAINING_CONFIG` in `config.py` to define the hyperparameters and training settings for the experiments.

### 2. Run the Main Script
Execute the `shiyan_main.py` script to start the hyperparameter tuning process:
```bash
python shiyan_main.py
```
This script will perform the following steps:
- Load the CIFAR-10 dataset.
- Initialize the hyperparameter tuner.
- Perform a grid search to find the best hyperparameters.
- Archive the best model to the `best_model` directory.
- Evaluate the best model on the test set and save the final results.

### 3. Visualize Results
The training curves for each experiment will be saved in the corresponding experiment directory. The combined results for all experiments will be saved in the specified location. You can find these visualizations in the `experiments` directory.

### 4. Evaluate the Model
You can use the `test.py` script to evaluate the model on the test set with a specific configuration and model path:
```bash
python test.py
```

## Code Explanation
### `config.py`
This file contains the configuration settings for the project, including hyperparameter grids, training configurations, and directory paths.

### `hyperparameter_tuning.py`
Defines the `HyperparameterTuner` class, which performs a grid search over the hyperparameter space. It trains multiple models with different hyperparameter combinations, saves the training results, and selects the best model based on validation accuracy.

### `model.py`
Defines the neural network model, including fully connected layers, activation layers, and the softmax function. It also provides methods for forward propagation, backward propagation, and loss computation.

### `trainer.py`
Contains the `Trainer` class for model training and the `EarlyStopper` class for early stopping. The `Trainer` class performs forward and backward propagation, parameter updates, and learning rate decay.

### `utils.py`
Provides utility functions for data loading, model saving, and experiment directory creation.

### `visualization.py`
Contains functions to plot learning curves and combined results for all experiments.

### `shiyan_main.py`
The main script that orchestrates the entire pipeline, including data loading, hyperparameter tuning, model evaluation, and result archiving.

### `test.py`
A script to evaluate the model on the test set with a specific configuration and model path.

## License
This project is licensed under the [MIT License](LICENSE).
