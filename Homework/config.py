# config.py
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 定义实验目录路径
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments")
BEST_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "best_model")

PARAM_GRID = {
    "hidden_size": [256, 512],
    "lr": [1e-2, 1e-3],
    "reg_strength": [0.01, 0.005],
    "init_method": ['he', 'xavier'],
    "activation": ['relu', 'tanh', 'sigmoid'],
    "batch_size": [128, 256],
    "lr_decay": [0.95],
    "decay_every": [10], 
}

TRAINING_CONFIG ={"max_epochs":70,
                "patience": 15,        # 早停耐心值
                "base_dir": os.path.join(PROJECT_ROOT, "experiments"),
                "best_model_dir": os.path.join(PROJECT_ROOT, "model/best")
                }
"""

PARAM_GRID = {
    "hidden_size": [256],
    "lr": [1e-3],
    "reg_strength": [0.001],
    "init_method": ['he'],
    "activation": ['relu'],
    "batch_size": [128],
    "lr_decay": [0.95],
    "decay_every": [10], 
    "max_epochs":2
}
"""

DATA_CONFIG = {
    "num_classes": 10,
    "input_dim": 3072  # 32x32x3 for CIFAR-10
}

# 实验目录配置
EXPERIMENT_CONFIG = {
    "base_dir": os.path.join(PROJECT_ROOT, "experiments"),
    "model_dir": os.path.join(PROJECT_ROOT, "model"),
    "best_model_dir": os.path.join(PROJECT_ROOT, "model", "best_model")
}