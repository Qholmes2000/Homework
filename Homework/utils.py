# utils.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from keras.datasets import cifar10
from model import FCLayer
from datetime import datetime
import json, csv
from visualization import plot_learning_curves
from config import EXPERIMENT_DIR, BEST_MODEL_DIR

def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
    
    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    # One-hot encode labels (修复部分)
    encoder = OneHotEncoder(sparse_output=False)  # 使用新版本参数名称
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_model(model, path):
    params = []
    for layer in model.layers:
        if isinstance(layer, FCLayer):
            params.append(layer.W)
    np.savez(path, *params)

def load_model(model, path):
    params = np.load(path)
    i = 0
    for layer in model.layers:
        if isinstance(layer, FCLayer):
            layer.W = params[f'arr_{i}']
            i += 1

def create_log_dir(base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def create_experiment_dir(params, base_dir="experiments"):
    """创建实验目录并返回路径"""
    project_root = os.path.abspath(os.path.dirname(__file__))
    # 生成目录名称
    folder_name = "_".join([
        f"hs{params['hidden_size']}",
        f"act{params['activation']}",
        f"bs{params['batch_size']}",
        f"lr{params['lr']:.0e}".replace('.', '').replace('e-0', 'e-'),
        f"reg{params['reg_strength']:.0e}".replace('.', '').replace('e-0', 'e-'),
        f"init{params['init_method']}",
        datetime.now().strftime("%m%d%H%M")
    ])
    exp_dir = os.path.join(project_root, base_dir, folder_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_training_metrics(exp_dir, metrics):
    """保存训练指标到CSV"""
    csv_path = os.path.join(exp_dir, "training_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

def save_config(exp_dir, config):
    """保存配置文件到当前实验目录"""
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def save_training_history(exp_dir, history):
    """保存训练历史"""
    npz_path = os.path.join(exp_dir, "training_history.npz")
    np.savez(npz_path, 
             train_loss=history['train_loss'],
             val_loss=history['val_loss'],
             val_acc=history['val_acc'])

def save_model_params(model, exp_dir):
    """保存模型参数到实验目录"""
    params_path = os.path.join(exp_dir, "model_params.npz")
    params = []
    for layer in model.layers:
        if isinstance(layer, FCLayer):
            params.append(layer.W)
    np.savez(params_path, *params)

def save_experiment(exp_dir, model, config, metrics, history):
    """保存实验完整数据"""
    # 保存配置
    save_config(exp_dir, config)
    
    # 保存模型参数
    save_model_params(model, exp_dir)
    
    # 保存训练指标
    save_training_metrics(exp_dir, metrics)
    
    # 保存训练历史
    np.savez(os.path.join(exp_dir, 'training_history.npz'), **history)
    
    # 保存可视化图表
    plot_learning_curves(history, exp_dir)