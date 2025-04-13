# main.py
import os
import shutil
import json
import numpy as np
from utils import load_cifar10, create_experiment_dir, load_model
from hyperparameter_tuning import HyperparameterTuner
from visualization import plot_learning_curves
import config
from model import NeuralNetwork
from config import EXPERIMENT_DIR, BEST_MODEL_DIR


def evaluate_best_model(best_exp_dir, X_test, y_test):
    """评估最佳模型在测试集上的表现"""
    # 加载配置文件
    config_path = os.path.join(best_exp_dir, 'config.json')
    with open(config_path) as f:
        params = json.load(f)
    
    # 重建模型结构
    model = NeuralNetwork(
        layer_sizes=[
            config.DATA_CONFIG['input_dim'],
            params['hidden_size'], 
            params['hidden_size'],
            config.DATA_CONFIG['num_classes']
        ],
        activations=[params['activation'], params['activation']],
        init_methods=[params['init_method']]*3,
        reg_strength=params['reg_strength']
    )
    
    # 加载模型参数
    model_path = os.path.join(best_exp_dir, 'model_params.npz')
    load_model(model, model_path)
    
    # 计算准确率
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    return np.mean(y_pred == y_true)

def archive_best_model(source_dir, target_dir):
    """归档最佳模型到项目根目录下的指定位置"""
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    # 需要复制的关键文件
    files_to_copy = [
        'config.json',
        'model_params.npz',
        'training_metrics.csv',
        'training_history.npz',
        'learning_curves.png'
    ]
    for fname in files_to_copy:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

def main():
    # 初始化实验目录
    os.makedirs(config.EXPERIMENT_CONFIG['base_dir'], exist_ok=True)
    
    # 加载数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10()
    
    # 初始化超参数调优器
    tuner = HyperparameterTuner(config.PARAM_GRID)
    
    # 执行超参数搜索
    best_exp_info = tuner.search(X_train, y_train, X_val, y_val)
    
    # 归档最佳模型
    best_exp_dir = best_exp_info['experiment_dir']
    archive_target = os.path.join(config.EXPERIMENT_CONFIG['best_model_dir'])
    archive_best_model(best_exp_dir, archive_target)
    
    # 在测试集上评估
    test_acc = evaluate_best_model(archive_target, X_test, y_test)
    
    # 保存最终结果
    final_result = {
        "test_accuracy": float(test_acc),
        "best_params": best_exp_info['params'],
        "source_experiment": best_exp_dir
    }
    with open(os.path.join(archive_target, 'final_result.json'), 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"\n=== 最终测试准确率: {test_acc:.4f} ===")
    print(f"最佳模型已归档至: {archive_target}")

if __name__ == "__main__":
    main()