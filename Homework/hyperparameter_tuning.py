# hyperparameter_tuning.py
import os
import csv
import itertools
import time
import json
import numpy as np
from datetime import datetime
from model import NeuralNetwork
from trainer import Trainer, EarlyStopper
from utils import save_model_params, load_model
from config import EXPERIMENT_DIR, TRAINING_CONFIG
from visualization import plot_learning_curves

class HyperparameterTuner:
    def __init__(self, param_grid):
        self.param_grid = self._validate_param_grid(param_grid)
        self.results = []
        self.best_acc = 0.0
        self.best_exp_info = None

    def _validate_param_grid(self, param_grid):
        """验证参数网格格式"""
        validated = {}
        for k, v in param_grid.items():
            if not isinstance(v, list):
                raise ValueError(f"参数 '{k}' 必须是列表类型，当前类型: {type(v)}")
            validated[k] = v
        return validated

    def search(self, X_train, y_train, X_val, y_val):
        """执行超参数搜索"""
        param_combinations = list(self._generate_param_combinations())
        total = len(param_combinations)
        print(f"开始超参数搜索，共 {total} 组参数组合")

        for idx, params in enumerate(param_combinations, 1):
            exp_start = time.time()
            exp_id = f"exp_{idx:03d}"
            print(f"\n正在训练参数组合 {idx}/{total} [{exp_id}]")

            # 创建实验目录
            exp_dir = self._create_experiment_dir(params, exp_id)
            print(f"实验目录: {os.path.abspath(exp_dir)}")

            # 训练模型
            model, train_result = self._train_single_model(params, X_train, y_train, X_val, y_val)

            # 保存实验数据
            self._save_experiment_data(model, exp_dir, params, train_result)

            # 更新最佳结果
            if train_result['best_val_acc'] > self.best_acc:
                self.best_acc = train_result['best_val_acc']
                self.best_exp_info = {
                    'exp_id': exp_id,
                    'params': params,
                    'dir': exp_dir,
                    'metrics': train_result
                }

            print(f"完成 [{exp_id}] 耗时: {time.time()-exp_start:.1f}s")
        
        print(f"\n最佳验证准确率: {self.best_acc:.4%}")
        print(f"最佳实验目录: {self.best_exp_info['dir']}")
        return self.best_exp_info

    def _generate_param_combinations(self):
        """生成参数组合列表（不再使用生成器）"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _create_experiment_dir(self, params, exp_id):
        """创建位于项目根目录下的实验目录"""
        dir_name = f"{exp_id}_hs{params['hidden_size']}_act{params['activation']}"
        exp_dir = os.path.join(EXPERIMENT_DIR, dir_name)  # 使用配置中的路径
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _train_single_model(self, params, X_train, y_train, X_val, y_val):
        # 构建网络结构参数
        input_size = X_train.shape[1]
        num_classes = y_train.shape[1]
        layer_sizes = [
            input_size,
            params['hidden_size'],
            params['hidden_size'],
            num_classes
        ]
        activations = [params['activation'], params['activation']]
        init_methods = [params['init_method']] * 3  # 3个全连接层需要初始化方法

        # 初始化模型
        model = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=activations,
            init_methods=init_methods,
            reg_strength=params['reg_strength']
        )

        # 初始化训练器
        trainer = Trainer(
            model=model,
            lr=params['lr'],
            reg_strength=params['reg_strength'],
            lr_decay=params['lr_decay'],
            decay_every=params['decay_every']
        )

        # 训练循环
        early_stopper = EarlyStopper(patience=TRAINING_CONFIG["patience"])
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr_history': [],
            'epoch_times': []
        }

        print(f"参数组合: {params}")
        print("┌─────────┬────────────┬────────────┬────────────┬────────────┐")
        print("│  Epoch  │ Train Loss │  Val Loss  │  Val Acc   │    LR      │")
        print("├─────────┼────────────┼────────────┼────────────┼────────────┤")

        for epoch in range(TRAINING_CONFIG["max_epochs"]):  # 最大100个epoch
            epoch_start = time.time()

            # 训练一个epoch
            train_loss = trainer.train_epoch(X_train, y_train, params['batch_size'])

            # 验证
            val_acc = trainer.validate(X_val, y_val)
            val_loss = model.compute_loss(y_val, params['reg_strength'])

            # 记录指标
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['lr_history'].append(trainer.lr)
            metrics['epoch_times'].append(time.time() - epoch_start)

            # 打印进度
            print(f"│ {epoch+1:^7} │ {train_loss:^10.4f} │ {val_loss:^10.4f} │ {val_acc:^10.4%} │ {trainer.lr:^10.4f} │")

            # 学习率衰减
            if (epoch + 1) % params['decay_every'] == 0:
                trainer.apply_lr_decay()

            # 早停检查
            if early_stopper.should_stop(val_acc):
                print(f"EarlyStopping 早停触发于 epoch {epoch+1}")
                break

        print("└─────────┴────────────┴────────────┴────────────┴────────────┘")
        return model, {
            'best_val_acc': early_stopper.best_acc,
            'total_epochs': len(metrics['train_loss']),
            'metrics': metrics
        }

    def _save_experiment_data(self, model, exp_dir, params, train_result):
        """保存所有实验数据"""
        # 保存模型参数
        save_model_params(model, exp_dir)

        # 保存配置文件
        config = {
            'experiment_id': os.path.basename(exp_dir),
            'start_time': datetime.now().isoformat(),
            **params,
            'best_val_acc': train_result['best_val_acc'],
            'total_epochs': train_result['total_epochs']
        }
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 保存训练指标
        self._save_training_metrics(exp_dir, params, train_result)
        
        self._save_training_history(exp_dir, train_result['metrics'])
        
        plot_learning_curves({
            'train_loss': train_result['metrics']['train_loss'],
            'val_loss': train_result['metrics']['val_loss'],
            'val_acc': train_result['metrics']['val_acc']
            }, exp_dir)

    def _save_training_metrics(self, exp_dir, params, train_result):
        """生成详细的训练指标CSV"""
        csv_path = os.path.join(exp_dir, 'training_metrics.csv')
        fieldnames = [
            'epoch', 'hidden_size', 'activation', 'init_method',
            'batch_size', 'lr', 'reg_strength', 'lr_decay', 'decay_every',
            'train_loss', 'val_loss', 'val_acc', 'current_lr', 'epoch_time'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(train_result['total_epochs']):
                writer.writerow({
                    'epoch': i+1,
                    'hidden_size': params['hidden_size'],
                    'activation': params['activation'],
                    'init_method': params['init_method'],
                    'batch_size': params['batch_size'],
                    'lr': params['lr'],
                    'reg_strength': params['reg_strength'],
                    'lr_decay': params['lr_decay'],
                    'decay_every': params['decay_every'],
                    'train_loss': train_result['metrics']['train_loss'][i],
                    'val_loss': train_result['metrics']['val_loss'][i],
                    'val_acc': train_result['metrics']['val_acc'][i],
                    'current_lr': train_result['metrics']['lr_history'][i],
                    'epoch_time': train_result['metrics']['epoch_times'][i]
                })
                
    def _save_training_history(self, exp_dir, metrics):
        """保存训练历史到NPZ文件"""
        npz_path = os.path.join(exp_dir, "training_history.npz")
        np.savez(
            npz_path,
            train_loss=np.array(metrics['train_loss']),
            val_loss=np.array(metrics['val_loss']),
            val_acc=np.array(metrics['val_acc']),
            lr_history=np.array(metrics['lr_history']),
            epoch_times=np.array(metrics['epoch_times'])
        )
        print(f"训练历史已保存至 {npz_path}")