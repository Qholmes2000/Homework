# trainer.py
import numpy as np
import time
from model import NeuralNetwork

class Trainer:
    def __init__(self, model, lr=1e-3, reg_strength=1e-4, lr_decay=0.95, decay_every=10):
        self.model = model
        self.lr = lr
        self.reg_strength = reg_strength
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.lr_history = [lr]

    def train_epoch(self, X_train, y_train, batch_size):
        """训练一个epoch"""
        num_train = X_train.shape[0]
        indices = np.random.permutation(num_train)
        total_loss = 0

        for i in range(0, num_train, batch_size):
            # 获取批次数据
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # 前向传播
            scores = self.model.forward(X_batch)

            # 计算梯度
            grad = self._compute_gradients(scores, y_batch)

            # 反向传播
            self.model.backward(grad)

            # 参数更新
            self._update_parameters()

            # 计算损失
            loss = self.model.compute_loss(y_batch, self.reg_strength)
            total_loss += loss * X_batch.shape[0]

        # 记录学习率
        self.lr_history.append(self.lr)
        return total_loss / num_train

    # trainer.py 的 _compute_gradients 方法修改
    def _compute_gradients(self, scores, y_true):
        """计算输出层梯度"""
        m = y_true.shape[0]
        
        # 计算概率分布
        probs = NeuralNetwork.softmax(scores)
        
        # 仅当 y_true 是 one-hot 编码时使用此方式
        grad = probs - y_true
        
        # 如果 y_true 是类别索引 (shape: (m,))，应使用：
        # grad = probs
        # grad[range(m), y_true] -= 1
        
        return grad / m

    def _update_parameters(self):
        """执行SGD参数更新"""
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                # 应用L2正则化
                layer.W -= self.lr * (layer.dW + self.reg_strength * layer.W)
                layer.b -= self.lr * layer.db

    def apply_lr_decay(self):
        """应用学习率衰减"""
        self.lr *= self.lr_decay
        print(f"学习率衰减至: {self.lr:.2e}")

    def validate(self, X_val, y_val):
        """在验证集上评估"""
        y_pred = self.model.predict(X_val)
        y_true = np.argmax(y_val, axis=1)
        return np.mean(y_pred == y_true)

class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_acc = -np.inf

    def should_stop(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience