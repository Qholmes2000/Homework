# visualization.py
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_learning_curves(history, save_dir):
    """绘制单个实验的训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', color='blue', linestyle='-')
    plt.plot(history['val_loss'], label='Validation', color='orange', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], color='green', linestyle='-.')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300)
    plt.close()
    
    print(f"训练曲线已保存至 {os.path.join(save_dir, 'learning_curves.png')}")
    
def plot_combined_results(all_results, save_path):
    """绘制所有实验的对比图表"""
    plt.figure(figsize=(15, 8))
    
    # 准确率对比
    plt.subplot(2, 2, 1)
    accuracies = [res['best_val_acc'] for res in all_results]
    labels = [f"Exp{i+1}" for i in range(len(all_results))]
    plt.barh(labels, accuracies, color='teal')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1)
    
    # 损失曲线对比
    plt.subplot(2, 2, 2)
    for i, res in enumerate(all_results):
        plt.plot(res['val_loss'], label=f"Exp{i+1}")
    plt.title('Validation Loss Trajectories')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 超参数热力图
    plt.subplot(2, 2, 3)
    params = [str(res['params']) for res in all_results]
    plt.scatter(
        x=range(len(all_results)),
        y=accuracies,
        c=accuracies,
        cmap='viridis',
        s=100,
        edgecolors='black'
    )
    plt.title('Hyperparameter Performance')
    plt.ylabel('Accuracy')
    plt.xticks([])
    plt.colorbar(label='Accuracy')
    
    # 训练时间分布
    plt.subplot(2, 2, 4)
    times = [np.sum(res['epoch_times']) for res in all_results]
    plt.pie(times, labels=labels, autopct='%1.1f%%')
    plt.title('Training Time Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()