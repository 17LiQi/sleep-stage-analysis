import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE
from .trainer import Trainer

logger = logging.getLogger(__name__)

class SMOTETrainer(Trainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.smote = SMOTE(
            sampling_strategy='auto',  # 使用自动采样策略
            k_neighbors=5,
            random_state=config.seed
        )
    
    def _apply_smote(self, train_loader):
        """应用SMOTE过采样"""
        # 收集所有训练数据
        all_data = []
        all_labels = []
        for data, labels in train_loader:
            all_data.append(data.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # 合并数据
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # 重塑数据以适应SMOTE
        n_samples, n_channels, n_timesteps = X.shape
        X_reshaped = X.reshape(n_samples, -1)
        
        # 应用SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X_reshaped, y)
        
        # 重塑回原始形状
        X_resampled = X_resampled.reshape(-1, n_channels, n_timesteps)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_resampled)
        y_tensor = torch.LongTensor(y_resampled)
        
        # 创建新的数据加载器
        resampled_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        resampled_loader = torch.utils.data.DataLoader(
            resampled_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 记录类别分布
        unique, counts = np.unique(y_resampled, return_counts=True)
        distribution = dict(zip(unique, counts))
        logger.info(f"SMOTE后的类别分布: {distribution}")
        
        return resampled_loader
    
    def train(self, train_loader, val_loader):
        """重写训练方法，加入SMOTE处理"""
        logger.info("开始SMOTE过采样...")
        resampled_train_loader = self._apply_smote(train_loader)
        logger.info("SMOTE过采样完成，开始训练...")
        
        # 调用父类的训练方法
        history = super().train(resampled_train_loader, val_loader)
        
        # 清理内存
        del resampled_train_loader
        torch.cuda.empty_cache()
        
        return history
    
    def plot_confusion_matrix(self, cm, epoch, is_best=False):
        """重写混淆矩阵绘制方法，添加SMOTE标识"""
        plt.figure(figsize=(10, 8))
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=stage_names, yticklabels=stage_names)
        plt.title(f'{self.config.model.name.upper()} Model (SMOTE) - Confusion Matrix (Epoch {epoch+1})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        filename = f'{self.config.model.name}_smote_best_confusion_matrix.png' if is_best else f'{self.config.model.name}_smote_confusion_matrix_epoch_{epoch+1}.png'
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices', filename))
        plt.close()
    
    def plot_training_curves(self):
        """重写训练曲线绘制方法，添加SMOTE标识"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title(f'{self.config.model.name.upper()} Model (SMOTE) - Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title(f'{self.config.model.name.upper()} Model (SMOTE) - Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_curves', f'{self.config.model.name}_smote_training_curves.png'))
        plt.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """重写检查点保存方法，添加SMOTE标识"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        checkpoint_path = os.path.join(self.output_dir, 'model_checkpoints', f'{self.config.model.name}_smote_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'model_checkpoints', f'{self.config.model.name}_smote_best_model.pth')
            torch.save(checkpoint, best_path)
    
    def save_history(self):
        """重写历史记录保存方法，添加SMOTE标识"""
        history_path = os.path.join(self.output_dir, f'{self.config.model.name}_smote_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4) 