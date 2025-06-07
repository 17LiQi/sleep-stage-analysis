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
from sklearn.metrics import confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器和学习率调度器
        self.optimizer = model.get_optimizer()
        self.scheduler = model.get_scheduler(self.optimizer)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'best_val_acc': 0
        }
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(config.output_dir, config.model.name, timestamp)
        for subdir in ['model_checkpoints', 'confusion_matrices', 'loss_curves']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        # 创建所有必要的子目录
        subdirs = [
            'model_checkpoints',
            'confusion_matrices',
            'loss_curves'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
            logger.info(f"创建目录: {os.path.join(self.output_dir, subdir)}")
        
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss/len(train_loader), 100.*correct/total
    
    @torch.no_grad()
    def evaluate(self, val_loader) -> Tuple[float, float, np.ndarray]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        
        return total_loss/len(val_loader), 100.*correct/total, cm
    
    def train(self, train_loader, val_loader):
        """训练模型"""
        logger.info("开始训练...")
        best_val_acc = 0
        patience_counter = 0
        max_epochs = 50
        actual_epochs = 0  # 添加实际训练轮数计数
        
        for epoch in range(max_epochs):
            actual_epochs = epoch + 1  # 更新实际训练轮数
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, cm = self.evaluate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_acc)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 打印进度
            logger.info(
                f'Epoch {actual_epochs}/{max_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.history['best_val_acc'] = best_val_acc
                self.save_checkpoint(actual_epochs - 1, is_best=True)
                patience_counter = 0
                
                # 保存最佳模型的混淆矩阵
                self.plot_confusion_matrix(cm, actual_epochs - 1, is_best=True)
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= 5:
                logger.info(f'Early stopping at epoch {actual_epochs}')
                break
            
            # 定期保存
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                self.plot_confusion_matrix(cm, epoch)
                self.plot_training_curves()
        
        # 保存最终模型（使用实际训练轮数）
        self.save_checkpoint(actual_epochs - 1)
        self.plot_confusion_matrix(cm, actual_epochs - 1)
        self.plot_training_curves()
        
        # 保存训练历史
        self.save_history()
        
        return self.history
    
    def plot_confusion_matrix(self, cm, epoch, is_best=False):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        # 添加标签
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=stage_names, yticklabels=stage_names)
        plt.title(f'Confusion Matrix (Epoch {epoch+1})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # 保存图片
        filename = 'best_confusion_matrix.png' if is_best else f'confusion_matrix_epoch_{epoch+1}.png'
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices', filename))
        plt.close()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_curves', 'training_curves.png'))
        plt.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'model_checkpoints', 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.output_dir, 'model_checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'] 