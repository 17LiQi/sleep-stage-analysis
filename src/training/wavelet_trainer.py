import json

import torch
import torch.nn as nn
import numpy as np
import pywt
from typing import Dict, Tuple, Optional
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from .trainer import Trainer

logger = logging.getLogger(__name__)


class WaveletTrainer(Trainer):
    def __init__(self, model, config, use_wavelet: bool = True, wavelet_config: Optional[Dict] = None):
        super().__init__(model, config)
        self.use_wavelet = use_wavelet
        self.wavelet_config = wavelet_config or {
            'wavelet': 'db4',
            'levels': 5,
            'focus_n1': True
        }
        self.wavelet = self.wavelet_config['wavelet']
        self.levels = self.wavelet_config['levels']
        self.focus_n1 = self.wavelet_config['focus_n1']

        # 创建小波分析目录
        if self.use_wavelet:
            wavelet_dir = f"{self.config.output_dir}/wavelet_analysis".replace('\\', '/')
            os.makedirs(wavelet_dir, exist_ok=True)
            logger.info(f"创建小波分析目录: {wavelet_dir}")

        # 如果启用 N1 优化，调整损失函数权重
        self.class_weights = None
        if self.focus_n1:
            # N1 阶段（标签为 1）给予更高权重
            weights = torch.ones(self.config.model.num_classes, dtype=torch.float32)
            weights[1] = 2.0  # 提高 N1 阶段权重
            self.class_weights = weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            logger.info(f"启用 N1 优化，类权重: {weights.tolist()}")

    def _apply_wavelet_transform(self, data: torch.Tensor) -> torch.Tensor:
        """对输入数据应用小波变换"""
        batch_size, channels, seq_len = data.shape
        transformed_data = []

        for b in range(batch_size):
            signal = data[b, 0].cpu().numpy()  # 提取单通道 EEG 数据
            # 应用离散小波变换
            coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.levels, mode='symmetric')
            # 重构小波系数，只保留近似系数 (cA) 和细节系数 (cD)
            coeffs_flat = np.concatenate([coeffs[0]] + coeffs[1:], axis=0)
            # 确保输出长度与输入一致（通过填充或截断）
            if len(coeffs_flat) > seq_len:
                coeffs_flat = coeffs_flat[:seq_len]
            elif len(coeffs_flat) < seq_len:
                coeffs_flat = np.pad(coeffs_flat, (0, seq_len - len(coeffs_flat)), mode='constant')
            transformed_data.append(coeffs_flat)

        transformed_data = np.stack(transformed_data)[:, np.newaxis, :]  # 恢复形状 [batch, 1, seq_len]
        return torch.tensor(transformed_data, dtype=torch.float32).to(self.device)

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个 epoch，加入小波变换"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Wavelet Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # 应用小波变换
            if self.use_wavelet:
                data = self._apply_wavelet_transform(data)

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
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        return total_loss / len(train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self, val_loader) -> Tuple[float, float, np.ndarray, float, float]:
        """评估模型，加入小波变换"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)

            # 应用小波变换
            if self.use_wavelet:
                data = self._apply_wavelet_transform(data)

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

        # 计算 F1 分数
        f1 = f1_score(all_targets, all_preds, average='macro')

        # 计算 Kappa 系数
        n_classes = len(np.unique(all_targets))
        observed_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        expected_accuracy = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (np.sum(cm) ** 2)
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

        return total_loss / len(val_loader), 100. * correct / total, cm, f1, kappa

    def plot_wavelet_coeffs(self, data: torch.Tensor, epoch: int, is_best: bool = False):
        """绘制小波变换系数（仅对第一个样本）"""
        if not self.use_wavelet:
            return

        data = data[0, 0].cpu().numpy()  # 取第一个样本
        coeffs = pywt.wavedec(data, wavelet=self.wavelet, level=self.levels, mode='symmetric')

        plt.figure(figsize=(12, 8))
        for i, coeff in enumerate(coeffs):
            plt.subplot(self.levels + 1, 1, i + 1)
            plt.plot(coeff)
            plt.title(f'Wavelet Level {i} Coefficients' if i > 0 else 'Approximation Coefficients')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

        plt.tight_layout()
        filename = f"{self.config.model.name}_best_wavelet_coeffs.png" if is_best else f"{self.config.model.name}_wavelet_coeffs_epoch_{epoch + 1}.png"
        save_path = f"{self.config.output_dir}/wavelet_analysis/{filename}".replace('\\', '/')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"保存小波系数图: {save_path}")

    def train(self, train_loader, val_loader):
        """训练模型，加入小波变换相关功能"""
        logger.info("开始小波训练...")
        best_val_acc = 0
        patience_counter = 0
        max_epochs = self.config.training.num_epochs
        actual_epochs = 0

        for epoch in range(max_epochs):
            actual_epochs = epoch + 1

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc, cm, f1, kappa = self.evaluate(val_loader)

            # 更新学习率
            self.scheduler.step(val_acc)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(f1)
            self.history['val_kappa'].append(kappa)

            # 打印进度
            logger.info(
                f'Epoch {actual_epochs}/{max_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                f'F1: {f1:.4f}, Kappa: {kappa:.4f}'
            )

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.history['best_val_acc'] = best_val_acc
                self.save_checkpoint(actual_epochs - 1, is_best=True)
                self.plot_confusion_matrix(cm, actual_epochs - 1, is_best=True)
                # 绘制最佳模型的小波系数
                if self.use_wavelet:
                    for data, _ in val_loader:
                        self.plot_wavelet_coeffs(data, actual_epochs - 1, is_best=True)
                        break
                patience_counter = 0
            else:
                patience_counter += 1

            # 定期保存
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                self.plot_confusion_matrix(cm, epoch)
                if self.use_wavelet:
                    for data, _ in val_loader:
                        self.plot_wavelet_coeffs(data, epoch)
                        break
                self.plot_training_curves()

            # 早停
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {actual_epochs}")
                break

        # 保存最终模型
        self.save_checkpoint(actual_epochs - 1)
        self.plot_confusion_matrix(cm, actual_epochs - 1)
        self.plot_training_curves()

        # 保存训练历史
        self.save_history()

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点，路径使用正斜杠"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'wavelet_config': self.wavelet_config
        }

        checkpoint_dir = f"{self.config.output_dir}/model_checkpoints".replace('\\', '/')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = f"{checkpoint_dir}/{self.config.model.name}_checkpoint.pth".replace('\\', '/')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = f"{checkpoint_dir}/{self.config.model.name}_best_model.pth".replace('\\', '/')
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")

    def save_history(self):
        """保存训练历史，路径使用正斜杠"""
        history_path = f"{self.config.output_dir}/{self.config.model.name}_history.json".replace('\\', '/')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"保存训练历史: {history_path}")