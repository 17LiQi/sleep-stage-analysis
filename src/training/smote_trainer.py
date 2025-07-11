import torch
import torch.nn as nn
import numpy as np
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from .trainer import Trainer

logger = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_pos = (anchor - positive).pow(2).sum(1)
        distance_neg = (anchor - negative).pow(2).sum(1)
        loss = torch.relu(distance_pos - distance_neg + self.margin)
        return loss.mean()


class SMOTETrainer(Trainer):
    def __init__(self, model, config):
        super().__init__(model, config)

        # 初始化 SMOTE
        self.smote = None
        self.k_neighbors = 5  # 增加邻居数

        # 调整类别权重
        self.class_weights = torch.tensor([
            1.0,  # Wake
            4.0,  # N1
            2.0,  # N2
            3.0,  # N3
            4.0   # REM
        ]).to(self.device)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.triplet_criterion = TripletLoss(margin=1.0)
        self.triplet_alpha = 0.3  # 降低三元组损失权重

        # 创建所有必要的输出目录
        self._create_output_directories()
        
        # 设置日志文件
        self.setup_logging()

    def _create_output_directories(self):
        """创建所有必要的输出目录"""
        directories = {
            'log_dir': 'logs',
            'tsne_dir': 'tsne',
            'confusion_matrix_dir': 'confusion_matrices',
            'loss_curves_dir': 'loss_curves',
            'checkpoint_dir': 'model_checkpoints'
        }
        
        for attr_name, dir_name in directories.items():
            dir_path = os.path.join(self.output_dir, dir_name)
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"创建目录: {dir_path}")

    def setup_logging(self):
        """设置详细的日志记录"""
        log_file = os.path.join(self.log_dir, f'{self.config.model.name}_smote_training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def _apply_smote(self, train_loader):
        """应用改进的 SMOTE 过采样策略"""
        # 收集训练数据
        all_data, all_labels = [], []
        for data, labels in train_loader:
            all_data.append(data.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # 记录原始类别分布
        original_distribution = np.bincount(y)
        logger.info(f"原始类别分布: {dict(zip(range(len(original_distribution)), original_distribution))}")

        # 改进的采样策略
        class_counts = np.bincount(y)
        max_count = class_counts.max()
        sampling_strategy = {
            0: max(class_counts[0], int(max_count * 0.8)),  # Wake - 保持原始数量或增加到目标数量
            1: max(class_counts[1], max_count),             # N1 - 完全平衡
            2: max(class_counts[2], int(max_count * 0.9)),  # N2
            3: max(class_counts[3], int(max_count * 0.7)),  # N3
            4: max(class_counts[4], int(max_count * 0.8))   # REM
        }

        # 初始化 SMOTE
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.config.seed
        )

        # 重塑数据
        n_samples, n_channels, n_timesteps = X.shape
        X_reshaped = X.reshape(n_samples, -1)

        # 应用 SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X_reshaped, y)

        # 重塑回原始形状
        X_resampled = X_resampled.reshape(-1, n_channels, n_timesteps)

        # 转换为 PyTorch 张量
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

        # 记录采样后的类别分布
        resampled_distribution = np.bincount(y_resampled)
        logger.info(f"SMOTE 后的类别分布: {dict(zip(range(len(resampled_distribution)), resampled_distribution))}")

        return resampled_loader

    def _construct_triplets(self, features, labels):
        """构造三元组（参考、正、负样本）"""
        triplets = []
        labels_np = labels.cpu().numpy()
        for i, label in enumerate(labels_np):
            pos_indices = np.where(labels_np == label)[0]
            neg_indices = np.where(labels_np != label)[0]
            if len(pos_indices) > 1 and len(neg_indices) > 0:
                pos_idx = np.random.choice(pos_indices)
                neg_idx = np.random.choice(neg_indices)
                if pos_idx != i:  # 确保正样本不是参考样本本身
                    triplets.append((i, pos_idx, neg_idx))
        return triplets

    def train_step(self, batch):
        """改进的训练步骤，添加详细日志"""
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        # 前向传播
        logits = self.model(data)
        features = self.model.extract_features(data)['lstm_features'][:, -1, :]

        # 交叉熵损失
        ce_loss = self.criterion(logits, labels)

        # 三元组损失
        triplets = self._construct_triplets(features, labels)
        triplet_loss = 0
        if triplets:
            for anchor_idx, pos_idx, neg_idx in triplets:
                anchor = features[anchor_idx]
                positive = features[pos_idx]
                negative = features[neg_idx]
                triplet_loss += self.triplet_criterion(anchor, positive, negative)
            triplet_loss /= len(triplets)

        # 总损失
        total_loss = ce_loss + self.triplet_alpha * triplet_loss

        # 记录每个类别的损失
        with torch.no_grad():
            for i in range(self.config.model.num_classes):
                mask = (labels == i)
                if mask.sum() > 0:
                    class_loss = ce_loss[mask].mean()
                    logger.info(f"类别 {i} 的平均损失: {class_loss.item():.4f}")

        return total_loss

    def train(self, train_loader, val_loader):
        """训练流程，加入 SMOTE 和 t-SNE 可视化"""
        logger.info("开始 SMOTE 过采样...")
        resampled_train_loader = self._apply_smote(train_loader)
        logger.info("SMOTE 过采样完成，开始训练...")

        # 调用父类的训练方法
        history = super().train(resampled_train_loader, val_loader)

        # 绘制 t-SNE 可视化（每 5 个 epoch 或最后一个 epoch）
        for epoch in range(0, self.config.training.num_epochs, 5):
            self.plot_tsne(val_loader, epoch)
        self.plot_tsne(val_loader, self.config.training.num_epochs - 1)

        # 清理内存
        del resampled_train_loader
        torch.cuda.empty_cache()

        return history

    def plot_tsne(self, loader, epoch):
        """绘制 t-SNE 可视化"""
        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for data, label in loader:
                data = data.to(self.device)
                feat_dict = self.model.extract_features(data)
                # 自动选择特征
                if 'lstm_features' in feat_dict:
                    feat = feat_dict['lstm_features'][:, -1, :].cpu().numpy()
                elif 'conv_features' in feat_dict:
                    feat = feat_dict['conv_features'].cpu().numpy()
                else:
                    raise ValueError("模型未实现有效的特征提取接口")
                features.append(feat)
                labels.append(label.numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        tsne = TSNE(n_components=2, random_state=self.config.seed)
        embeddings = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        for stage in range(self.config.model.num_classes):
            mask = labels == stage
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=stage_names[stage])
        plt.legend()
        plt.title(f't-SNE Visualization (SMOTE) - Epoch {epoch + 1}')
        
        # 确保目录存在
        os.makedirs(self.tsne_dir, exist_ok=True)
        plt.savefig(os.path.join(self.tsne_dir, f'tsne_epoch_{epoch + 1}.png'))
        plt.close()

    def plot_confusion_matrix(self, cm, epoch, is_best=False):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=stage_names, yticklabels=stage_names)
        plt.title(f'{self.config.model.name.upper()} Model (SMOTE) - Confusion Matrix (Epoch {epoch + 1})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        filename = f'{self.config.model.name}_smote_best_confusion_matrix.png' if is_best else f'{self.config.model.name}_smote_confusion_matrix_epoch_{epoch + 1}.png'
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices', filename))
        plt.close()

    def plot_training_curves(self):
        """绘制训练曲线"""
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
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }

        checkpoint_path = os.path.join(self.output_dir, 'model_checkpoints',
                                       f'{self.config.model.name}_smote_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.output_dir, 'model_checkpoints',
                                     f'{self.config.model.name}_smote_best_model.pth')
            torch.save(checkpoint, best_path)

    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.output_dir, f'{self.config.model.name}_smote_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)