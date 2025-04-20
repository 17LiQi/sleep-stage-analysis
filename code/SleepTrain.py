import os
import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import umap
# from torch._dynamo.logging import pbar
from tqdm import tqdm

from FocalLoss import FocalLoss


class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.setup_training()

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_kappas = []
        self.best_epoch = 0
        self.class_distribution = self._analyze_class_distribution()
        self.setup_training()
        self.logged_samples = False

        self.batch_times = []
        self.epoch_times = []
        self.eval_times = []

    def _analyze_class_distribution(self):
        """分析各数据集的类别分布"""

        def get_dist(loader):
            labels = []
            for _, lbls in loader:
                labels.extend(lbls.numpy())
            return np.bincount(labels, minlength=5)

        return {
            'train': get_dist(self.train_loader),
            'val': get_dist(self.val_loader),
            'test': get_dist(self.test_loader)
        }

    def setup_training(self):
        # 增强类别权重计算
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.numpy())

        # 确保所有类别都有表示
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"训练集类别分布: {dict(zip(unique, counts))}")

        # 添加平滑处理防止无穷大权重
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(5),  # 强制包含所有类别
            y=all_labels
        )
        class_weights = np.clip(class_weights, 0.1, 10)  # 限制权重范围
        print(f"类别权重: {class_weights}")

        # 使用Focal Loss
        self.criterion = FocalLoss(
            weight=torch.FloatTensor(class_weights).to(self.device),
            gamma=2.0
        )

        # 优化器添加动量
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5
        )

        self.scaler = torch.amp.GradScaler('cuda')

        def extract_features(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.features(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            return x  # Return LSTM output before classifier

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_times = []

        with tqdm(loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}[Train]", unit="batch",leave=False)as pbar:
            for batch_idx, (eeg, labels) in enumerate(pbar):
                start_time = time.time()

                eeg, labels = eeg.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                # 使用混合精度训练
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(eeg)
                    loss = self.criterion(outputs, labels)
                    if epoch == 0:
                        print(f"output dtype: {outputs.dtype}, labels dtype: {labels.dtype}")

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if epoch == 0 and not self.logged_sample:  # 首次epoch打印输入示例
                    pbar.set_postfix({'status': 'logging data sample'})
                    self._log_data_sample((eeg, labels))
                    self.logged_sample = True

                batch_time = time.time() - start_time
                batch_times.append(batch_time)

                pbar.set_postfix({
                    'loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'acc': f"{correct / total:.4f}",
                    'batch_time': f"{batch_time:.3f}s"
                })

        if epoch == 0:
            print(f"First batch - Output dtype: {outputs.dtype}, Loss dtype: {loss.dtype}")

        self.batch_times.extend(batch_times)
        return total_loss / len(loader), correct / total

    def evaluate(self, loader, dataset_name="Validation"):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        start_time = time.time()

        with tqdm(loader, desc=f"{dataset_name}", unit="batch") as pbar:
            with torch.no_grad():
                for eeg, labels in loader:
                    eeg, labels = eeg.to(self.device), labels.to(self.device)
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(eeg)
                        loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    pbar.set_postfix({'loss': f"{total_loss / (pbar.n + 1):.4f}", 'acc': f"{correct / total:.4f}"})

        avg_loss = total_loss / len(loader)
        kappa = cohen_kappa_score(all_labels, all_preds)

        eval_time = time.time() - start_time
        self.eval_times.append((dataset_name, eval_time))

        print(f"{dataset_name} 评估: ")
        print(classification_report(all_labels, all_preds, 
                                  target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))
        print(f"{dataset_name} Cohen's Kappa: {kappa:.3f}")
        print(f"{dataset_name} 平均损失: {avg_loss:.4f}")
        print(f"{dataset_name} 评估时间: {eval_time:.3f}s")

        self.plot_confusion_matrix(all_labels, all_preds, dataset_name)
        
        return kappa, avg_loss

    def plot_confusion_matrix(self, y_true, y_pred, dataset_name):
        cm = confusion_matrix(y_true, y_pred)


        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='.2f',
            xticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
            yticklabels=['Wake', 'N1', 'N2', 'N3', 'REM']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{dataset_name} Confusion Matrix')
        save_path = os.path.join(self.config.OUTPUTRESULT_PATH, f'cm_{dataset_name.lower()}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"混淆矩阵已保存到: {save_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_kappas, label='Val Kappa')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Training Accuracy & Validation Kappa')
        plt.legend()

        plt.savefig(os.path.join(self.config.OUTPUTRESULT_PATH, 'training_curves.png'))
        plt.close()

    def analyze_class_performance(self):
        """分析各类别表现"""
        print("\n=== 类别分布分析 ===")
        for phase in ['train', 'val', 'test']:
            dist = self.class_distribution[phase]
            print(f"{phase} 类别分布: Wake:{dist[0]} N1:{dist[1]} N2:{dist[2]} N3:{dist[3]} REM:{dist[4]}")

        # 计算样本权重影响
        sample_weights = np.array([self.criterion.weight.cpu().numpy()[l] for l in np.arange(5)])
        print(f"\n样本权重影响: {sample_weights}")

        # 梯度分析（示例）
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"参数 {name} 梯度均值: {param.grad.abs().mean().item():.4e}")



    def _log_data_sample(self, batch):
        """记录输入数据样本"""
        eeg, labels = batch
        eeg, labels = eeg.to(self.device), labels.to(self.device)
        sample = eeg[0].cpu().numpy()
        print(f"\n输入数据验证:")
        print(f"形状: {sample.shape}")
        print(f"数值范围: {sample.min():.3f} ~ {sample.max():.3f}")
        print(f"均值: {sample.mean():.3f} ± {sample.std():.3f}")
        # plt.plot(sample.squeeze())
        # plt.title(f"示例波形 (标签: {labels[0].item()})")
        # plt.savefig(os.path.join(self.config.PROCESSED_EEG_PATH, 'sample_waveform.png'))
        # plt.close()

    # 在训练后添加可视化
    def visualize_features(self, loader):
        self.model.eval()
        features, labels = [], []
        with tqdm(loader, desc="Visualizing Features", unit="batch") as pbar:
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.amp.autocast(device_type='cuda'):
                        feats = self.model.extract_features(x).mean(dim=1)  # 全局平均
                    features.append(feats.cpu())
                    labels.append(y.cpu())

        embeddings = umap.UMAP().fit_transform(torch.cat(features))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=torch.cat(labels), alpha=0.6)
        plt.savefig(os.path.join(self.config.OUTPUTRESULT_PATH, 'feature_embedding.png'))

    def train(self):
        best_kappa = 0
        patience = 0

        with tqdm(range(self.config.NUM_EPOCHS), desc="Training", unit="epoch") as epoch_bar:
            for epoch in epoch_bar:
                start_time = time.time()

                train_loss, train_acc = self.train_epoch(self.train_loader, epoch)
                val_kappa, val_loss = self.evaluate(self.val_loader, "Validation")

                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_kappas.append(val_kappa)
                self.scheduler.step(val_kappa)

                if val_kappa > best_kappa:
                    best_kappa = val_kappa
                    self.best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), '../output_results/best_model.pth')
                    patience = 0
                else:
                    patience += 1
                    if patience > 10:
                        print("Early stopping!")
                        break

                epoch_time = time.time() - start_time
                self.epoch_times.append(epoch_time)

                print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Kappa: {val_kappa:.4f}, Val Loss: {val_loss:.4f}')
                print(f'Epoch 耗时: {epoch_time:.3f}秒')
                epoch_bar.set_postfix({'val_kappa': f"{val_kappa:.3f}", 'epoch_time': f"{epoch_time:.3f}s"})

        self.plot_training_curves()
        self.analyze_class_performance()

        test_kappa, test_loss = self.evaluate(self.test_loader, "Test")
        print(f"\n最终测试集结果: Kappa {test_kappa:.3f}, Loss {test_loss:.4f}")

        self.visualize_features(self.test_loader)
