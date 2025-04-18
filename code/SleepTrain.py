import os

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt


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

    def setup_training(self):
        # 计算类别权重
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.numpy())
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5
        )
        
        # 设置混合精度训练
        self.scaler = torch.amp.GradScaler()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for eeg, labels in loader:
            eeg, labels = eeg.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            # 使用混合精度训练
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(eeg)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(loader), correct / total

    def evaluate(self, loader, dataset_name="Validation"):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for eeg, labels in loader:
                eeg, labels = eeg.to(self.device), labels.to(self.device)
                outputs = self.model(eeg)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        kappa = cohen_kappa_score(all_labels, all_preds)

        print(f"{dataset_name} 评估: ")
        print(classification_report(all_labels, all_preds, 
                                  target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))
        print(f"{dataset_name} Cohen's Kappa: {kappa:.3f}")
        print(f"{dataset_name} 平均损失: {avg_loss:.4f}")

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
        save_path = os.path.join(self.config.PROCESSED_EEG_PATH, f'cm_{dataset_name.lower()}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"混淆矩阵已保存到: {save_path}")

    def train(self):
        best_kappa = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch(self.train_loader)
            val_kappa, val_loss = self.evaluate(self.val_loader, "Validation")
            
            self.scheduler.step(val_kappa)
            
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Kappa: {val_kappa:.4f}, Val Loss: {val_loss:.4f}')

        test_kappa, test_loss = self.evaluate(self.test_loader, "Test")
        print(f"\n最终测试集结果: Kappa {test_kappa:.3f}, Loss {test_loss:.4f}")