import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
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
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # 设置混合精度训练
        self.scaler = GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        print(classification_report(all_labels, all_preds, 
                                  target_names=['Wake', 'N1', 'N2', 'N3']))
        print(f"Cohen's Kappa: {cohen_kappa_score(all_labels, all_preds):.3f}")
        
        self.plot_confusion_matrix(all_labels, all_preds)
        
        return cohen_kappa_score(all_labels, all_preds)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='.2f',
                    xticklabels=['Wake', 'N1', 'N2', 'N3'],
                    yticklabels=['Wake', 'N1', 'N2', 'N3'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def train(self, epochs=50):
        best_kappa = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_kappa = self.evaluate()
            
            self.scheduler.step(val_kappa)
            
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Kappa: {val_kappa:.4f}')