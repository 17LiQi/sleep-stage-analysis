# src/trainer/trainer.py

import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score


class Trainer:
    """
    一个通用的、封装了完整训练流程的训练器类。
    """

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, config, output_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.output_dir = output_dir

        # 从配置中读取训练参数
        self.epochs = config['training']['epochs']
        self.es_config = config['training'].get('early_stopping')

        # 初始化早停变量
        if self.es_config:
            self.patience = self.es_config.get('patience', 10)
            self.min_delta = self.es_config.get('min_delta', 0)
            self.monitor = self.es_config.get('monitor', 'val_loss')
            self.mode = self.es_config.get('mode', 'min')
            self.patience_counter = 0
            self.best_metric = float('inf') if self.mode == 'min' else -float('inf')
            print(f"早停机制已启用: 监控 '{self.monitor}', 耐心值={self.patience}")
        else:
            self.patience = float('inf')  # 永不早停

    def train(self):
        """
        执行完整的训练循环，包括早停和保存最佳模型。
        """
        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            # 训练和验证
            self._train_epoch()
            val_results = self._validate_epoch()

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 检查早停
            if self.es_config:
                if self._check_early_stopping(val_results):
                    break

        print("\n训练结束。")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Training", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

        if len(all_labels) > 0:
            acc = accuracy_score(all_labels, all_preds)
            kappa = cohen_kappa_score(all_labels, all_preds)
        else:
            acc, kappa = 0.0, 0.0

        print(f"\nValidation Results -> Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, Kappa: {kappa:.4f}")
        return {"val_loss": avg_loss, "val_acc": acc, "val_kappa": kappa}

    def _check_early_stopping(self, val_results):
        current_metric = val_results[self.monitor]
        is_improvement = False

        if self.mode == 'max':
            if current_metric > self.best_metric + self.min_delta:
                is_improvement = True
        else:  # mode == 'min'
            if current_metric < self.best_metric - self.min_delta:
                is_improvement = True

        if is_improvement:
            self.best_metric = current_metric
            self.patience_counter = 0
            best_model_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"监控指标改善: {self.monitor} = {current_metric:.4f}。最佳模型已保存。")
        else:
            self.patience_counter += 1
            print(f"监控指标未改善。早停计数器: {self.patience_counter}/{self.patience}")

        if self.patience_counter >= self.patience:
            print("\n早停机制触发 停止训练。")
            return True
        return False