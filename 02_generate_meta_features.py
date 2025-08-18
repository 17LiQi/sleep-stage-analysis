# 02_generate_meta_features.py

import os
import torch
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm

from src.losses.focal_loss import FocalLoss
# (本地模块导入)
from src.utils.path_manager import get_path_manager
from src.utils.config import load_config
from src.dataset.dataset import SleepEDFDataset
from src.models import get_model
from src.trainer.trainer import Trainer

# ===================================================================
# >> 在这里配置要用于生成元特征的基础模型 <<
# ===================================================================
BASE_MODELS_CONFIGS = [
    "models/Attn_ResNet",
    "models/Attn_CNN_GRU",
    "models/se_resnet"
]


# ===================================================================

def generate_meta_features():
    print("--- 步骤2: 开始为Stacking生成元特征 ---")
    path = get_path_manager()
    base_config = load_config("base", path.CONFIG_ROOT)
    device = torch.device(base_config['device'])

    # 1. --- 加载完整的开发集数据 ---
    dev_data_path = path.DATA_ROOT / "development" / base_config['data']['channel']
    dev_files = list(dev_data_path.glob("*.npz"))
    dev_dataset = SleepEDFDataset(dev_files)

    # 2. --- 初始化用于存储OOF预测的数组 ---
    oof_preds = {
        config_name: np.zeros((len(dev_dataset), 5))
        for config_name in BASE_MODELS_CONFIGS
    }

    # 3. --- 交叉验证循环 ---
    skf = StratifiedKFold(n_splits=base_config['data']['n_splits_cv'], shuffle=True, random_state=base_config['seed'])

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(dev_dataset)), dev_dataset.labels)
    ):
        print(f"\n{'=' * 20} 元特征生成: FOLD {fold + 1}/{base_config['data']['n_splits_cv']} {'=' * 20}")

        val_subset = Subset(dev_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)  # 验证loader可以提前创建

        # 4. --- 在当前fold的训练集上，为每个基础模型训练一个临时模型 ---
        for config_name in BASE_MODELS_CONFIGS:
            print(f"\n--- 正在为 FOLD {fold + 1} 训练临时模型: {config_name} ---")
            config = load_config(config_name, path.CONFIG_ROOT)

            # --- 【核心修复】使用临时的、fold特定的输出目录 ---
            temp_output_dir = path.OUTPUT_ROOT / "temp_meta_feature_models" / f"fold_{fold + 1}" / config[
                'experiment_id']
            temp_output_dir.mkdir(parents=True, exist_ok=True)

            train_subset = Subset(dev_dataset, train_idx)
            train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)

            val_subset = Subset(dev_dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)

            # 3. --- 初始化所有组件 ---
            device = torch.device(config['device'])
            model = get_model(config['model']['name'], **config['model']['params'])

            # A. 加载预训练权重 (如果配置了)
            pretrain_config = config.get('pretrain')
            if pretrain_config and pretrain_config.get('encoder_path'):
                full_encoder_path = path.PROJECT_ROOT / pretrain_config['encoder_path']
                if full_encoder_path.exists():
                    if hasattr(model, 'load_encoder'):
                        model.load_encoder(str(full_encoder_path))
                        print(f"成功加载预训练权重。")
                else:
                    print(f"警告: 未找到预训练权重文件: {full_encoder_path}")

            model.to(device)

            # B. 设置优化器 (支持差分学习率)
            optimizer_config = config['training']['optimizer']
            OptimizerClass = getattr(torch.optim, optimizer_config.get('name', 'AdamW'))

            # 检查是否需要并可以使用差分学习率
            use_diff_lr = (
                    pretrain_config
                    and not pretrain_config.get('freeze_encoder', True)
                    and 'encoder_lr' in optimizer_config
                    and hasattr(model, 'cnn_encoder')
            )

            if use_diff_lr:
                print("使用差分学习率优化器。")
                encoder_params = model.cnn_encoder.parameters()
                other_params = [p for n, p in model.named_parameters() if not n.startswith('cnn_encoder.')]
                param_groups = [
                    {'params': encoder_params, 'lr': optimizer_config['encoder_lr']},
                    {'params': other_params, 'lr': optimizer_config['params']['lr']}
                ]
                optimizer = OptimizerClass(param_groups, **optimizer_config.get('params', {}))
            else:
                print("使用标准优化器。")
                optimizer = OptimizerClass(model.parameters(), **optimizer_config.get('params', {}))

            # C. 设置损失函数 (带类别权重)
            # 在训练子集上计算权重，以避免数据泄露
            train_labels = dev_dataset.labels[train_idx]
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

            loss_config = config['training'].get('loss', {})
            if loss_config.get('name') == 'FocalLoss':
                print(f"使用 FocalLoss (gamma={loss_config.get('params', {}).get('gamma', 2.0)})。")
                criterion = FocalLoss(alpha=class_weights, **loss_config.get('params', {}))
            else:
                print("\n使用 CrossEntropyLoss (带类别权重)。")
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

            # D. 设置学习率调度器 (简化版)
            scheduler_config = config['training'].get('scheduler')
            scheduler = None
            if scheduler_config:
                SchedulerClass = getattr(torch.optim.lr_scheduler, scheduler_config['name'])
                scheduler = SchedulerClass(optimizer, **scheduler_config.get('params', {}))
                print(f"使用学习率调度器: {scheduler_config['name']}")

            model_output_dir = path.OUTPUT_ROOT / "final_single_models" / config['experiment_id']
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # 4. --- 调用Trainer进行训练 ---
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                device=device, config=config, output_dir=str(temp_output_dir)
            )
            trainer.train()

            print(f"最终模型已保存至: {model_output_dir}")


            # 5. --- 使用训练好的临时模型对验证集进行预测 ---
            print(f"--- 正在为 FOLD {fold + 1} 的验证集生成预测: {config_name} ---")
            best_model_path = temp_output_dir / "best_model.pth"
            if not best_model_path.exists():
                print(f"警告: 在 {temp_output_dir} 中未找到模型，跳过此模型的该折预测。")
                continue

            model = get_model(config['model']['name'], **config['model']['params']).to(device)
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            model.eval()

            val_probs = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    val_probs.append(probs.cpu().numpy())

            val_probs = np.concatenate(val_probs)

            # 将预测概率填充到oof_preds数组的对应位置
            oof_preds[config_name][val_idx] = val_probs

    # 6. --- 保存元特征 ---
    meta_features_dir = path.OUTPUT_ROOT / "meta_features"
    meta_features_dir.mkdir(parents=True, exist_ok=True)

    # 将所有模型的oof预测拼接起来
    meta_features = np.concatenate([oof_preds[name] for name in BASE_MODELS_CONFIGS], axis=1)

    # 保存元特征和对应的真实标签
    np.save(meta_features_dir / "meta_features.npy", meta_features)
    np.save(meta_features_dir / "meta_labels.npy", dev_dataset.labels)

    print("\n元特征生成完毕")
    print(f"Meta Features Shape: {meta_features.shape}")
    print(f"已保存至: {meta_features_dir}")


if __name__ == "__main__":
    generate_meta_features()