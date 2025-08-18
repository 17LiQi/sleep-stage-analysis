# 01_train_base_models.py

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import shutil

# 本地模块导入
from src.utils.path_manager import get_path_manager
from src.utils.config import load_config
from src.dataset.dataset import SleepEDFDataset
from src.models import get_model
from src.losses.focal_loss import FocalLoss
from src.trainer.trainer import Trainer
from sklearn.utils.class_weight import compute_class_weight

# ===================================================================
# >> 在这里配置您要训练的所有最终单模型 <<
# ===================================================================
MODELS_TO_TRAIN = [
    # "models/Attn_CNN_GRU",
    # "models/Attn_ResNet",
    # "models/Attn_SimpleCNN",
    # "models/CNN_GRU",
    # "models/ResNet",
    # "models/SimpleCNN"
    "models/se_resnet"
]


# ===================================================================

def train_final_model(config_name: str):
    """
    在一个配置文件指导下，使用整个开发集训练一个最终的单模型。
    """
    path = get_path_manager()
    config = load_config(config_name, path.CONFIG_ROOT)

    model_output_dir = path.OUTPUT_ROOT / "final_single_models" / config['experiment_id']
    model_output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_output_dir / "best_model.pth"

    if best_model_path.exists():
        print(f"最终模型 {best_model_path} 已存在，跳过训练。")
        return

    print(f"\n{'=' * 25}\n 正在训练最终模型: {config['experiment_id']} \n{'=' * 25}")

    # 1. --- 加载整个开发集数据 ---
    dev_data_path = path.DATA_ROOT / "development" / config['data']['channel']
    dev_files = list(dev_data_path.glob("*.npz"))
    dev_dataset = SleepEDFDataset(dev_files)

    # 2. --- 内部划分训练/验证集 (用于早停) ---
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['seed'])
    train_idx, val_idx = next(sgkf.split(X=np.zeros(len(dev_dataset)), y=dev_dataset.labels, groups=dev_dataset.groups))

    train_subset = Subset(dev_dataset, train_idx)
    val_subset = Subset(dev_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    print(f"开发集内部划分为: {len(train_subset)} 训练样本, {len(val_subset)} 验证样本。")

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

    # 4. --- 调用Trainer进行训练 ---
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, config=config, output_dir=str(model_output_dir)
    )
    trainer.train()

    # 5. --- 保存配置文件以供追溯 ---
    # config_source_path = path.CONFIG_ROOT / f"{config_name}.yaml"
    # shutil.copy(config_source_path, model_output_dir / "config.yaml")
    # print(f"最终模型和配置已保存至: {model_output_dir}")
    print(f"最终模型已保存至: {model_output_dir}")


def main():
    for config_name in MODELS_TO_TRAIN:
        try:
            train_final_model(config_name)
        except Exception as e:
            print(f"\n训练模型 {config_name} 失败")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()