import os
import torch
import logging
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from utils.config import Config
from models.sleep_net import SleepNet
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from utils.wavelet_transform import WaveletCNN
from data.dataset import SleepDataset, SleepDataLoader
from training.trainer import Trainer
from training.smote_trainer import SMOTETrainer
from training.wavelet_trainer import WaveletTrainer
import json
import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model(model_name: str, config: Config):
    """根据模型名称创建模型"""
    if model_name == 'cnn':
        return CNNModel(config)
    elif model_name == 'lstm':
        return LSTMModel(config)
    elif model_name == 'sleep_net':
        return SleepNet(config)
    elif model_name == 'wavelet_cnn':
        return WaveletCNN(config)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

def get_trainer(model_name: str, model, config: Config, use_smote: bool = False, use_wavelet: bool = False):
    """根据配置创建训练器"""
    if use_wavelet:
        logger.info(f"使用小波训练器训练模型: {model_name}")
        wavelet_config = {
            'wavelet': config.wavelet.wavelet,
            'levels': config.wavelet.levels,
            'focus_n1': config.wavelet.focus_n1
        }
        return WaveletTrainer(model, config, use_wavelet=True, wavelet_config=wavelet_config)
    elif use_smote:
        logger.info(f"使用SMOTE训练器训练模型: {model_name}")
        return SMOTETrainer(model, config)
    else:
        logger.info(f"使用标准训练器训练模型: {model_name}")
        return Trainer(model, config)

def load_all_subjects_data(data_loader: SleepDataLoader, subject_ids: List[str]) -> Tuple[SleepDataset, Dict[str, int]]:
    """加载所有受试者的数据并合并"""
    all_segments = []
    all_labels = []
    
    for subject_id in subject_ids:
        logger.info(f"加载受试者 SC{subject_id} 的数据")
        segments, labels = data_loader.load_subject_data(subject_id)
        all_segments.extend(segments)
        all_labels.extend(labels)
    
    return SleepDataset(all_segments, all_labels, augment=True), data_loader.label_mapping

def main():
    # 要训练的模型列表
    # model_names = ['cnn', 'lstm', 'sleep_net', 'wavelet_cnn']
    model_names = ['cnn']

    # # 受试者ID列表
    # subject_ids = [
    #     '4001', '4002', '4011', '4012', '4021', '4022', '4031', '4032',
    #     '4041', '4042', '4051', '4052', '4061', '4062', '4071', '4072',
    #     '4081', '4082', '4091', '4092'
    # ]
    # 受试者ID列表
    subject_ids = [
        '4001', '4002', '4011', '4012', '4021', '4022', '4031', '4032',
        '4041', '4042', '4051', '4052'
    ]
    
    # 是否使用SMOTE
    # use_smote = False
    use_smote = False
    
    # 是否使用小波变换
    use_wavelet = True

    # 加载配置
    if use_wavelet:
        config = Config.get_wavelet_config(model_names[0], enable_wavelet=True)
    else:
        config = Config.get_default_config(model_names[0])
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建数据加载器
    data_loader = SleepDataLoader(config)
    
    # 加载所有受试者的数据
    logger.info("开始加载所有受试者数据...")
    full_dataset, label_mapping = load_all_subjects_data(data_loader, subject_ids)
    
    # 创建K折交叉验证
    kfold = KFold(n_splits=config.data.n_splits, shuffle=True, random_state=config.seed)
    
    for model_name in model_names:
        logger.info(f"开始训练模型: {model_name}")
        
        # 更新模型配置
        if use_wavelet:
            config = Config.get_wavelet_config(model_name, enable_wavelet=True)
        else:
            config = Config.get_default_config(model_name)
        
        # 创建模型特定的输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_output_dir = os.path.join(config.output_dir, model_name, timestamp)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 更新配置中的输出目录
        config.output_dir = model_output_dir
        
        # 存储所有折的结果
        all_fold_results = []
        
        # 对每一折进行训练和评估
        for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
            logger.info(f"开始第 {fold+1} 折训练")
            
            # 创建数据加载器
            train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # 创建模型
            model = get_model(model_name, config)
            
            # 创建训练器
            trainer = get_trainer(model_name, model, config, use_smote, use_wavelet)
            
            # 训练模型
            history = trainer.train(train_loader, val_loader)
            
            # 记录结果
            fold_result = {
                'fold': fold + 1,
                'best_val_acc': history['best_val_acc'],
                'best_val_f1': max(history['val_f1']),
                'best_val_kappa': max(history['val_kappa'])
            }
            all_fold_results.append(fold_result)
            
            # 清理显存
            del model, trainer
            torch.cuda.empty_cache()
        
        # 计算平均结果
        overall_result = {
            'model_name': model_name,
            'use_smote': use_smote,
            'use_wavelet': use_wavelet,
            'wavelet_config': config.wavelet.__dict__ if use_wavelet else None,
            'mean_val_acc': np.mean([r['best_val_acc'] for r in all_fold_results]),
            'mean_val_f1': np.mean([r['best_val_f1'] for r in all_fold_results]),
            'mean_val_kappa': np.mean([r['best_val_kappa'] for r in all_fold_results]),
            'fold_results': all_fold_results,
            'label_mapping': label_mapping
        }
        
        # 保存结果
        results_path = os.path.join(model_output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(overall_result, f, indent=4)
        
        logger.info(
            f"模型 {model_name} ({'小波+SMOTE' if use_wavelet and use_smote else '小波' if use_wavelet else 'SMOTE' if use_smote else '标准'}) 训练完成 - "
            f"总体平均准确率: {overall_result['mean_val_acc']:.2f}%, "
            f"总体平均F1: {overall_result['mean_val_f1']:.4f}, "
            f"总体平均Kappa: {overall_result['mean_val_kappa']:.4f}"
        )

if __name__ == '__main__':
    main() 