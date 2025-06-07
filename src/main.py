import os
import torch
import logging
from torch.utils.data import DataLoader, random_split
from utils.config import Config
from models.sleep_net import SleepNet
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from data.dataset import SleepDataset, SleepDataLoader
from training.trainer import Trainer
import json

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
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

def main():
    # 要训练的模型列表
    model_names = ['cnn', 'lstm', 'sleep_net']
    
    for model_name in model_names:
        logger.info(f"开始训练模型: {model_name}")
        
        # 加载配置
        config = Config.get_default_config(model_name)
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # 创建数据加载器
        data_loader = SleepDataLoader(config)
        
        # 加载所有受试者数据
        all_segments = []
        all_labels = []
        
        # 受试者ID列表
        subject_ids = [
            '4001', '4002', '4011', '4012', '4021', '4022', '4031', '4032',
            '4041', '4042', '4051', '4052', '4061', '4062', '4071', '4072',
            '4081', '4082', '4091', '4092'
        ]
        
        for subject_id in subject_ids:
            try:
                segments, labels = data_loader.load_subject_data(subject_id)
                all_segments.append(segments)
                all_labels.append(labels)
                logger.info(f"成功加载受试者 SC{subject_id} 的数据")
            except Exception as e:
                logger.error(f"加载受试者 SC{subject_id} 数据时出错: {str(e)}")
        
        if not all_segments:
            logger.error("没有成功加载任何受试者数据，程序退出")
            return
        
        # 合并所有数据
        all_segments = torch.cat(all_segments, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 创建数据集
        dataset = SleepDataset(all_segments, all_labels, augment=True)
        
        # 划分数据集
        total_size = len(dataset)
        val_size = int(total_size * config.training.validation_split)
        test_size = int(total_size * config.training.test_split)
        train_size = total_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 创建模型
        model = get_model(model_name, config)
        
        # 创建训练器
        trainer = Trainer(model, config)
        
        # 训练模型
        history = trainer.train(train_loader, val_loader)
        
        # 在测试集上评估
        test_loss, test_acc, test_cm = trainer.evaluate(test_loader)
        logger.info(f"测试集结果 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        # 保存最终结果
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_val_acc': history['best_val_acc']
        }
        
        results_path = os.path.join(trainer.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"模型 {model_name} 训练完成")

if __name__ == '__main__':
    main() 