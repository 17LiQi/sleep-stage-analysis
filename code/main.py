import glob
import os

from SleepConfig import SleepConfig
from SleepDataPreprocessing import EEGProcessor
from SleepDataset import SleepDataset, create_data_loaders_by_file
from SleepNet import SleepNet
from SleepTrain import Trainer


def main(reprocess_data=False):
    try:
        print("=== 开始睡眠数据处理 ===")

        # 加载配置
        config = SleepConfig()

        # 数据预处理（仅在需要时运行）
        processed_dir = config.PROCESSED_EEG_PATH
        if reprocess_data or not os.path.exists(processed_dir) or not glob.glob(os.path.join(processed_dir, '*.pt')):
            print("\n=== 重新处理 EDF 文件 ===")
            processor = EEGProcessor()
            processor.process_all_files(merge_output=False)
        else:
            print("\n=== 使用已保存的 .pt 文件 ===")

        # 获取所有 .pt 文件
        pt_files = glob.glob(os.path.join(processed_dir, '*.pt'))
        if not pt_files:
            raise ValueError(f"目录 {processed_dir} 中没有 .pt 文件")
        print(f"\n=== 找到 {len(pt_files)} 个 .pt 文件 ===")


        # 创建数据集和 DataLoader
        print("\n=== 创建数据集 ===")
        train_loader, val_loader, test_loader = create_data_loaders_by_file(pt_files, batch_size=config.BATCH_SIZE)

        # 模型训练
        print("\n=== 开始模型训练 ===")
        model = SleepNet(**config.MODEL_CONFIG)
        trainer = Trainer(model, config, train_loader, val_loader, test_loader)
        trainer.train()
        
        print("\n=== 训练完成 ===")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()