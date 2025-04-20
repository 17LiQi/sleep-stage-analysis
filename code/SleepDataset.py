import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SleepDataset(Dataset):
    def __init__(self, segments, labels, augment=False):
        self.data = torch.FloatTensor(segments).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment:
            # 数据增强
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        if torch.rand(1) < 0.5:
            # 随机翻转
            x = torch.flip(x, dims=[1])
        x += 0.03 * torch.randn_like(x)

        return x


def create_data_loaders(segments, labels, batch_size=64):
    # 划分训练集和测试集
    # train_data, test_data, train_labels, test_labels = train_test_split(
    #     segments, labels, test_size=0.2, stratify=labels
    # )
    #
    # # 从训练集中划分验证集
    # train_data, val_data, train_labels, val_labels = train_test_split(
    #     train_data, train_labels, test_size=0.1, stratify=train_labels
    # )

    train_data, test_data, train_labels, test_labels = train_test_split(
        segments, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.1, stratify=train_labels, random_state=42
    )

    # 创建数据集
    train_dataset = SleepDataset(train_data, train_labels, augment=True)
    val_dataset = SleepDataset(val_data, val_labels)
    test_dataset = SleepDataset(test_data, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_data_loaders_by_file(pt_files, batch_size=64):
    # 按文件划分
    train_files, test_files = train_test_split(pt_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)
    print(f"训练文件: {len(train_files)}, 验证文件: {len(val_files)}, 测试文件: {len(test_files)}")

    # 加载数据
    def load_data(files):
        segments = []
        labels = []
        for pt_file in files:
            data_dict = torch.load(pt_file)
            eeg = data_dict['eeg']  # (n_segments, 1, 3000)
            lbl = data_dict['labels']  # (n_segments,)
            segments.append(eeg.squeeze(1).numpy())  # (n_segments, 3000)
            labels.append(lbl.numpy())
        return np.concatenate(segments, axis=0), np.concatenate(labels, axis=0)

    train_segments, train_labels = load_data(train_files)
    val_segments, val_labels = load_data(val_files)
    test_segments, test_labels = load_data(test_files)

    # 创建数据集
    train_dataset = SleepDataset(train_segments, train_labels, augment=True)
    val_dataset = SleepDataset(val_segments, val_labels)
    test_dataset = SleepDataset(test_segments, test_labels)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader