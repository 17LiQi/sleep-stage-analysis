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
    train_data, test_data, train_labels, test_labels = train_test_split(
        segments, labels, test_size=0.2, stratify=labels
    )
    
    # 从训练集中划分验证集
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.1, stratify=train_labels
    )
    
    # 创建数据集
    train_dataset = SleepDataset(train_data, train_labels, augment=True)
    val_dataset = SleepDataset(val_data, val_labels)
    test_dataset = SleepDataset(test_data, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader