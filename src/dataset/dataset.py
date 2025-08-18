# src/dataset/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import ntpath
from typing import List, Callable, Optional


class SleepEDFDataset(Dataset):
    def __init__(
            self,
            npz_files: List[str],
            sequence_length: int = 1,
            transforms: Optional[Callable] = None,
            mode: str = 'finetune'
    ):
        """
        Args:
            npz_files (List[str]): NPZ文件路径列表。
            sequence_length (int): 每个样本返回的连续epoch数量。
                                   1 表示单epoch模式。
            transforms (Optional[Callable]): 一个函数/变换，用于在线数据增强。
            mode (str): 'finetune' (返回数据+标签) 或 'predict' (只返回数据)。
        """
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.mode = mode

        # --- 1. 预加载元数据，但不加载庞大的信号数据 ---
        print(f"正在从 {len(npz_files)} 个文件中预加载元数据...")
        self.file_map = {}
        self.epoch_metadata = []  # 存储 (文件路径, epoch在该文件中的索引, 标签, 受试者ID)

        for file_path in npz_files:
            try:
                # 使用 np.load 的内存映射模式，只读取元数据而不加载整个数组
                with np.load(file_path, mmap_mode='r') as data:
                    num_epochs = len(data['y'])
                    subject_id = ntpath.basename(file_path).replace(".npz", "")

                    self.file_map[file_path] = data['x_1d']  # 保持对内存映射数组的引用

                    for i in range(num_epochs):
                        self.epoch_metadata.append(
                            (file_path, i, data['y'][i], subject_id)
                        )
            except Exception as e:
                print(f"警告: 无法加载或处理文件 {file_path}, 跳过. 错误: {e}")

        if not self.epoch_metadata:
            raise ValueError("未能从任何文件中加载有效数据。")

        # 提取标签和分组信息，用于交叉验证
        self.labels = np.array([meta[2] for meta in self.epoch_metadata])
        self.groups = np.array([meta[3] for meta in self.epoch_metadata])

        print("元数据加载完成。")

    def __len__(self):
        return len(self.epoch_metadata)

    def __getitem__(self, idx):
        """
        获取单个样本。根据 sequence_length 返回单epoch或序列。
        """
        if self.sequence_length <= 1:
            # --- 单epoch模式 ---
            file_path, epoch_idx, label, _ = self.epoch_metadata[idx]

            # 按需从内存映射的数组中读取数据
            segment = self.file_map[file_path][epoch_idx].copy()  # .copy()确保数据被加载到内存

            # 应用数据增强
            if self.transforms:
                segment = self.transforms(segment)

            output_tensor = torch.from_numpy(segment).float().unsqueeze(0)

            if self.mode == 'finetune':
                return output_tensor, torch.tensor(label, dtype=torch.long)
            else:
                return output_tensor

        else:
            # --- 序列模式 ---
            # 检查当前索引是否是一个有效的序列起点
            # (即序列不跨越受试者边界)
            target_subject = self.epoch_metadata[idx][3]
            end_idx = idx + self.sequence_length

            # 如果序列超出总长度，或者序列的最后一个epoch属于不同受试者，则为无效样本
            if end_idx > len(self.epoch_metadata) or self.epoch_metadata[end_idx - 1][3] != target_subject:
                return None  # 返回None，由collate_fn处理

            # 提取序列
            sequence_metadata = self.epoch_metadata[idx:end_idx]

            # 按需加载序列中每个epoch的数据
            sequence_data = np.array([
                self.file_map[meta[0]][meta[1]] for meta in sequence_metadata
            ])

            # 序列的标签
            sequence_labels = np.array([meta[2] for meta in sequence_metadata])

            output_tensor = torch.from_numpy(sequence_data).float().unsqueeze(1)
            label_tensor = torch.from_numpy(sequence_labels).long()

            if self.mode == 'finetune':
                return output_tensor, label_tensor
            else:
                return output_tensor


def seq_context_collate_fn(batch):
    """
    自定义的collate_fn，用于过滤掉序列模式下因边界问题而产生的无效(None)样本。
    """
    # 过滤掉返回None的项
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        # 如果整个批次都无效，返回空的张量
        return torch.tensor([]), torch.tensor([])

    # 使用默认的collate函数来处理过滤后的有效样本
    return torch.utils.data.dataloader.default_collate(batch)