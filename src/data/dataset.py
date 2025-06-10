import os
import numpy as np
import torch
import mne
from torch.utils.data import Dataset
import logging
import random
import pandas as pd

"""
不使用滤波处理,验证模型鲁棒性
"""
logger = logging.getLogger(__name__)

def parse_annotation(label_path, window_sec, sampling_rate, stage_mapping):
    """
    解析标签文件，将长时段标签拆分为 30 秒窗口
    Args:
        label_path: 标签文件路径
        window_sec: 窗口大小（秒）
        sampling_rate: 采样率
        stage_mapping: 标签映射字典
    Returns:
        list of tuples: (start_sample, end_sample, label)
    """
    labels = []
    try:
        # 使用mne读取EDF格式的标签文件
        logger.info(f"读取EDF标签文件: {label_path}")
        annotations = mne.read_annotations(label_path)
        
        if not annotations:
            raise ValueError(f"标签文件 {label_path} 为空")
            
        logger.info(f"成功读取 {len(annotations)} 个标注")
        
        # 处理每个标注
        for annot in annotations:
            try:
                onset = float(annot['onset'])
                duration = float(annot['duration'])
                stage = str(annot['description']).strip()
                
                # 获取标签值
                if stage not in stage_mapping:
                    logger.warning(f"未知的标注 {stage}，跳过")
                    continue
                    
                label = stage_mapping[stage]
                if label == -1 or duration <= 0:
                    continue
                    
                # 计算该记录可以切分的完整窗口数
                n_windows = int(duration / window_sec)
                remaining_duration = duration % window_sec
                
                # 切分完整窗口
                for i in range(n_windows):
                    window_start = onset + i * window_sec
                    window_end = window_start + window_sec
                    start_sample = int(window_start * sampling_rate)
                    end_sample = int(window_end * sampling_rate)
                    labels.append((start_sample, end_sample, label))
                
                # 处理剩余时间（如果超过窗口的一半）
                if remaining_duration >= window_sec / 2:
                    window_start = onset + n_windows * window_sec
                    window_end = window_start + window_sec
                    start_sample = int(window_start * sampling_rate)
                    end_sample = int(window_end * sampling_rate)
                    labels.append((start_sample, end_sample, label))
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"处理标注时出错: {e}, 标注数据: {annot}")
                continue
        
        # 按时间排序
        labels.sort(key=lambda x: x[0])
        
        if not labels:
            raise ValueError(f"标签文件 {label_path} 中没有有效的睡眠阶段标签")
            
        logger.info(f"成功解析 {len(labels)} 个标签段")
        return labels
        
    except Exception as e:
        logger.error(f"解析标签文件时出错: {e}")
        raise

def match_labels_to_segments(labels, total_samples, window_sec, sampling_rate):
    """
    将标签分配到 30 秒窗口，返回张量格式
    Args:
        labels: 解析后的标签列表
        total_samples: 总样本数
        window_sec: 窗口大小（秒）
        sampling_rate: 采样率
    Returns:
        numpy.ndarray: 每个窗口的标签
    """
    window_samples = int(window_sec * sampling_rate)
    n_windows = total_samples // window_samples
    seg_labels = np.full(n_windows, -1, dtype=np.int64)
    
    # 为每个窗口分配标签
    for start_sample, end_sample, label in labels:
        start_window = start_sample // window_samples
        end_window = (end_sample - 1) // window_samples + 1
        
        # 确保窗口索引在有效范围内
        start_window = max(0, start_window)
        end_window = min(n_windows, end_window)
        
        # 分配标签
        seg_labels[start_window:end_window] = label
    
    # 检查是否有未标记的窗口
    unlabeled = np.sum(seg_labels == -1)
    if unlabeled > 0:
        logger.warning(f"存在 {unlabeled} 个未标记的窗口")
    
    return seg_labels

class SleepDataset(Dataset):
    def __init__(self, segments: torch.Tensor, labels: torch.Tensor, augment: bool = True):
        self.segments = segments
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        if self.augment and random.random() < 0.5:
            segment = self._augment(segment)
        return segment, label

    def _augment(self, segment: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(segment) * 0.1
        return segment + noise

class SleepDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.data.dataset_path
        self.label_mapping = config.data.stage_mapping  # 添加标签映射属性

    def load_subject_data(self, subject_id: str) -> tuple:
        psg_file = os.path.join(
            self.dataset_path,
            self.config.data.psg_file_pattern.format(subject_id=subject_id)
        )
        logger.info(f"加载PSG文件: {psg_file}")

        hypnogram_types = ['C', 'H', 'J', 'P']
        hypnogram_file = None
        for hyp_type in hypnogram_types:
            temp_hyp_file = os.path.join(
                self.dataset_path,
                self.config.data.hypnogram_file_pattern.format(
                    subject_id=subject_id,
                    type=hyp_type
                )
            )
            if os.path.exists(temp_hyp_file):
                hypnogram_file = temp_hyp_file
                logger.info(f"找到Hypnogram文件: {temp_hyp_file}")
                break

        if not os.path.exists(psg_file):
            logger.error(f"找不到PSG文件: {psg_file}")
            raise FileNotFoundError(f"找不到PSG文件: {psg_file}")
        if not hypnogram_file:
            logger.error(f"找不到Hypnogram文件: {subject_id}")
            raise FileNotFoundError(f"找不到Hypnogram文件: {subject_id}")

        raw = mne.io.read_raw_edf(psg_file, preload=True)
        available_channels = raw.ch_names
        logger.info(f"可用通道: {available_channels}")

        if self.config.data.target_channel not in available_channels:
            logger.error(f"目标通道 {self.config.data.target_channel} 不在PSG文件中！")
            raise ValueError(f"目标通道 {self.config.data.target_channel} 不在PSG文件中！")

        raw = raw.pick_channels([self.config.data.target_channel])
        sfreq = raw.info['sfreq']
        logger.info(f"选用通道: {self.config.data.target_channel}, 采样率: {sfreq} Hz")
        data = raw.get_data()[0]
        total_samples = len(data)

        # 解析标签
        labels = parse_annotation(
            hypnogram_file,
            self.config.data.window_sec,
            sfreq,  # 用实际采样率
            self.config.data.stage_mapping
        )

        # 切分信号
        window_samples = int(self.config.data.window_sec * sfreq)
        n_windows = total_samples // window_samples
        segments = data[:n_windows * window_samples].reshape(n_windows, window_samples)
        # 标准化
        segments = (segments - np.mean(segments, axis=1, keepdims=True)) / (np.std(segments, axis=1, keepdims=True) + 1e-8)
        segments = np.clip(segments, -5, 5)

        # 标签分配
        seg_labels = match_labels_to_segments(
            labels, total_samples, self.config.data.window_sec, sfreq
        )
        mask = seg_labels != -1
        segments = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)[mask]
        seg_labels = torch.tensor(seg_labels, dtype=torch.long)[mask]

        label_counts = np.bincount(seg_labels.numpy(), minlength=5)
        logger.info(f"受试者 {subject_id}: 总窗口数 {n_windows}, 有效窗口数 {mask.sum()}, 标签分布: {label_counts}")

        return segments, seg_labels