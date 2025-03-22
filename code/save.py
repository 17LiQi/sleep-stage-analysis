import os
import torch
import numpy as np
import mne


def preprocess_and_segment_EEG(file_path, save_path, window_sec=30):
    # 读取 EEG 数据
    raw_data = mne.io.read_raw_edf(file_path, preload=True)

    # 选择指定通道(EEG Fpz-Cz)
    selected_channel = 'EEG Fpz-Cz'
    raw_data.pick_channels([selected_channel])

    """简单预处理"""
    raw_filtered = raw_data.copy()  # 复制数据进行滤波
    raw_filtered.notch_filter(freqs=49)  # 应用 49Hz 陷波滤波器
    raw_filtered.filter(l_freq=0.5, h_freq=40)  # 应用 0.5-40Hz 带通滤波
    # 获取预处理后的数据
    filtered_data, times = raw_filtered[:]

    """*********"""

    sfreq = int(raw_filtered.info['sfreq'])
    window_samples = sfreq * window_sec
    filtered_data, _ = raw_filtered[:, :]

    # 数据分割
    n_samples = filtered_data.shape[1]
    n_windows = n_samples // window_samples
    truncated_samples = n_windows * window_samples

    # 转换为PyTorch Tensor
    data = raw_filtered.get_data()[0, :n_windows * window_samples]  # 获取单通道数据
    tensor_data = torch.FloatTensor(data.reshape(n_windows, window_samples))

    # 保存（Tensor + 元数据）
    torch.save({
        'eeg_data': tensor_data.unsqueeze(1),  # 添加通道维度 (n, 1, 3000)
        'metadata': {
            'sfreq': sfreq,
            'window_sec': window_sec,
            'channels': ['EEG Fpz-Cz']
        }
    }, save_path)

    print(f"Saved {n_windows} segments to {save_path}")


def process_files(input_dir, output_dir, window_sec=30):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.edf'):
            input_path = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0] + '.pt'
            output_path = os.path.join(output_dir, output_name)

            try:
                preprocess_and_segment_EEG(input_path, output_path, window_sec)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


input_directory = '../PSG'
output_directory = '../processed_eeg_data'
process_files(input_directory, output_directory)
