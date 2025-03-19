import numpy as np
import mne


def preprocess_and_segment_EEG(file_path, save_path, window_sec=30):
    # 读取 EEG 数据
    raw_data = mne.io.read_raw_edf(file_path, preload=True)

    # 选择指定通道(EEG Fpz-Cz)
    selected_channel = 'EEG Fpz-Cz'
    raw_data.pick_channels([selected_channel])

    # 预处理流程
    raw_filtered = raw_data.copy()

    # 1. 应用49Hz陷波滤波器
    raw_filtered.notch_filter(freqs=49)

    # 2. 应用0.5-40Hz带通滤波
    raw_filtered.filter(l_freq=0.5, h_freq=40)

    # 获取预处理后的数据
    filtered_data, times = raw_filtered[:]

    # 数据分割参数
    sfreq = int(raw_filtered.info['sfreq'])  # 获取采样率
    window_samples = window_sec * sfreq  # 每个窗口的样本数

    # 计算可分割的完整窗口数量
    n_samples = filtered_data.shape[1]
    n_windows = n_samples // window_samples

    # 截断数据至完整窗口数
    truncated_samples = n_windows * window_samples
    truncated_data = filtered_data[:, :truncated_samples]

    # 分割为30秒窗口 (形状：n_windows, 1通道, 时间点)
    segmented_data = truncated_data.reshape(1, n_windows, window_samples).transpose(1, 0, 2)

    # 保存处理后的数据
    np.save(save_path, {
        'data': segmented_data,
        'sfreq': sfreq,
        'window_sec': window_sec,
        'channels': [selected_channel],
        'timestamps': times[:truncated_samples].reshape(n_windows, window_samples)
    })

    print(f"预处理完成，保存了{n_windows}个{window_sec}秒片段")
    return segmented_data


# 使用示例
processed_data = preprocess_and_segment_EEG(
    file_path='../PSG/ST7011J0-PSG.edf',
    save_path='processed_eeg_data.npy'
)