import numpy as np
import mne
import matplotlib.pyplot as plt

# 读取 EEG 数据
raw_data = mne.io.read_raw_edf('../PSG/ST7011J0-PSG.edf', preload=True)

# 选择指定通道(EEG Fpz-Cz,脑电从Fpz(鼻根处)到Cz(头顶)的信号)
selected_channel = 'EEG Fpz-Cz'
raw_data.pick_channels([selected_channel])  # 仅保留目标通道

# 获取 EEG 数据值(用于绘制)
data, _ = raw_data[:]
min_val, max_val = np.min(data), np.max(data)
amplitude_range = max_val - min_val  # 计算最大-最小幅度范围

# 设定缩放系数 通过实际显示效果,由于原始数据范围过大,设置为最大-最小幅度的2倍
scaling_factor = amplitude_range / 2

# 打印缩放信息
print(f"EEG 数据范围: [{min_val:.6f}, {max_val:.6f}], 幅度范围: {amplitude_range:.6f}")
print(f"缩放系数值: {scaling_factor:.6f}")

# 绘制原始 EEG 数据
raw_data.plot(
    duration=raw_data.times[-1],  # 设置绘图时长为整个数据的持续时间
    n_channels=1,
    scalings=dict(eeg=scaling_factor),  # 使用动态计算的缩放系数值
    title=f'Raw EEG Data - {selected_channel}'
)

# 复制数据进行滤波
raw_filtered = raw_data.copy()

# 应用 49Hz 陷波滤波器
raw_filtered.notch_filter(freqs=49)

# 应用 0.5-40Hz 带通滤波
raw_filtered.filter(l_freq=0.5, h_freq=40)

# 重新计算滤波后数据的缩放系数
filtered_data, _ = raw_filtered[:]
min_val_f, max_val_f = np.min(filtered_data), np.max(filtered_data)
amplitude_range_f = max_val_f - min_val_f

# 经过滤波后的数据范围缩小,因此可以设置为 最大-最小幅度的3倍
scaling_factor_f = amplitude_range_f / 3

# 绘制滤波后的 EEG 数据
raw_filtered.plot(
    duration=raw_filtered.times[-1],
    n_channels=1,
    scalings=dict(eeg=scaling_factor_f),  # 使用动态计算的缩放系数值
    title=f'Filtered EEG Data - {selected_channel}'
)

# 频谱分析
psd_raw = raw_data.compute_psd(method='welch')
psd_raw.plot()
plt.title(f'Raw PSD - {selected_channel}')

psd_filtered = raw_filtered.compute_psd(method='welch')
psd_filtered.plot()
plt.title(f'Filtered PSD - {selected_channel}')

# 计算数据差异
data_difference = np.sum(np.abs(data - filtered_data))
print(f"数据差异总和: {data_difference:.6f}")
