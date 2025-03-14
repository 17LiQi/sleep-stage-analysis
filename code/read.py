import mne
import pyedflib


# 读取 EEG 数据
raw_data = mne.io.read_raw_edf('../PSG/ST7011J0-PSG.edf', preload=True)

# 查看数据基本信息
print(raw_data.info)

# 这里发现mne发出警告:
# RuntimeWarning: Channels contain different highpass filters. Highest filter setting will be stored.
# RuntimeWarning: Channels contain different lowpass filters. Lowest filter setting will be stored.
# 这个警告说明各个通道的高通和低通滤波器不同
# 但用pyedflib查看发现每个通道的高通和低通都是一样的,对于此警告暂时可以忽略

# 打开 EDF 文件
edf_file = pyedflib.EdfReader('../PSG/ST7011J0-PSG.edf')

# 获取文件中的通道数
n_channels = edf_file.signals_in_file

# 遍历每个通道，获取其标签和预处理信息
for ch in range(n_channels):
    ch_label = edf_file.getLabel(ch)
    prefilter = edf_file.getPrefilter(ch)
    print(f"Channel {ch_label}: {prefilter}")

# 但mne.io.read_raw_edf函数将所有高通滤器都设置为0.0 Hz 与实际的高通滤波器设置不同,可能对模型训练有影响,考虑后期手动设置高通滤波器

edf_file.close()
