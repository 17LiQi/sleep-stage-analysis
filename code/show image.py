import mne
import matplotlib.pyplot as plt
import os


def plot_edf_channels(file_path):
    """
    读取单个 EDF 文件，并为每个通道单独生成一张波形图。
    参数:
        file_path: str, EDF 文件的路径
    """
    # 读取 EDF 文件
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # 获取数据和通道信息
    data, times = raw.get_data(return_times=True)
    ch_names = raw.ch_names

    # 创建保存图片的文件夹
    # save_dir = os.path.dirname(file_path)

    # 遍历每个通道，单独绘图
    for i, ch in enumerate(ch_names):
        plt.figure(figsize=(12, 4))  # 每个通道单独一个图
        plt.plot(times, data[i, :], label=f"{ch}", color="b")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Channel: {ch}")
        plt.legend(loc="upper right")
        plt.grid(True)

        plt.show()

edf_file_path = r"../PSG/ST7011J0-PSG.edf"

# 执行绘图
plot_edf_channels(edf_file_path)
