# ======================= #
# 1. 模块导入与配置加载
# ======================= #
import os
import numpy as np
import mne
import torch

from SleepConfig import SleepConfig
from SleepUtils import parse_annotation, match_labels_to_segments

# ======================= #
# 2. EEGProcessor 类定义
# ======================= #
class EEGProcessor:
    def __init__(self):
        self.sfreq = None

    # ----------------------- #
    # 2.1 查找标签文件
    # ----------------------- #
    def find_label_file(self, edf_file):
        base = os.path.splitext(edf_file)[0]
        for fn in os.listdir(SleepConfig.LABEL_PATH):
            if fn.startswith(base) and fn.endswith('.txt'):
                return os.path.join(SleepConfig.LABEL_PATH, fn)
        return None



    # ----------------------- #
    # 2.3 应用带通和陷波滤波器
    # ----------------------- #
    def _apply_filters(self, raw):
        self.sfreq = raw.info['sfreq']
        nyquist = self.sfreq / 2.0
        if self.sfreq != 100:
            raise ValueError(f"本处理仅支持 100 Hz 数据，当前: {self.sfreq} Hz")

        # 带通滤波: 0.3–40 Hz
        l_freq, h_freq = 0.3, 40.0
        print(f"带通滤波: {l_freq}-{h_freq} Hz (奈奎斯特={nyquist:.1f} Hz)")
        raw.filter(l_freq=l_freq, h_freq=h_freq,
                   method='fir', fir_window='hamming',
                   fir_design='firwin', phase='zero')

        # 陷波滤波: 48 Hz
        notch_freq, notch_width = 48.0, 1.0
        print(f"陷波滤波: {notch_freq} Hz")
        raw.notch_filter(freqs=notch_freq, notch_widths=notch_width,
                         trans_bandwidth=1.0, method='fir',
                         fir_window='hamming', fir_design='firwin',
                         phase='zero')
        return raw

    # ----------------------- #
    # 2.4 处理单个 EDF 文件
    # ----------------------- #
    def process_single_file(self, edf_path, label_path):
        print(f"\n=== 处理文件: {os.path.basename(edf_path)} ===")
        print(f"标签文件: {label_path}")
        try:
            # 读取 EDF 文件
            print(f"读取 EDF 文件: {edf_path}")
            raw = mne.io.read_raw_edf(
                edf_path, preload=True,
                include=[SleepConfig.TARGET_CHANNEL],
                verbose=True
            )
            if SleepConfig.TARGET_CHANNEL not in raw.ch_names:
                raise ValueError(f"通道 {SleepConfig.TARGET_CHANNEL} 不存在于 {edf_path}")
            print(f"通道: {raw.ch_names}")
            print(f"采样率: {raw.info['sfreq']} Hz")
            print(f"记录起始时间: {raw.info['meas_date']}")
            print(f"信号长度: {raw.n_times} 样本 ({raw.n_times / raw.info['sfreq']:.2f} 秒)")

            # 应用滤波
            print("应用滤波...")
            raw = self._apply_filters(raw)

            # 获取数据
            data = raw.get_data()  # 形状: (n_channels=1, n_samples)
            total = data.shape[1]
            print(f"数据形状: {data.shape}, 总样本数: {total}")

            # 解析标签
            print("解析标签文件...")
            labels = parse_annotation(label_path)

            # 分配标签
            print("分配标签到信号...")
            seg_labels = match_labels_to_segments(labels, total, raw.info['meas_date'], sfreq=raw.info['sfreq'])

            # 分段
            seg_len = SleepConfig.WINDOW_SEC * SleepConfig.SAMPLING_RATE
            n_seg = total // seg_len
            print(f"分段: 窗口长度 {seg_len} 样本 ({SleepConfig.WINDOW_SEC} 秒), 段数 {n_seg}")
            segments = data[0, :n_seg * seg_len].reshape(n_seg, seg_len)  # 形状: (n_seg, seg_len)

            print("应用标准化...")
            segments = (segments - np.mean(segments, axis=1, keepdims=True)) / np.std(segments, axis=1, keepdims=True)
            segments = np.clip(segments, -5, 5)  # 防止异常值

            # 添加通道信息验证
            if SleepConfig.TARGET_CHANNEL not in raw.ch_names:
                available = '\n'.join(raw.ch_names)
                raise ValueError(f"目标通道 {SleepConfig.TARGET_CHANNEL} 不存在！可用通道：\n{available}")

            # 添加标签验证
            unique_labels = np.unique(seg_labels.numpy())
            if len(unique_labels) < 2:
                raise ValueError(f"文件 {edf_path} 仅包含单一标签: {unique_labels}")

            # 转换为张量
            segments = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)  # 形状: (n_seg, 1, seg_len)
            print(f"分段后数据形状: {segments.shape}, 标签形状: {seg_labels.shape}")

            # 过滤无效标签
            mask = seg_labels != -1
            segments = segments[mask]
            seg_labels = seg_labels[mask]
            print(f"有效段数: {segments.shape[0]}/{n_seg} ({100 * segments.shape[0] / n_seg:.2f}%)")
            if segments.shape[0] == 0:
                raise ValueError(f"文件 {edf_path} 没有有效的标签段")

            # 保存单独的 .pt 文件
            base_name = os.path.splitext(os.path.basename(edf_path))[0]
            output_path = os.path.join(SleepConfig.PROCESSED_EEG_PATH, f"{base_name}.pt")
            torch.save({
                'eeg': segments,  # 形状: (n_valid_segments, 1, seg_len)
                'labels': seg_labels,  # 形状: (n_valid_segments,)
                'filename': base_name,
                'sampling_rate': raw.info['sfreq'],
                'meas_date': raw.info['meas_date']
            }, output_path)
            print(f"已保存单独文件: {output_path}, 段数: {segments.shape[0]}")

            return segments, seg_labels
        except Exception as e:
            print(f"处理失败: {e}")
            print(f"文件: {edf_path}")
            raise

    # ----------------------- #
    # 2.5 批量处理所有 EDF 文件
    # ----------------------- #
    def process_all_files(self, merge_output=False):
        """
        批量处理所有 EDF 文件
        :param merge_output: 是否合并所有数据到一个 npz 文件
        :return: 如果 merge_output=True，返回合并的数据和标签；否则返回 None
        """
        if not os.path.isdir(SleepConfig.RAW_EEG_PATH):
            raise ValueError(f"数据目录不存在: {SleepConfig.RAW_EEG_PATH}")
        if not os.path.isdir(SleepConfig.LABEL_PATH):
            raise ValueError(f"标签目录不存在: {SleepConfig.LABEL_PATH}")

        os.makedirs(SleepConfig.PROCESSED_EEG_PATH, exist_ok=True)
        all_data, all_labels = [], []
        edf_list = [f for f in os.listdir(SleepConfig.RAW_EEG_PATH) if f.endswith('.edf')]
        print(f"发现 {len(edf_list)} 个 EDF 文件")

        for edf in edf_list:
            edf_full = os.path.join(SleepConfig.RAW_EEG_PATH, edf)
            lbl = self.find_label_file(edf)
            if lbl is None:
                print(f"未找到标签文件: {edf}")
                continue
            try:
                segs, labs = self.process_single_file(edf_full, lbl)
                if merge_output:
                    all_data.append(segs)
                    all_labels.append(labs)
                print(f"成功处理: {edf}, 段数: {segs.shape[0]}, 标签分布: {np.bincount(labs.numpy())}")
            except Exception as e:
                print(f"失败: {edf}, 错误: {e}")
                print(f"跳过文件: {edf}")
                continue

        if merge_output:
            if not all_data:
                raise RuntimeError("没有成功处理任何文件")

            data = torch.cat(all_data, dim=0)  # 形状: (n_total_segments, 1, seg_len)
            labels = torch.cat(all_labels, dim=0)  # 形状: (n_total_segments,)
            print(f"\n合并数据: 总段数 {data.shape[0]}")
            print(f"总标签分布: {np.bincount(labels.numpy())}")

            out = os.path.join(SleepConfig.PROCESSED_EEG_PATH, 'processed_data_100hz.npz')
            np.savez_compressed(out, eeg=data.numpy().astype(np.float32), labels=labels.numpy())
            print(f"已保存合并文件: {out}, 样本数 {data.shape[0]}")

            return data, labels
        else:
            print("所有文件已单独保存为 .pt 文件")
            return None

if __name__ == "__main__":
    processor = EEGProcessor()
    try:
        processor.process_all_files()
    except Exception as e:
        print(f"处理过程中出现错误: {e}")