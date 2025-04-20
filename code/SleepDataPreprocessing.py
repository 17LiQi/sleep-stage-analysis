# ======================= #
# 1. 模块导入与配置加载
# ======================= #
import os
import numpy as np
import mne
import torch
from datetime import datetime, timedelta

from SleepConfig import SleepConfig
from SleepUtils import parse_annotation, match_labels_to_segments

# ======================= #
# 2. EEGProcessor 类定义
# ======================= #
class EEGProcessor:
    def __init__(self):
        self.sfreq = None
        if not hasattr(SleepConfig, 'LABEL_PATH'):
            SleepConfig.LABEL_PATH = '../tag'
        if not hasattr(SleepConfig, 'OUTPUTRESULT_PATH'):
            SleepConfig.OUTPUTRESULT_PATH = '../output_results'
        os.makedirs(SleepConfig.LABEL_PATH, exist_ok=True)
        os.makedirs(SleepConfig.OUTPUTRESULT_PATH, exist_ok=True)

        self.label_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4,
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def find_label_file(self, edf_file):
        base = os.path.splitext(os.path.basename(edf_file))[0]
        for fn in os.listdir(SleepConfig.LABEL_PATH):
            if fn.startswith(base) and fn.endswith('.txt'):
                return os.path.join(SleepConfig.LABEL_PATH, fn)
        return None

    def process_and_save_labels(self, annotations_file, edf_file, meas_date=None):
        print(f"\n=== 读取标注文件: {os.path.basename(annotations_file)} ===")
        try:
            annotations = mne.read_annotations(annotations_file)
            print("Annotations raw data (before cleaning):")
            for annot in annotations:
                print(annot)

            label_data = []
            for annot in annotations:
                onset = annot['onset']
                duration = annot['duration']
                description = annot['description']
                if '年' in description:
                    print(f"警告: 检测到异常字符 '年' 在描述中: {description}")
                    description = description.replace('年', '').strip()
                    print(f"清理后描述: {description}")
                label_data.append((onset, duration, description))

            processed_labels = []
            for onset, duration, description in label_data:
                if description in self.label_map:
                    processed_labels.append((onset, duration, description))
                else:
                    print(f"警告: 未知的标注 {description}，跳过")
                    continue

            if not processed_labels:
                raise ValueError(f"标注文件 {annotations_file} 中没有有效的睡眠阶段标签")

            # 保存标签到 LABEL_PATH，使用手动导出的格式
            base_name = os.path.splitext(os.path.basename(edf_file))[0]
            label_file = os.path.join(SleepConfig.LABEL_PATH, f"{base_name}_labels.txt")

            # 如果没有 meas_date，使用一个默认日期
            if meas_date is None:
                print("警告: meas_date 不可用，使用默认日期 01-01-2000 00:00:00")
                meas_date = datetime(2000, 1, 1, 0, 0, 0)

            with open(label_file, 'w') as f:
                # 写入表头
                f.write("Date,Time,Recording onset,Duration,Annotation,Linked channel\n")
                # 写入数据
                for onset, duration, description in processed_labels:
                    # 计算该段的绝对时间
                    onset_time = meas_date + timedelta(seconds=onset)
                    date_str = onset_time.strftime("%d-%m-%Y")
                    time_str = onset_time.strftime("%H:%M:%S")
                    # 写入一行
                    f.write(f"{date_str},{time_str},{onset:.2f},{duration:.2f},{description},\n")

            print(f"已保存标签文件: {label_file}")
            return label_file
        except Exception as e:
            print(f"处理标注文件失败: {e}")
            raise

    def _apply_filters(self, raw):
        self.sfreq = raw.info['sfreq']
        nyquist = self.sfreq / 2.0
        if self.sfreq != 100:
            raise ValueError(f"本处理仅支持 100 Hz 数据，当前: {self.sfreq} Hz")

        l_freq, h_freq = 0.3, 40.0
        print(f"带通滤波: {l_freq}-{h_freq} Hz (奈奎斯特={nyquist:.1f} Hz)")
        raw.filter(l_freq=l_freq, h_freq=h_freq,
                   method='fir', fir_window='hamming',
                   fir_design='firwin', phase='zero')

        notch_freq, notch_width = 48.0, 1.0
        print(f"陷波滤波: {notch_freq} Hz")
        raw.notch_filter(freqs=notch_freq, notch_widths=notch_width,
                         trans_bandwidth=1.0, method='fir',
                         fir_window='hamming', fir_design='firwin',
                         phase='zero')
        return raw

    def process_single_file(self, edf_path, annotations_path=None):
        print(f"\n=== 处理文件: {os.path.basename(edf_path)} ===")
        try:
            # 读取 EDF 文件以获取 meas_date
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

            meas_date = raw.info['meas_date']
            if meas_date is not None:
                if isinstance(meas_date, str) and '年' in meas_date:
                    print(f"警告: 检测到 meas_date 中包含 '年': {meas_date}")
                    meas_date_cleaned = meas_date.replace('年', '-').replace('月', '-').replace('日', '')
                    try:
                        meas_date = datetime.strptime(meas_date_cleaned, '%Y-%m-%d %H:%M:%S%z')
                        raw.info['meas_date'] = meas_date
                    except ValueError as e:
                        print(f"无法解析日期: {meas_date_cleaned}, 错误: {e}")
                        raw.info['meas_date'] = None
                print(f"记录起始时间: {raw.info['meas_date']}")
            else:
                print("记录起始时间: 未提供")

            print(f"信号长度: {raw.n_times} 样本 ({raw.n_times / raw.info['sfreq']:.2f} 秒)")

            # 查找或生成标签文件，传递 meas_date
            label_path = self.find_label_file(edf_path)
            if label_path is None:
                if annotations_path is None:
                    raise ValueError(f"未找到标签文件，且未提供标注文件: {edf_path}")
                label_path = self.process_and_save_labels(annotations_path, edf_path, meas_date=meas_date)
            print(f"标签文件: {label_path}")

            print("应用滤波...")
            raw = self._apply_filters(raw)

            data = raw.get_data()
            total = data.shape[1]
            print(f"数据形状: {data.shape}, 总样本数: {total}")

            print("解析标签文件...")
            labels = parse_annotation(label_path)

            print("分配标签到信号...")
            seg_labels = match_labels_to_segments(labels, total, raw.info['meas_date'], sfreq=raw.info['sfreq'])

            seg_len = SleepConfig.WINDOW_SEC * SleepConfig.SAMPLING_RATE
            n_seg = total // seg_len
            print(f"分段: 窗口长度 {seg_len} 样本 ({SleepConfig.WINDOW_SEC} 秒), 段数 {n_seg}")
            segments = data[0, :n_seg * seg_len].reshape(n_seg, seg_len)

            print("应用标准化...")
            segments = (segments - np.mean(segments, axis=1, keepdims=True)) / np.std(segments, axis=1, keepdims=True)
            segments = np.clip(segments, -5, 5)

            if SleepConfig.TARGET_CHANNEL not in raw.ch_names:
                available = '\n'.join(raw.ch_names)
                raise ValueError(f"目标通道 {SleepConfig.TARGET_CHANNEL} 不存在！可用通道：\n{available}")

            unique_labels = np.unique(seg_labels.numpy())
            if len(unique_labels) < 2:
                raise ValueError(f"文件 {edf_path} 仅包含单一标签: {unique_labels}")

            segments = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
            print(f"分段后数据形状: {segments.shape}, 标签形状: {seg_labels.shape}")

            mask = seg_labels != -1
            segments = segments[mask]
            seg_labels = seg_labels[mask]
            print(f"有效段数: {segments.shape[0]}/{n_seg} ({100 * segments.shape[0] / n_seg:.2f}%)")
            if segments.shape[0] == 0:
                raise ValueError(f"文件 {edf_path} 没有有效的标签段")

            base_name = os.path.splitext(os.path.basename(edf_path))[0]
            output_path = os.path.join(SleepConfig.PROCESSED_EEG_PATH, f"{base_name}.pt")
            torch.save({
                'eeg': segments,
                'labels': seg_labels,
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

    def process_all_files(self, annotations_dir=None, merge_output=False):
        if not os.path.isdir(SleepConfig.RAW_EEG_PATH):
            raise ValueError(f"数据目录不存在: {SleepConfig.RAW_EEG_PATH}")
        if not os.path.isdir(SleepConfig.LABEL_PATH):
            raise ValueError(f"标签目录不存在: {SleepConfig.LABEL_PATH}")
        if annotations_dir and not os.path.isdir(annotations_dir):
            raise ValueError(f"标注目录不存在: {annotations_dir}")

        os.makedirs(SleepConfig.OUTPUTRESULT_PATH, exist_ok=True)
        all_data, all_labels = [], []

        all_files = os.listdir(SleepConfig.RAW_EEG_PATH)
        print(f"RAW_EEG_PATH 中的所有文件: {all_files}")
        edf_list = [f for f in all_files if f.lower().endswith('.edf')]
        print(f"发现 {len(edf_list)} 个 EDF 文件: {edf_list}")

        if annotations_dir:
            hypno_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('-hypnogram.edf')]
            print(f"annotations_dir 中的所有 hypnogram 文件: {hypno_files}")

        for edf in edf_list:
            edf_full = os.path.join(SleepConfig.RAW_EEG_PATH, edf)
            if annotations_dir:
                base_name = os.path.splitext(edf)[0]
                prefix = base_name.split('J')[0]

                annot_file = None
                for hypno in hypno_files:
                    if hypno.startswith(prefix) and hypno.lower().endswith('-hypnogram.edf'):
                        annot_file = os.path.join(annotations_dir, hypno)
                        break

                if annot_file is None:
                    print(f"未找到与 {edf} 对应的标注文件 (前缀: {prefix})")
                    print(f"annotations_dir 中的可用 hypnogram 文件: {hypno_files}")
                    continue
                else:
                    print(f"找到匹配的标注文件: {annot_file} (用于 {edf})")
            else:
                annot_file = None

            try:
                segs, labs = self.process_single_file(edf_full, annot_file)
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

            data = torch.cat(all_data, dim=0)
            labels = torch.cat(all_labels, dim=0)
            print(f"\n合并数据: 总段数 {data.shape[0]}")
            print(f"总标签分布: {np.bincount(labels.numpy())}")

            out = os.path.join(SleepConfig.OUTPUTRESULT_PATH, 'processed_data_100hz.npz')
            np.savez_compressed(out, eeg=data.numpy().astype(np.float32), labels=labels.numpy())
            print(f"已保存合并文件: {out}, 样本数 {data.shape[0]}")

            return data, labels
        else:
            print("所有文件已单独保存为 .pt 文件")
            return None

if __name__ == "__main__":
    processor = EEGProcessor()
    try:
        processor.process_all_files(annotations_dir='../Hypnogram')
    except Exception as e:
        print(f"处理过程中出现错误: {e}")