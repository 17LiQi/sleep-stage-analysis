import numpy as np
import pandas as pd
import torch
from SleepConfig import SleepConfig


def parse_annotation(label_path):
    """
    解析标签文件，将长时段标签拆分为 30 秒窗口
    :param label_path: txt格式的标签文件路径
    :return: 标签列表，每个元素为(start_sample, end_sample, label)
    """
    labels = []
    window_sec = SleepConfig.WINDOW_SEC  # 30 秒窗口
    sampling_rate = SleepConfig.SAMPLING_RATE  # 100 Hz

    try:
        # 读取 CSV 文件
        print(f"尝试读取标签文件: {label_path}")
        df = pd.read_csv(label_path, encoding='utf-8', skipinitialspace=True)
        print(f"标签文件 {label_path} 共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")

        # 检查必要列
        required_columns = ['Recording onset', 'Duration', 'Annotation']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"标签文件缺少列: {missing_columns}")

        for idx, row in df.iterrows():
            try:
                # 提取字段
                start_sec = row['Recording onset']
                duration = row['Duration']
                stage = row['Annotation'].strip()

                # 验证字段
                try:
                    start_sec = float(start_sec)
                    duration = float(duration)
                    if start_sec < 0:
                        raise ValueError(f"Recording onset 不能为负: {start_sec}")
                    if duration <= 0:
                        raise ValueError(f"Duration 必须为正: {duration}")
                except (TypeError, ValueError) as e:
                    raise ValueError(f"无效的 Recording onset 或 Duration: {start_sec}, {duration}, 错误: {e}")

                # 验证 Annotation
                label = SleepConfig.STAGE_WINDOW_SEC.get(stage, -1)
                if label == -1:
                    print(f"警告: 行 {idx + 1} 未知 Annotation: {stage}, 跳过")
                    continue

                # 计算时间段
                end_sec = start_sec + duration

                # 将时间段拆分为 30 秒窗口
                current_sec = start_sec
                while current_sec < end_sec:
                    window_end_sec = min(current_sec + window_sec, end_sec)
                    if window_end_sec - current_sec < window_sec / 2:
                        # 忽略小于 15 秒的残余窗口
                        print(f"行 {idx + 1}: 跳过残余窗口 {current_sec}-{window_end_sec} 秒（不足 {window_sec / 2} 秒）")
                        break

                    start_sample = int(current_sec * sampling_rate)
                    end_sample = int(window_end_sec * sampling_rate)
                    labels.append((start_sample, end_sample, label))
                    current_sec += window_sec

            except Exception as e:
                print(f"行 {idx + 1} 解析失败: {row.to_dict()}, 错误: {e}")
                continue  # 跳过有问题的行

        if not labels:
            raise ValueError(f"标签文件 {label_path} 没有有效的标签")

        # 按 start_sample 排序
        labels = sorted(labels, key=lambda x: x[0])
        print(f"解析完成: 共生成 {len(labels)} 个 30 秒窗口标签")

        # 检查时间段连续性
        for i in range(1, len(labels)):
            if labels[i][0] < labels[i - 1][1]:
                print(f"警告: 标签 {i} 与 {i - 1} 重叠: {labels[i - 1]} -> {labels[i]}")
            elif labels[i][0] > labels[i - 1][1]:
                print(f"警告: 标签 {i} 与 {i - 1} 存在间隙: {labels[i - 1][1]} -> {labels[i][0]}")

        return labels

    except Exception as e:
        print(f"解析标签文件 {label_path} 失败: {e}")
        raise

def match_labels_to_segments(labels, total_samples, recording_start=None, sfreq=SleepConfig.SAMPLING_RATE):
    """
    将标签分配到 30 秒窗口，返回张量格式
    :param labels: 标签列表 [(start_sample, end_sample, label), ...]
    :param total_samples: 信号总样本数
    :param recording_start: 记录起始时间 (raw.info['meas_date'])
    :param sfreq: 采样率
    :return: 张量格式的标签数组 (torch.Tensor)
    """
    print(f"\n开始分配标签: 总样本数 {total_samples}, 采样率 {sfreq} Hz")
    if recording_start:
        print(f"记录起始时间: {recording_start}")

    window_sec = SleepConfig.WINDOW_SEC
    window_samples = int(window_sec * sfreq)
    n_windows = total_samples // window_samples
    seg_labels = np.full(n_windows, -1, dtype=np.int64)
    print(f"窗口长度: {window_samples} 样本 ({window_sec} 秒), 窗口数: {n_windows}")

    for i, (start_sample, end_sample, label) in enumerate(labels):
        try:
            if start_sample < 0 or end_sample > total_samples:
                print(f"警告: 标签 {i+1} 超出信号范围: {start_sample}-{end_sample}, 信号长度 {total_samples}")
                continue
            if start_sample >= end_sample:
                print(f"警告: 标签 {i+1} 无效时间段: {start_sample} >= {end_sample}")
                continue

            start_window = start_sample // window_samples
            end_window = (end_sample - 1) // window_samples + 1

            for w in range(max(0, start_window), min(n_windows, end_window)):
                seg_labels[w] = label

        except Exception as e:
            print(f"标签 {i+1} 处理失败: 样本 {start_sample}-{end_sample}, 标签 {label}, 错误: {e}")
            continue

    valid_labels = seg_labels[seg_labels != -1]
    print(f"标签分配完成: 有效窗口数 {len(valid_labels)}/{n_windows} ({100 * len(valid_labels)/n_windows:.2f}%)")
    print(f"标签分布: {np.bincount(seg_labels[seg_labels != -1], minlength=6)}")

    # 转换为 PyTorch 张量
    seg_labels_tensor = torch.tensor(seg_labels, dtype=torch.long)
    return seg_labels_tensor