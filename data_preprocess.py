# src/preprocess/data_preprocess.py

import os
import glob
import ntpath
import logging
from typing import List
import numpy as np
import pyedflib
from sklearn.model_selection import train_test_split

# 本地模块导入
from src.utils.path_manager import get_path_manager

# ===================================================================
# >> 在这里配置您的预处理任务 <<
# ===================================================================

# 1. 定义您想要处理的所有EEG通道
# CHANNELS_TO_PROCESS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
CHANNELS_TO_PROCESS = ["EEG Fpz-Cz"]

# 2. 定义最终测试集的划分比例
TEST_SET_RATIO = 0.2  # 20% 的数据作为最终测试集

# 3. 定义随机种子，以保证每次划分结果都一样
RANDOM_SEED = 42

# 4. 定义睡眠核心时段前后的清醒期边缘 (分钟)
WAKE_EDGE_MINS = 30
# ===================================================================


# --- 配置区 (保持不变) ---
STAGE_DICT = {
    "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6
}
ANN_TO_LABEL = {
    "Sleep stage W": STAGE_DICT["W"], "Sleep stage 1": STAGE_DICT["N1"],
    "Sleep stage 2": STAGE_DICT["N2"], "Sleep stage 3": STAGE_DICT["N3"],
    "Sleep stage 4": STAGE_DICT["N3"], "Sleep stage R": STAGE_DICT["REM"],
    "Sleep stage ?": STAGE_DICT["UNK"], "Movement time": STAGE_DICT["MOVE"]
}


# --- 日志设置 ---
def setup_logger(log_file_path: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# --- 核心处理函数 ---
def process_subject(
        psg_filepath: str, ann_filepath: str, output_dir: str, target_channels: List[str]
):
    logger = logging.getLogger(__name__)
    subject_id = ntpath.basename(psg_filepath).replace("-PSG.edf", "")
    logger.info(f"\n--- 开始处理受试者: {subject_id} ---")

    psg_f, ann_f = None, None
    try:
        psg_f = pyedflib.EdfReader(psg_filepath)
        ann_f = pyedflib.EdfReader(ann_filepath)

        # 1. 验证文件元数据
        if psg_f.getStartdatetime() != ann_f.getStartdatetime():
            logger.warning(f"[{subject_id}] 开始时间不匹配！跳过。")
            return
        epoch_duration = psg_f.datarecord_duration
        if epoch_duration != 30.0:
            logger.warning(f"[{subject_id}] 分段时长为 {epoch_duration}s (非30s)，跳过。")
            return

        # 2. 解析标签
        labels = []
        expected_onset = 0
        ann_onsets, ann_durations, ann_stages_b = ann_f.readAnnotations()
        for onset, duration, stage_b in zip(ann_onsets, ann_durations, ann_stages_b):
            stage_str = str(stage_b).strip()
            if int(onset) != expected_onset or duration % epoch_duration != 0:
                logger.warning(f"[{subject_id}] 标签不连续或时长不规范。跳过。")
                return
            if stage_str in ANN_TO_LABEL:
                labels.extend([ANN_TO_LABEL[stage_str]] * int(duration / epoch_duration))
                expected_onset += int(duration)
            else:
                logger.warning(f"[{subject_id}] 发现未知标注 '{stage_str}'，跳过。")
                return
        labels = np.array(labels, dtype=np.int32)

        # 3. 为每个目标通道提取、处理并保存数据
        available_channels = [ch.strip() for ch in psg_f.getSignalLabels()]
        for ch_name in target_channels:
            if ch_name not in available_channels:
                logger.warning(f"[{subject_id}] 通道 '{ch_name}' 未找到，跳过该通道。")
                continue

            ch_idx = available_channels.index(ch_name)
            sampling_rate = psg_f.getSampleFrequency(ch_idx)
            n_epoch_samples = int(epoch_duration * sampling_rate)

            full_signal = psg_f.readSignal(ch_idx)
            n_epochs_in_signal = len(full_signal) // n_epoch_samples
            signals_reshaped = full_signal[:n_epochs_in_signal * n_epoch_samples].reshape(-1, n_epoch_samples)

            current_labels = labels[:n_epochs_in_signal]

            # 4. 智能选择“睡眠核心+边缘”数据
            non_wake_indices = np.where(current_labels != STAGE_DICT["W"])[0]
            if len(non_wake_indices) == 0:
                logger.info(f"[{subject_id}] 在有效信号范围内不含睡眠阶段，跳过。")
                continue

            epochs_per_min = int(60 / epoch_duration)
            edge_epochs = WAKE_EDGE_MINS * epochs_per_min
            select_start = max(0, non_wake_indices[0] - edge_epochs)
            select_end = min(len(current_labels), non_wake_indices[-1] + edge_epochs)

            select_indices = np.arange(select_start, select_end)

            # 5. 移除无用阶段
            labels_subset = current_labels[select_indices]
            valid_mask = (labels_subset != STAGE_DICT["MOVE"]) & (labels_subset != STAGE_DICT["UNK"])
            final_indices = select_indices[valid_mask]

            x_1d = signals_reshaped[final_indices].astype(np.float32)
            y = current_labels[final_indices]

            if len(x_1d) == 0:
                logger.warning(f"[{subject_id}][{ch_name}] 经过筛选后无有效数据，不保存。")
                continue

            # 6. 保存NPZ文件
            ch_output_dir = os.path.join(output_dir, ch_name.replace(' ', '_').replace('-', '_'))
            os.makedirs(ch_output_dir, exist_ok=True)
            output_filename = f"{subject_id}.npz"
            save_path = os.path.join(ch_output_dir, output_filename)

            np.savez(save_path, x_1d=x_1d, y=y, fs=sampling_rate, ch_name=ch_name)
            logger.info(f"[{subject_id}][{ch_name}] 成功保存1D数据到: {save_path}")

    except Exception as e:
        logger.error(f"处理受试者 {subject_id} 时发生严重错误: {e}", exc_info=False)
    finally:
        if psg_f: psg_f.close()
        if ann_f: ann_f.close()


def main():
    path = get_path_manager()
    raw_data_dir = path.SLEEP_EDF_CASSETTE
    processed_data_root = path.DATA_ROOT

    log_file = os.path.join(processed_data_root, "preprocess_1d_split.log")
    logger = setup_logger(log_file)
    logger.info("开始1D数据预处理与划分流程...")

    all_psg_files = np.array(sorted(glob.glob(os.path.join(raw_data_dir, "*PSG.edf"))))
    all_ann_files = np.array(sorted(glob.glob(os.path.join(raw_data_dir, "*Hypnogram.edf"))))

    # 使用文件名进行配对和划分
    subject_ids = np.array([ntpath.basename(f).replace("-PSG.edf", "")[:6] for f in all_psg_files])

    indices = np.arange(len(subject_ids))
    dev_indices, test_indices = train_test_split(
        indices, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED
    )

    dev_output_dir = os.path.join(processed_data_root, "development")
    test_output_dir = os.path.join(processed_data_root, "held_out_test")

    logger.info(
        f"\n总共 {len(all_psg_files)} 个受试者。划分为: {len(dev_indices)} 开发集 和 {len(test_indices)} 测试集。")

    logger.info("\n--- 正在处理开发集 (Development Set) ---")
    for idx in dev_indices:
        process_subject(all_psg_files[idx], all_ann_files[idx], dev_output_dir, CHANNELS_TO_PROCESS)

    logger.info("\n--- 正在处理最终测试集 (Held-out Test Set) ---")
    for idx in test_indices:
        process_subject(all_psg_files[idx], all_ann_files[idx], test_output_dir, CHANNELS_TO_PROCESS)

    logger.info("\n所有1D数据预处理并划分完毕！")


if __name__ == "__main__":
    main()