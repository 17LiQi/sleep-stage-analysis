# 05_final_fusion.py

import os
import numpy as np
from datetime import datetime

# 本地模块导入
from src.utils.path_manager import get_path_manager
from src.utils.smoother import ViterbiSmoother
from src.utils.evaluation import generate_report_and_visuals
from src.utils.config import load_config
from src.dataset.dataset import SleepEDFDataset  # 需要它来训练HMM

# ===================================================================
# >> 在这里配置最终的融合策略 <<
# ===================================================================
# 1. 定义要融合的系统的输出目录名
#    这些目录名是在04脚本中创建的
SYSTEMS_TO_FUSE = {
    "xgb_stacking": "Stacking_XGB",
    "lr_stacking": "Stacking_LR"
}

# 2. 定义融合权重
#    顺序必须与上面的字典对应
FUSION_WEIGHTS = [0.46, 0.54]

BASE_CONFIG_NAME = "models/Attn_ResNet"


# ===================================================================


def run_final_fusion():
    print("--- 最终步骤: 开始进行多系统融合 ---")
    path = get_path_manager()
    config = load_config(BASE_CONFIG_NAME, path.CONFIG_ROOT)



    # --- 1. 加载所有必需的概率文件和标签文件 ---
    print("正在加载各个系统的预测概率...")

    probabilities = {}
    all_labels = None

    base_eval_path = path.OUTPUT_ROOT / "final_evaluation"

    for key, dir_name in SYSTEMS_TO_FUSE.items():
        # 我们使用平滑前的概率进行融合，因为HMM应该在融合后应用
        # !! 请确保04脚本产出的概率文件名是统一的 !!
        prob_file = base_eval_path / dir_name / f"probs_stacking_{dir_name.split('_')[-1].lower()}_smoothed.npy"  # 这是一个示例路径
        # 您需要根据04脚本的实际输出来调整文件名
        # 为了简单起见，我们假设概率文件都叫 final_test_labels.npy
        # prob_file_simple = base_eval_path / dir_name / "final_test_labels.npy"

        if not prob_file.exists():
            raise FileNotFoundError(f"未找到概率文件: {prob_file}。请先完整运行 '04_final_evaluation.py'。")

        probabilities[key] = np.load(prob_file)

        # 加载一次标签即可
        if all_labels is None:
            labels_file = base_eval_path / dir_name / "final_test_labels.npy"
            if labels_file.exists():
                all_labels = np.load(labels_file)

    if all_labels is None:
        raise FileNotFoundError("未能加载任何标签文件。")

    # --- 2. 执行加权平均融合 ---
    print(f"使用权重 {FUSION_WEIGHTS} 进行融合...")

    fused_probs = np.zeros_like(list(probabilities.values())[0])

    for i, key in enumerate(SYSTEMS_TO_FUSE.keys()):
        fused_probs += probabilities[key] * FUSION_WEIGHTS[i]

    fused_preds_raw = np.argmax(fused_probs, axis=1)

    # --- 3. Viterbi平滑与最终报告 ---
    print("\n--- 训练最终的Viterbi平滑器 ---")
    dev_data_path = path.DATA_ROOT / "development" / config['data']['channel']
    dev_files = list(dev_data_path.glob("*.npz"))
    dev_dataset = SleepEDFDataset(dev_files)

    # 假设类别名在所有模型配置中都一样
    class_names = config['model']['class_names']
    smoother = ViterbiSmoother(n_classes=len(class_names))

    dev_labels_by_subject = {}
    for i in range(len(dev_dataset)):
        subject_id = dev_dataset.groups[i]
        if subject_id not in dev_labels_by_subject: dev_labels_by_subject[subject_id] = []
        dev_labels_by_subject[subject_id].append(dev_dataset.labels[i])
    smoother.fit(list(dev_labels_by_subject.values()))

    print("\n--- 生成最终的融合评估报告 ---")
    output_dir = path.OUTPUT_ROOT / "final_evaluation" / "Ultimate_Fusion"
    output_dir.mkdir(parents=True, exist_ok=True)


    generate_report_and_visuals(fused_preds_raw, all_labels, class_names, str(output_dir), "_raw_fusion")

    log_fused_probs = np.log(fused_probs + 1e-9)
    smoothed_preds = smoother.predict(log_fused_probs)
    generate_report_and_visuals(smoothed_preds, all_labels, class_names, str(output_dir), "_smoothed_fusion")

    print("\n研究流程结束！最终融合报告已生成。")


if __name__ == "__main__":
    run_final_fusion()