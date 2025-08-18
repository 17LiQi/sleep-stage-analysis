# final_fusion.py
import numpy as np
import os

# (本地模块导入)
from src.utils.path_manager import get_path_manager
from src.utils.smoother import ViterbiSmoother
from src.utils.evaluation import generate_report_and_visuals


# ...

def run_final_fusion():
    path = get_path_manager()

    # --- 1. 加载两个系统的最终概率输出 ---
    # !! 您需要确保这两个文件已经被 04_final_evaluation.py 生成 !!
    probs_xgb_path = path.OUTPUT_ROOT / "final_evaluation/Stacking_Ensemble_XGB/stacking_probs_smoothed.npy"
    probs_lr_path = path.OUTPUT_ROOT / "final_evaluation/Stacking_Ensemble_LR/stacking_probs_smoothed.npy"

    if not probs_xgb_path.exists() or not probs_lr_path.exists():
        raise FileNotFoundError("请先分别运行XGBoost和LogisticRegression的Stacking评估，并确保它们保存了概率输出。")

    probs_xgb = np.load(probs_xgb_path)
    probs_lr = np.load(probs_lr_path)

    # 加载真实标签 (假设它们是一样的)
    # all_labels = ...

    # --- 2. 【核心】进行最终的加权平均融合 ---
    # 我们可以从一个简单的50/50平均开始
    # 或者根据它们各自的Kappa值来分配权重
    kappa_xgb = 0.7201
    kappa_lr = 0.7468

    weight_xgb = kappa_xgb / (kappa_xgb + kappa_lr)
    weight_lr = kappa_lr / (kappa_xgb + kappa_lr)

    print(f"融合权重 -> XGB: {weight_xgb:.2f}, LR: {weight_lr:.2f}")

    final_probs = weight_xgb * probs_xgb + weight_lr * probs_lr

    # --- 3. 得到最终预测并生成报告 ---
    final_preds = np.argmax(final_probs, axis=1)

    output_dir = path.OUTPUT_ROOT / "final_evaluation" / "Ultimate_Fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # (调用 generate_report_and_visuals 生成最终报告)


if __name__ == "__main__":
    run_final_fusion()