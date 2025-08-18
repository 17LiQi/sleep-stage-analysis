# 03a_find_best_threshold.py

import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 本地模块导入
from src.utils.path_manager import get_path_manager
from src.utils.config import load_config

# ===================================================================
# >> 在这里配置阈值搜索 <<
# ===================================================================
# 我们需要一个配置文件来获取类别名称
# 假设我们用AttnResNet的配置，因为它类别名最全
BASE_CONFIG_NAME = "models/Attn_ResNet"


# ===================================================================

def apply_threshold(probabilities, n1_threshold):
    """
    应用自定义的N1概率阈值来生成最终预测。
    """
    # 1. 默认预测是标准 argmax
    preds = np.argmax(probabilities, axis=1)

    # 2. 找到所有N1概率超过阈值的样本
    #    probabilities[:, 1] 表示所有样本的N1类别的概率
    n1_override_indices = np.where(probabilities[:, 1] >= n1_threshold)[0]

    # 3. 将这些样本的预测强制修改为N1 (标签为1)
    preds[n1_override_indices] = 1

    return preds


def find_best_threshold():
    print("--- 步骤3.5: 开始为Stacking元模型寻找最佳N1阈值 ---")
    path = get_path_manager()
    config = load_config(BASE_CONFIG_NAME, path.CONFIG_ROOT)
    class_names = config['model']['class_names']

    # 1. --- 加载元数据和训练好的元模型 ---
    meta_features_dir = path.OUTPUT_ROOT / "meta_features"
    meta_model_dir = path.OUTPUT_ROOT / "meta_models"

    X_meta = np.load(meta_features_dir / "meta_features.npy")
    y_meta = np.load(meta_features_dir / "meta_labels.npy")

    meta_model = joblib.load(meta_model_dir / "meta_model_xgb.pkl")
    print("元数据和元模型加载成功。")

    # 2. --- 通过交叉验证，为每个阈值评估性能 ---
    # 我们在元数据集内部进行交叉验证，以获得对阈值性能的无偏估计

    # 定义要搜索的阈值范围
    thresholds_to_search = np.arange(0.1, 0.51, 0.01)

    # 初始化用于存储每个阈值性能的字典
    f1_n1_scores = {t: [] for t in thresholds_to_search}
    kappa_scores = {t: [] for t in thresholds_to_search}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['seed'])

    print("\n通过交叉验证搜索最佳阈值...")
    for fold, (train_idx, val_idx) in enumerate(tqdm(
            skf.split(X_meta, y_meta), total=5, desc="CV for Thresholding"
    )):
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y_meta[train_idx], y_meta[val_idx]

        # 在当前折的训练集上训练一个临时的元模型
        temp_meta_model = joblib.load(meta_model_dir / "meta_model_xgb.pkl")  # 加载原型
        temp_meta_model.fit(X_train, y_train)

        # 在验证集上预测概率
        val_probs = temp_meta_model.predict_proba(X_val)

        # 评估每个阈值的效果
        for t in thresholds_to_search:
            preds_with_threshold = apply_threshold(val_probs, t)

            # 计算并存储N1的F1分数和总体的Kappa
            f1_n1 = f1_score(y_val, preds_with_threshold, labels=[1], average='macro', zero_division=0)
            kappa = cohen_kappa_score(y_val, preds_with_threshold)

            f1_n1_scores[t].append(f1_n1)
            kappa_scores[t].append(kappa)

    # 3. --- 计算平均性能并找到最佳阈值 ---
    avg_f1_n1 = {t: np.mean(scores) for t, scores in f1_n1_scores.items()}
    avg_kappa = {t: np.mean(scores) for t, scores in kappa_scores.items()}

    # 找到使N1 F1分数最大化的最佳阈值
    best_threshold_for_n1 = max(avg_f1_n1, key=avg_f1_n1.get)

    print("\n--- 阈值搜索结果 ---")
    print(f"最佳N1 F1分数对应的阈值: {best_threshold_for_n1:.2f}")
    print(f"  - 在此阈值下，交叉验证的平均 N1 F1-Score: {avg_f1_n1[best_threshold_for_n1]:.4f}")
    print(f"  - 在此阈值下，交叉验证的平均 Kappa: {avg_kappa[best_threshold_for_n1]:.4f}")

    # (可选) 找到使Kappa最大化的阈值
    best_threshold_for_kappa = max(avg_kappa, key=avg_kappa.get)
    print(f"\n最佳Kappa对应的阈值: {best_threshold_for_kappa:.2f}")
    print(f"  - 在此阈值下，交叉验证的平均 N1 F1-Score: {avg_f1_n1[best_threshold_for_kappa]:.4f}")
    print(f"  - 在此阈值下，交叉验证的平均 Kappa: {avg_kappa[best_threshold_for_kappa]:.4f}")

    # 4. --- 可视化结果 ---
    plt.figure(figsize=(12, 6))
    plt.plot(list(avg_f1_n1.keys()), list(avg_f1_n1.values()), marker='o', label='Average N1 F1-Score')
    plt.plot(list(avg_kappa.keys()), list(avg_kappa.values()), marker='x', label='Average Kappa Score')
    plt.axvline(x=best_threshold_for_n1, color='r', linestyle='--',
                label=f'Best Threshold for N1 ({best_threshold_for_n1:.2f})')
    plt.axvline(x=best_threshold_for_kappa, color='g', linestyle='--',
                label=f'Best Threshold for Kappa ({best_threshold_for_kappa:.2f})')
    plt.xlabel("N1 Prediction Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning for Stacking Meta-Model")
    plt.legend()
    plt.grid(True)

    # 保存图表
    viz_path = path.OUTPUT_ROOT / "meta_models" / "threshold_tuning_curve.png"
    plt.savefig(viz_path)
    print(f"\n阈值调优曲线已保存至: {viz_path}")
    plt.show()

    print("\n--- 行动建议 ---")
    print(f"请将您选择的最佳阈值 ( {best_threshold_for_n1:.2f}) 更新到 '04_final_evaluation.py' 脚本中，")
    print("然后在该脚本的 evaluate_stacking 函数中应用这个新规则，以获得最终的测试集性能。")


if __name__ == "__main__":
    find_best_threshold()