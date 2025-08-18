# src/utils/evaluation.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, f1_score


def calculate_metrics(labels, preds, class_names):
    """
    一个核心的辅助函数，只负责计算所有性能指标并返回一个结构化的字典。
    """
    if len(labels) == 0 or len(preds) == 0:
        return {}  # 如果没有数据，返回空字典

    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)

    # 使用output_dict=True来一次性获取所有指标
    report_dict = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0
    )

    results = {
        'accuracy': acc,
        'kappa': kappa,
        'f1_macro': report_dict['macro avg']['f1-score']
    }

    # 动态地为每个类别添加F1和Recall
    for cname in class_names:
        s_cname = cname.replace('/', '')  # 创建安全的key
        if cname in report_dict:
            results[f'f1_{s_cname}'] = report_dict[cname]['f1-score']
            results[f'recall_{s_cname}'] = report_dict[cname]['recall']

    return results


def generate_report_and_visuals(
        preds,
        labels,
        class_names,
        output_dir,
        suffix: str = ""
):
    """
    生成并保存详细的分类报告和混淆矩阵图。
    这个函数现在只负责“可视化”和“保存”，不再计算和返回指标。
    在交叉验证的每一折内部和在没有交叉验证的最终测试中生成最终的、单一的评估报告和图表
    """
    if len(labels) == 0 or len(preds) == 0:
        print(f"警告 ({suffix}): 标签或预测为空，跳过报告生成。")
        return

    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)

    # 1. 生成并保存分类报告 (.txt)
    report_text = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)
    final_suffix = f"_{suffix}" if suffix else ""
    report_path = os.path.join(output_dir, f"classification_report{final_suffix}.txt")

    with open(report_path, "w") as f:
        f.write(f"Evaluation Report ({suffix})\n" + "=" * 30 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n\n")
        f.write(report_text)
    print(f"分类报告 ({suffix}) 已保存至: {report_path}")

    # 2. 生成并保存混淆矩阵图 (.png)
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({suffix.replace('_', ' ')}, Accuracy: {acc:.2%})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(output_dir, f"confusion_matrix{final_suffix}.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵 ({suffix}) 已保存至: {cm_path}")


def summarize_cv_results(all_fold_results, output_dir, experiment_name):
    """
    将交叉验证的多折结果汇总成一个总结报告。
    只在有交叉验证的情况下使用
    """
    if not all_fold_results:
        print("警告: a`ll_fold_results`为空，无法生成总结报告。")
        return

    # 1. 创建DataFrame
    summary_df = pd.DataFrame(all_fold_results)

    # 2. 保存原始的、逐折的结果到CSV
    summary_df.to_csv(os.path.join(output_dir, f"summary_raw_{experiment_name}.csv"))

    # 3. 计算均值和标准差
    mean_stats = summary_df.mean()
    std_stats = summary_df.std()

    # 4. 生成总结文本 (.txt)
    summary_text = f"交叉验证总览报告: {experiment_name}\n" + "=" * 40 + "\n\n"
    summary_text += f"总折数: {len(summary_df)}\n\n"
    summary_text += "各折性能指标:\n"
    summary_text += summary_df.to_string(index=True) + "\n\n"
    summary_text += "-" * 40 + "\n\n"
    summary_text += "平均性能 (Mean ± Std Dev):\n"

    # 对DataFrame中的每一列（每个指标）进行格式化输出
    for metric in mean_stats.index:
        summary_text += f"- {metric}: {mean_stats[metric]:.4f} ± {std_stats[metric]:.4f}\n"

    # 5. 保存总结文件
    summary_report_path = os.path.join(output_dir, f"summary_report_{experiment_name}.txt")
    with open(summary_report_path, "w") as f:
        f.write(summary_text)

    print(f"交叉验证总结报告已保存至: {summary_report_path}")