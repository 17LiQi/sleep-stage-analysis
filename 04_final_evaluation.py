# 04_final_evaluation.py
import os

import torch
import torch.nn.functional as F
import numpy as np
import joblib

from torch.utils.data import DataLoader
from tqdm import tqdm


# 本地模块导入
from src.utils.path_manager import get_path_manager
from src.utils.config import load_config
from src.dataset.dataset import SleepEDFDataset
from src.models import get_model
from src.utils.smoother import ViterbiSmoother
from src.utils.evaluation import generate_report_and_visuals

# ===================================================================
# >> 在这里配置要进行评估的模型和策略 <<
# ===================================================================
# 1. 评估的单模型列表
SINGLE_MODELS_TO_EVAL = [
    "Attn_CNN_GRU",
    "Attn_ResNet",
    "se_resnet"
]

# 2. 是否进行同质集成评估
EVALUATE_ENSEMBLE = True
ENSEMBLE_WEIGHTS = None # 简单平均

# 3. 评估的Stacking元模型列表
STACKING_META_MODELS_TO_EVAL = [
    'lr',
    'xgb'
]
# ===================================================================

def evaluate_and_save_probs(
        model,
        test_loader,
        device,
        output_dir,
        class_names,
        smoother=None,
        suffix=""
):
    """
    一个通用的评估函数，它会进行预测，保存概率，并生成报告。
    """
    model.to(device).eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"评估 {suffix}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # 保存概率文件
    np.save(os.path.join(output_dir, f"probs{suffix}.npy"), all_probs)

    # 生成报告
    generate_report_and_visuals(all_preds, all_labels, class_names, output_dir, f"_raw{suffix}")
    if smoother:
        log_probs = np.log(all_probs + 1e-9)
        smoothed_preds = smoother.predict(log_probs)
        generate_report_and_visuals(smoothed_preds, all_labels, class_names, output_dir, f"_smoothed{suffix}")

    return all_probs  # 返回概率以供后续使用

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


def evaluate_single_model(config_name, test_loader, device, output_dir, smoother):
    """在一个最终的单模型上进行评估，并生成报告和数据文件。"""
    print(f"\n--- 正在评估单模型: {config_name} ---")
    path = get_path_manager()
    config = load_config(f"models/{config_name}", path.CONFIG_ROOT)
    model_path = path.OUTPUT_ROOT / "final_single_models" / config['experiment_id'] / "best_model.pth"
    if not model_path.exists(): return

    model = get_model(config['model']['name'], **config['model']['params']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"评估 {config_name}"):
            outputs = model(inputs.to(device))
            probs = F.softmax(outputs, dim=1)
            all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs)

    np.save(os.path.join(output_dir, "final_probabilities.npy"), all_probs)
    np.save(os.path.join(output_dir, "final_labels.npy"), all_labels)

    generate_report_and_visuals(all_preds, all_labels, config['model']['class_names'], output_dir, "_raw")
    if smoother:
        smoothed_preds = smoother.predict(np.log(all_probs + 1e-9))
        generate_report_and_visuals(smoothed_preds, all_labels, config['model']['class_names'], output_dir, "_smoothed")

def evaluate_ensemble(model_configs, test_loader, device, output_dir, smoother, weights=None):
    """
    在一个模型集合上进行集成评估，并生成报告。
    """
    print("\n--- 正在评估集成模型 ---")
    path = get_path_manager()

    # --- 加载所有模型 ---
    ensemble_models = []
    for config_name in model_configs:
        config = load_config(f"models/{config_name}", path.CONFIG_ROOT)
        model_path = path.OUTPUT_ROOT / "final_single_models" / config['experiment_id'] / "best_model.pth"
        if not model_path.exists():
            print(f"警告: 未找到集成所需的模型 {model_path}，跳过。")
            continue
        model = get_model(config['model']['name'], **config['model']['params'])
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        ensemble_models.append(model.to(device).eval())

    if not ensemble_models:
        print("错误: 没有任何模型被成功加载，无法进行集成评估。")
        return

    print(f"成功加载 {len(ensemble_models)} 个模型用于集成。")

    # --- 集成预测 ---
    all_ensemble_probs, all_labels = [], []

    if weights:
        print(f"使用加权集成，权重: {weights}")
        w = torch.tensor(weights, device=device).view(len(weights), 1, 1)
    else:
        print("使用简单平均集成。")

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="集成预测中"):
            inputs = inputs.to(device)
            batch_probs_list = [F.softmax(model(inputs), dim=1) for model in ensemble_models]
            stacked_probs = torch.stack(batch_probs_list, dim=0)

            if weights:
                ensemble_probs = torch.sum(stacked_probs * w, dim=0)
            else:
                ensemble_probs = torch.mean(stacked_probs, dim=0)

            all_ensemble_probs.append(ensemble_probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_ensemble_probs = np.concatenate(all_ensemble_probs)
    all_preds = np.argmax(all_ensemble_probs, axis=1)
    all_labels = np.array(all_labels)

    np.save(os.path.join(output_dir, "final_probabilities.npy"), all_ensemble_probs)
    np.save(os.path.join(output_dir, "final_labels.npy"), all_labels)
    print(f"Ensemble 的概率和标签已保存。")


    # --- 生成报告 ---
    class_names = load_config(f"models/{model_configs[0]}", path.CONFIG_ROOT)['model']['class_names']
    generate_report_and_visuals(all_preds, all_labels, class_names, output_dir, "_raw_ensemble")

    if smoother:
        log_probs = np.log(all_ensemble_probs + 1e-9)
        smoothed_preds = smoother.predict(log_probs)
        generate_report_and_visuals(smoothed_preds, all_labels, class_names, output_dir, "_smoothed_ensemble")

def evaluate_stacking(base_model_configs, test_loader, device, output_dir, smoother, meta_model_type='lr'):
    """在一个模型集合上进行Stacking集成评估，并保存概率。"""
    print(f"\n--- 正在评估Stacking集成模型 ({meta_model_type.upper()}) ---")
    path = get_path_manager()

    # 1. --- 加载所有基础模型 ---
    base_models = []
    for config_name in base_model_configs:
        config = load_config(f"models/{config_name}", path.CONFIG_ROOT)
        model_path = path.OUTPUT_ROOT / "final_single_models" / config['experiment_id'] / "best_model.pth"
        if not model_path.exists():
            print(f"警告: 未找到Stacking所需的基础模型 {model_path}，跳过。")
            continue
        model = get_model(config['model']['name'], **config['model']['params'])
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        base_models.append(model.to(device).eval())

    if len(base_models) != len(base_model_configs):
        print(f"错误: 缺少部分基础模型，无法为 {meta_model_type.upper()} Stacking进行评估。")
        return

    # 2. --- 加载训练好的元模型 ---
    meta_model_path = path.OUTPUT_ROOT / "meta_models" / f"meta_model_{meta_model_type}.pkl"
    if not meta_model_path.exists():
        raise FileNotFoundError(f"未找到元模型: {meta_model_path}。请先运行 '03_train_meta_model.py'。")
    meta_model = joblib.load(meta_model_path)
    print(f"{meta_model_type.upper()} 元模型加载成功。")

    # 3. --- 为测试集生成元特征 ---
    test_meta_features_list, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"生成元特征 ({meta_model_type.upper()})"):
            inputs = inputs.to(device)
            batch_meta_features = [F.softmax(model(inputs), dim=1).cpu().numpy() for model in base_models]
            test_meta_features_list.append(np.concatenate(batch_meta_features, axis=1))
            all_labels.append(labels.numpy())

    test_meta_features = np.concatenate(test_meta_features_list)
    all_labels = np.concatenate(all_labels)

    # 4. --- 使用元模型进行概率预测 ---
    stacking_probs = meta_model.predict_proba(test_meta_features)
    stacking_preds_raw = np.argmax(stacking_probs, axis=1)

    # 保存概率文件
    np.save(os.path.join(output_dir, f"probs_stacking_{meta_model_type}_smoothed.npy"), stacking_probs)
    # 同时保存标签文件，方便05脚本加载
    np.save(os.path.join(output_dir, "final_test_labels.npy"), all_labels)
    print(f"Stacking ({meta_model_type.upper()}) 的概率和标签已保存。")

    # 5. --- 生成报告并保存概率 ---
    first_config = load_config(f"models/{base_model_configs[0]}", path.CONFIG_ROOT)
    class_names = first_config['model']['class_names']

    generate_report_and_visuals(stacking_preds_raw, all_labels, class_names, output_dir,
                                f"_stacking_{meta_model_type}_raw")

    if smoother:
        log_probs = np.log(stacking_probs + 1e-9)
        smoothed_preds = smoother.predict(log_probs)
        generate_report_and_visuals(smoothed_preds, all_labels, class_names, output_dir,
                                    f"_stacking_{meta_model_type}_smoothed")


def main():
    path = get_path_manager()
    base_config = load_config("base", path.CONFIG_ROOT)
    device = torch.device(base_config['device'])

    # 1. --- 加载最终测试集 ---
    print("--- 步骤1: 加载最终测试集 ---")
    test_data_path = path.DATA_ROOT / "held_out_test" / base_config['data']['channel']
    test_files = list(test_data_path.glob("*.npz"))
    test_dataset = SleepEDFDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. --- 在整个开发集上训练平滑器 ---
    smoother = None
    if base_config.get('viterbi', {}).get('enabled', False):
        print("\n--- 步骤2: 训练Viterbi平滑器 ---")
        dev_data_path = path.DATA_ROOT / "development" / base_config['data']['channel']
        dev_files = list(dev_data_path.glob("*.npz"))
        dev_dataset = SleepEDFDataset(dev_files)

        # 假设类别名在所有模型配置中都一样
        class_names = load_config(f"models/{SINGLE_MODELS_TO_EVAL[0]}", path.CONFIG_ROOT)['model']['class_names']
        smoother = ViterbiSmoother(n_classes=len(class_names))

        dev_labels_by_subject = {}
        for i in range(len(dev_dataset)):
            subject_id = dev_dataset.groups[i]
            if subject_id not in dev_labels_by_subject: dev_labels_by_subject[subject_id] = []
            dev_labels_by_subject[subject_id].append(dev_dataset.labels[i])
        smoother.fit(list(dev_labels_by_subject.values()))

    # 3. --- 评估独立的最终单模型 ---
    print("\n--- 步骤3: 评估独立的最终单模型 ---")
    for config_name in SINGLE_MODELS_TO_EVAL:
        model_eval_dir = path.OUTPUT_ROOT / "final_evaluation" / config_name
        model_eval_dir.mkdir(parents=True, exist_ok=True)
        evaluate_single_model(config_name, test_loader, device, str(model_eval_dir), smoother)

    # 4. --- 评估同质集成模型 ---
    if EVALUATE_ENSEMBLE:
        ensemble_eval_dir = path.OUTPUT_ROOT / "final_evaluation" / "Ensemble_Homogeneous"
        ensemble_eval_dir.mkdir(parents=True, exist_ok=True)
        evaluate_ensemble(SINGLE_MODELS_TO_EVAL, test_loader, device, str(ensemble_eval_dir), smoother,
                          weights=ENSEMBLE_WEIGHTS)

    # 5. --- 循环评估所有指定的Stacking模型 ---
    if STACKING_META_MODELS_TO_EVAL:
        for meta_type in STACKING_META_MODELS_TO_EVAL:
            stacking_eval_dir = path.OUTPUT_ROOT / "final_evaluation" / f"Stacking_{meta_type.upper()}"
            stacking_eval_dir.mkdir(parents=True, exist_ok=True)
            evaluate_stacking(
                base_model_configs=SINGLE_MODELS_TO_EVAL,
                test_loader=test_loader,
                device=device,
                output_dir=str(stacking_eval_dir),
                smoother=smoother,
                meta_model_type=meta_type
            )

if __name__ == "__main__":
    main()