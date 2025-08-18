# 03_train_meta_model.py

import numpy as np
import joblib
import xgboost as xgb

from src.utils.path_manager import get_path_manager


def train_meta_model():
    print("--- 步骤3: 开始训练Stacking的元模型 ---")
    path = get_path_manager()

    # 1. --- 加载元特征和标签 ---
    meta_features_dir = path.OUTPUT_ROOT / "meta_features"
    meta_features_path = meta_features_dir / "meta_features.npy"
    meta_labels_path = meta_features_dir / "meta_labels.npy"

    if not meta_features_path.exists() or not meta_labels_path.exists():
        raise FileNotFoundError("未找到元特征或标签文件。请先运行 '02_generate_meta_features.py'。")

    X_meta = np.load(meta_features_path)
    y_meta = np.load(meta_labels_path)

    print(f"加载元特征，形状: {X_meta.shape}")

    # 2. --- 定义并训练元模型 ---
    # 2. --- 定义并训练元模型 (XGBoost) ---
    print("正在训练 XGBoost 元模型...")

    # XGBoost可以自动处理类别不平衡
    # 通过 scale_pos_weight 参数，但对于多分类，我们先让它自己学
    # 也可以通过 early_stopping_rounds 来防止过拟合
    meta_model = xgb.XGBClassifier(
        objective='multi:softprob',  # 输出概率
        n_estimators=500,  # 树的数量
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    # 我们可以创建一个小的验证集来使用早停
    # from sklearn.model_selection import train_test_split
    # X_train, X_val, y_train, y_val = train_test_split(X_meta, y_meta, test_size=0.2, stratify=y_meta, random_state=42)
    # meta_model.fit(X_train, y_train,
    #                eval_set=[(X_val, y_val)],
    #                early_stopping_rounds=50,
    #                verbose=False)

    # 为简化，我们先在全部元数据上训练
    meta_model.fit(X_meta, y_meta)

    print("元模型训练完成。")

    # 3. --- 保存元模型 ---
    meta_models_dir = path.OUTPUT_ROOT / "meta_models"
    meta_models_dir.mkdir(parents=True, exist_ok=True)
    meta_model_path = meta_models_dir / "meta_model_xgb.pkl"

    joblib.dump(meta_model, meta_model_path)
    print(f"元模型已保存至: {meta_model_path}")


if __name__ == "__main__":
    train_meta_model()