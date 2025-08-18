# src/utils/config.py

import yaml
from pathlib import Path
from typing import Dict, Any


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """递归地更新字典。"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_name: str, config_dir: Path) -> Dict[str, Any]:
    """
    加载并合并配置文件。支持从 'defaults' 键继承。
    """
    # 构造当前配置文件的完整路径
    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 递归加载并合并 'defaults'
    final_config = {}
    if 'defaults' in config:
        for default_name in config['defaults']:
            if default_name == '_self_':
                continue
            # 'defaults' 中的路径是相对于当前配置文件的
            base_config_path = (config_path.parent / f"{default_name}.yaml").resolve()
            # 需要将路径转换回相对于总配置目录的相对路径
            relative_base_path = base_config_path.relative_to(config_dir.parent)

            # 递归加载基础配置
            default_config = load_config(str(relative_base_path).replace('.yaml', ''), config_dir.parent)
            final_config = _deep_update(final_config, default_config)

    # 最后，用当前配置覆盖基础配置
    config.pop('defaults', None)
    final_config = _deep_update(final_config, config)

    return final_config