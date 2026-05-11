# utils/config_loader.py

import json
import os
from pathlib import Path

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"


def load_config(dataset: str, env: str = None) -> dict:
    """
    Fusionne env + dataset en un seul dict.
    L'env est résolu via : argument > variable MAIN_ENV > défaut 'local'

    Args:
        dataset : nom du fichier dataset sans extension, ex: "test_data_1"
        env     : "local" | "server" | "cluster"
    """
    env = env or os.environ.get("MAIN_ENV", "local")

    env_path     = CONFIG_DIR / f"env.{env}.json"
    dataset_path = CONFIG_DIR / "datasets" / f"{dataset}.json"

    with open(env_path)     as f: cfg = json.load(f)
    with open(dataset_path) as f: cfg.update(json.load(f))

    # Stocker le nom du dataset (= nom du fichier json)
    cfg["dataset_name"] = dataset

    # Construire les paths finaux
    cfg["dataset_path"]  = str(Path(cfg["root_data_path"])   / cfg["relative_dataset_path"])
    cfg["fs_subjects_dir"] = str(Path(cfg["root_data_path"])   / cfg["os_SUBJECTS_DIR"])

    return cfg