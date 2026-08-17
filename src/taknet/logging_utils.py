from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expected_count(file_list: Path) -> int:
    return sum(1 for line in file_list.read_text(encoding="utf-8").splitlines() if line.strip())


def metric_complete(metric_path: Path, file_list: Path) -> bool:
    if not metric_path.is_file():
        return False
    try:
        data = json.loads(metric_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("n") == expected_count(file_list) and data.get("global", {}).get("dice") is not None


