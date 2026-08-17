from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    dataset_root: Path
    split_root: Path
    output_dir: Path
    pretrained_path: Path
    vit_name: str
    img_size: int
    num_classes: int
    n_skip: int
    vit_patches_size: int
    folds: list[int]
    epochs: int
    batch_size: int
    grad_accum: int
    num_workers: int
    log_interval: int
    lr: float
    weight_decay: float
    adam_eps: float
    warmup_steps: int
    max_grad_norm: float
    seed: int
    patience: int
    min_delta: float
    min_epochs: int
    threshold: float
    mixed_precision: str
    lgd_aux_weight: float
    lgd_locator_weight: float
    locator_dilate_radius: int
    foreground_tversky_weight: float
    foreground_tversky_alpha: float
    foreground_tversky_beta: float
    mce_residual_init_weight: float
    lgd_gate_gamma: float
    device: str
    ruler_scale_csv: str = ""
    enable_mcrc: bool = True
    mcrc_gamma: float = 0.0
    mcrc_levels: str = "1,2"
    enable_dsrc: bool = False
    dsrc_gamma: float = 0.02
    dsrc_levels: str = "1,2"
    dsrc_dilations: str = "1,2,4"
    dsrc_reduction: int = 4
    enable_locator_head: bool = True
    enable_locator_gate: bool = True
    enable_aux_head: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        root = config_path.resolve().parents[1]

        def resolve(value: str) -> Path:
            path_value = Path(value)
            return path_value if path_value.is_absolute() else root / path_value

        path_keys = {"dataset_root", "split_root", "output_dir", "pretrained_path", "ruler_scale_csv"}
        resolved: dict[str, Any] = {
            key: resolve(value) if key in path_keys and value else value for key, value in data.items()
        }
        return cls(**resolved)

    def with_updates(self, **updates: Any) -> "ExperimentConfig":
        values = {field: getattr(self, field) for field in self.__dataclass_fields__}
        values.update({key: value for key, value in updates.items() if value is not None})
        return ExperimentConfig(**values)
