from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .models import TAKNet
from .models.vit_seg_modeling import CONFIGS as VIT_CONFIGS


def build_vit_config(cfg: ExperimentConfig):
    vit_config = copy.deepcopy(VIT_CONFIGS[cfg.vit_name])
    vit_config.n_classes = cfg.num_classes
    vit_config.n_skip = cfg.n_skip
    if "R50" in cfg.vit_name:
        grid = int(cfg.img_size / cfg.vit_patches_size)
        vit_config.patches.grid = (grid, grid)
    else:
        vit_config.patches.size = (cfg.vit_patches_size, cfg.vit_patches_size)
    return vit_config


def build_model(
    cfg: ExperimentConfig,
    device: torch.device,
    load_pretrained: bool = True,
) -> TAKNet:
    vit_config = build_vit_config(cfg)
    model = TAKNet(
        vit_config,
        img_size=cfg.img_size,
        mce_residual_init_weight=cfg.mce_residual_init_weight,
        lgd_gate_gamma=cfg.lgd_gate_gamma,
        enable_locator_head=cfg.enable_locator_head,
        enable_locator_gate=cfg.enable_locator_gate,
        enable_aux_head=cfg.enable_aux_head,
        enable_mcrc=cfg.enable_mcrc,
        mcrc_gamma=cfg.mcrc_gamma,
        mcrc_levels=cfg.mcrc_levels,
        enable_dsrc=cfg.enable_dsrc,
        dsrc_gamma=cfg.dsrc_gamma,
        dsrc_levels=cfg.dsrc_levels,
        dsrc_dilations=cfg.dsrc_dilations,
        dsrc_reduction=cfg.dsrc_reduction,
    )
    pretrained_path = Path(cfg.pretrained_path)
    if load_pretrained and not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained ViT weights not found: {pretrained_path}")
    if load_pretrained and pretrained_path.is_file():
        model.load_from(np.load(pretrained_path))
    return model.to(device)
