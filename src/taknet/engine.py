from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import make_loader, tensor_to_device
from .evaluation import evaluate_prediction_dir
from .logging_utils import seed_everything, write_json
from .losses import (
    lgd_auxiliary_loss,
    lgd_localization_loss,
    foreground_tversky_loss,
    segmentation_loss,
    valid_label_mask,
)
from .modeling import build_model


def device_from_config(cfg: ExperimentConfig) -> torch.device:
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(cfg.device)


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = tensor_to_device(batch, device)
            logits = model(batch["bmode"], batch["ceus"])
            label = batch["label"].long()
            losses.append(float(segmentation_loss(logits, label, cfg.num_classes).detach().cpu()))
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            valid = valid_label_mask(label)
            pred = (prob > cfg.threshold) & valid
            gt = (label == 1) & valid
            tp += torch.logical_and(pred, gt).sum().item()
            fp += torch.logical_and(pred, valid & ~gt).sum().item()
            fn += torch.logical_and(valid & ~pred, gt).sum().item()
            tn += torch.logical_and(valid & ~pred, valid & ~gt).sum().item()
    eps = 1e-6
    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_dice": float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)),
        "val_iou": float((tp + eps) / (tp + fp + fn + eps)),
        "val_precision": float((tp + eps) / (tp + fp + eps)),
        "val_recall": float((tp + eps) / (tp + fn + eps)),
        "val_specificity": float((tn + eps) / (tn + fp + eps)),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_dice: float,
    cfg: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_dice,
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(cfg).items()
                },
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_one_fold(cfg: ExperimentConfig, fold: int, logger: logging.Logger) -> Path:
    device = device_from_config(cfg)
    seed_everything(cfg.seed)
    fold_name = f"fold_{fold}"
    checkpoint_dir = cfg.output_dir / "checkpoints" / fold_name
    log_dir = cfg.output_dir / "logs" / fold_name
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"

    train_loader = make_loader(
        cfg,
        cfg.split_root / fold_name / "train_files.txt",
        augment=True,
        shuffle=True,
    )
    val_loader = make_loader(
        cfg,
        cfg.split_root / fold_name / "val_files.txt",
        augment=False,
        shuffle=False,
    )
    logger.info("fold %s train samples=%s val samples=%s", fold, len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(cfg, device, load_pretrained=True)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=cfg.adam_eps,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum))
    total_steps = optimizer_steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, cfg.warmup_steps, total_steps),
    )
    use_amp = device.type == "cuda" and cfg.mixed_precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    csv_path = log_dir / "train_history.csv"
    best_dice = -1.0
    best_epoch = -1
    patience_anchor = -1.0
    epochs_without_improvement = 0

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_seg_loss",
                "train_aux_loss",
                "train_locator_loss",
                "val_loss",
                "val_dice",
                "val_iou",
                "val_precision",
                "val_recall",
                "val_specificity",
                "lr",
                "grad_norm",
            ],
        )
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_values = []
            train_seg_values = []
            train_aux_values = []
            train_locator_values = []
            last_grad_norm = 0.0
            epoch_start = time.time()

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = tensor_to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(batch["bmode"], batch["ceus"], return_aux=True)
                    label = batch["label"].long()
                    seg_loss = segmentation_loss(out["logits"], label, cfg.num_classes)
                    if cfg.foreground_tversky_weight > 0:
                        seg_loss = seg_loss + cfg.foreground_tversky_weight * foreground_tversky_loss(
                            out["logits"],
                            label,
                            alpha=cfg.foreground_tversky_alpha,
                            beta=cfg.foreground_tversky_beta,
                        )
                    aux_loss = lgd_auxiliary_loss(out["aux_logits"], label)
                    locator_loss = lgd_localization_loss(
                        out["locator_logits"],
                        label,
                        cfg.locator_dilate_radius,
                    )
                    loss = seg_loss + cfg.lgd_aux_weight * aux_loss + cfg.lgd_locator_weight * locator_loss
                    scaled_loss = loss / max(1, cfg.grad_accum)

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                train_loss_values.append(float(loss.detach().cpu()))
                train_seg_values.append(float(seg_loss.detach().cpu()))
                train_aux_values.append(float(aux_loss.detach().cpu()))
                train_locator_values.append(float(locator_loss.detach().cpu()))

                should_step = batch_idx % cfg.grad_accum == 0 or batch_idx == len(train_loader)
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    last_grad_norm = float(grad_norm.detach().cpu())
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if cfg.log_interval > 0 and (batch_idx % cfg.log_interval == 0 or batch_idx == len(train_loader)):
                    logger.info(
                        "fold %s epoch %03d batch %04d/%04d loss=%.6f seg=%.6f aux=%.6f locator=%.6f",
                        fold,
                        epoch,
                        batch_idx,
                        len(train_loader),
                        float(loss.detach().cpu()),
                        float(seg_loss.detach().cpu()),
                        float(aux_loss.detach().cpu()),
                        float(locator_loss.detach().cpu()),
                    )

            metrics = validation_metrics(model, val_loader, cfg, device)
            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_loss_values)),
                "train_seg_loss": float(np.mean(train_seg_values)),
                "train_aux_loss": float(np.mean(train_aux_values)),
                "train_locator_loss": float(np.mean(train_locator_values)),
                **metrics,
                "lr": current_lr,
                "grad_norm": last_grad_norm,
            }
            writer.writerow(row)
            handle.flush()
            logger.info(
                "fold %s epoch %03d/%03d train_loss=%.6f seg=%.6f aux=%.6f "
                "locator=%.6f val_dice=%.6f val_iou=%.6f lr=%.8f grad=%.4f time=%.1fs",
                fold,
                epoch,
                cfg.epochs,
                row["train_loss"],
                row["train_seg_loss"],
                row["train_aux_loss"],
                row["train_locator_loss"],
                metrics["val_dice"],
                metrics["val_iou"],
                current_lr,
                last_grad_norm,
                time.time() - epoch_start,
            )

            if metrics["val_dice"] > best_dice:
                best_dice = metrics["val_dice"]
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, epoch, best_dice, cfg)

            if metrics["val_dice"] > patience_anchor + cfg.min_delta:
                patience_anchor = metrics["val_dice"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch >= cfg.min_epochs and epochs_without_improvement >= cfg.patience:
                logger.info(
                    "fold %s early stopping at epoch %s, best_epoch=%s best_val_dice=%.6f",
                    fold,
                    epoch,
                    best_epoch,
                    best_dice,
                )
                break

    write_json(
        checkpoint_dir / "checkpoint_paths.json",
        {
            "fold": fold,
            "best_checkpoint": str(best_path),
            "best_epoch": best_epoch,
            "best_val_dice": best_dice,
            "monitor": "val_dice",
        },
    )
    return best_path


def load_model_for_inference(cfg: ExperimentConfig, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = build_model(cfg, device, load_pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def save_prediction_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def predict_test_fold(
    cfg: ExperimentConfig,
    fold: int,
    checkpoint_path: Path,
    logger: logging.Logger,
) -> Path:
    device = device_from_config(cfg)
    fold_name = f"fold_{fold}"
    pred_mask_dir = cfg.output_dir / "outputs" / fold_name / "pred_masks"
    metrics_dir = cfg.output_dir / "metrics" / fold_name
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_list = cfg.split_root / fold_name / "test_files.txt"
    test_loader = make_loader(cfg, test_list, augment=False, shuffle=False)
    model = load_model_for_inference(cfg, checkpoint_path, device)
    threshold = float(cfg.threshold)
    logger.info(
        "fold %s test samples=%s checkpoint=%s threshold=%.4f",
        fold,
        len(test_loader.dataset),
        checkpoint_path,
        threshold,
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = tensor_to_device(batch, device)
            prob = torch.softmax(model(batch["bmode"], batch["ceus"]).float(), dim=1)[:, 1:2]
            for item_idx in range(prob.shape[0]):
                original_size = tuple(int(x) for x in batch["original_size"][item_idx].detach().cpu().tolist())
                prob_resized = F.interpolate(
                    prob[item_idx : item_idx + 1],
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
                prob_np = prob_resized[0, 0].detach().cpu().numpy().astype(np.float32)
                file_name = batch["file_name"][item_idx]
                save_prediction_png(prob_np >= threshold, pred_mask_dir / file_name)

    write_json(
        cfg.output_dir / "logs" / fold_name / "inference_summary.json",
        {
            "fold": fold,
            "checkpoint": str(checkpoint_path),
            "num_test_images": len(test_loader.dataset),
            "operating_threshold": threshold,
            "pred_mask_dir": str(pred_mask_dir),
        },
    )
    logger.info("fold %s inference completed", fold)
    return evaluate_prediction_dir(
        gt_dir=cfg.dataset_root,
        pred_dir=pred_mask_dir,
        file_list=test_list,
        output_dir=metrics_dir,
        run_name=f"fold_{fold}_test",
        ruler_csv=cfg.ruler_scale_csv,
    )

