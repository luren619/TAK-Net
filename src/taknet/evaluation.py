from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_file_list, resolve_sample_path
from .metrics import MM_KEYS, compute_binary_metrics, mean_std
from .ruler import load_ruler_scale


def load_prediction_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def load_ground_truth(
    path: Path,
    foreground_value: int = 255,
    ignore_value: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(Image.open(path).convert("L"))
    return array == foreground_value, array != ignore_value


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image = image.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(image) > 0


def evaluate_prediction_dir(
    gt_dir: Path,
    pred_dir: Path,
    file_list: Path,
    output_dir: Path,
    run_name: str,
    ruler_csv: str | Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = load_ruler_scale(ruler_csv) if ruler_csv and Path(ruler_csv).is_file() else {}
    rows = []
    global_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    n_calibrated = 0
    file_names = read_file_list(file_list)
    if not file_names:
        raise ValueError(f"File list is empty: {file_list}")
    for name in file_names:
        gt_path = resolve_sample_path(gt_dir, "masks", name)
        gt, valid = load_ground_truth(gt_path)
        pred = resize_like(load_prediction_mask(pred_dir / name), gt.shape)
        px_per_cm = scale.get(name)
        if px_per_cm:
            n_calibrated += 1
        metrics = compute_binary_metrics(pred, gt, valid, px_per_cm=px_per_cm)
        for key in global_counts:
            global_counts[key] += int(metrics[key])
        rows.append({"file_name": name, **metrics})

    per_image_csv = output_dir / f"{run_name}_per_image_metrics.csv"
    with per_image_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tp, fp, fn, tn = (global_counts[key] for key in ("tp", "fp", "fn", "tn"))
    eps = 1e-8
    summary = {
        "n": len(rows),
        "n_calibrated": n_calibrated,
        "ruler_csv": str(ruler_csv) if scale else None,
        "global": {
            "dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
            "iou": float(tp / (tp + fp + fn + eps)),
            "precision": float(tp / (tp + fp + eps)),
            "recall": float(tp / (tp + fn + eps)),
            "specificity": float(tn / (tn + fp + eps)),
            "f1": float((2 * tp) / (2 * tp + fp + fn + eps)),
        },
        "per_image": {
            metric: mean_std([float(row[metric]) for row in rows])
            for metric in [
                "dice",
                "iou",
                "precision",
                "recall",
                "specificity",
                "f1",
                "hd95_pixels",
                "assd_pixels",
                "wt_gt_mean_pixels",
                "wt_pred_mean_pixels",
                "wt_gt_median_pixels",
                "wt_pred_median_pixels",
                "wt_mean_error_pixels",
                "wt_abs_mean_difference_pixels",
                "wt_profile_points",
                "wt_profile_match_fraction",
                "wt_profile_bias_pixels",
                "wt_profile_mae_pixels",
                "wt_profile_p95_abs_error_pixels",
                "wt_upper_gt_mean_pixels",
                "wt_upper_pred_mean_pixels",
                "wt_upper_mean_error_pixels",
                "wt_upper_abs_mean_difference_pixels",
                "wt_upper_profile_match_fraction",
                "wt_upper_profile_mae_pixels",
                "wt_lower_gt_mean_pixels",
                "wt_lower_pred_mean_pixels",
                "wt_lower_mean_error_pixels",
                "wt_lower_abs_mean_difference_pixels",
                "wt_lower_profile_match_fraction",
                "wt_lower_profile_mae_pixels",
                *MM_KEYS,
            ]
        },
        "files": {"per_image_metrics_csv": str(per_image_csv)},
    }
    output_json = output_dir / f"{run_name}_metrics.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_json
