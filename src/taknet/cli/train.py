from __future__ import annotations

import argparse
from pathlib import Path

from taknet.config import ExperimentConfig
from taknet.engine import predict_test_fold, train_one_fold
from taknet.logging_utils import metric_complete, setup_logger
from taknet.summary import summarize_metric_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate TAK-Net.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    updates = {}
    if args.folds is not None:
        updates["folds"] = args.folds
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.device is not None:
        updates["device"] = args.device
    cfg = cfg.with_updates(**updates)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg.output_dir / "logs" / "five_fold_run.log")
    logger.info("config=%s", args.config)
    logger.info("output_dir=%s", cfg.output_dir)
    logger.info("folds=%s", cfg.folds)

    metric_paths = []
    for fold in cfg.folds:
        fold_name = f"fold_{fold}"
        fold_logger = setup_logger(cfg.output_dir / "logs" / fold_name / "run.log")
        metric_path = cfg.output_dir / "metrics" / fold_name / f"fold_{fold}_test_metrics.json"
        test_list = cfg.split_root / fold_name / "test_files.txt"
        if args.skip_existing and metric_complete(metric_path, test_list):
            fold_logger.info("fold %s already has complete metrics: %s", fold, metric_path)
            metric_paths.append(metric_path)
            continue
        if not args.evaluate_only:
            train_one_fold(cfg, fold, fold_logger)
        checkpoint_path = cfg.output_dir / "checkpoints" / fold_name / "best_model.pth"
        metric_paths.append(predict_test_fold(cfg, fold, checkpoint_path, fold_logger))

    if len(metric_paths) > 1:
        summary_path = summarize_metric_files(metric_paths, cfg.output_dir / "metrics" / "test_5fold_summary.json")
        logger.info("summary=%s", summary_path)


if __name__ == "__main__":
    main()
