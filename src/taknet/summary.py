from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def summarize_metric_files(metric_json: list[Path], output_json: Path) -> Path:
    records = []
    for fold_index, path in enumerate(metric_json, start=1):
        data = json.loads(path.read_text(encoding="utf-8"))
        record = {"fold_index": fold_index, "path": str(path)}
        for key, value in data.get("global", {}).items():
            record[f"global_{key}"] = value
        for key, value in data.get("per_image", {}).items():
            record[f"mean_{key}"] = value.get("mean")
        records.append(record)

    metric_keys = [key for key in records[0] if key not in {"fold_index", "path"}]
    summary = {"folds": records, "summary": {}}
    for key in metric_keys:
        values = np.asarray([float(record[key]) for record in records], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary["summary"][key] = {"mean": float("nan"), "std": float("nan")}
            continue
        summary["summary"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_json.with_suffix(".csv")
    summary["files"] = {"summary_csv": str(output_csv)}
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "fold_index", "path", "metric", "value", "mean", "std"],
        )
        writer.writeheader()
        for record in records:
            for key in metric_keys:
                writer.writerow(
                    {
                        "scope": "fold",
                        "fold_index": record["fold_index"],
                        "path": record["path"],
                        "metric": key,
                        "value": record[key],
                        "mean": "",
                        "std": "",
                    }
                )
        for key, value in summary["summary"].items():
            writer.writerow(
                {
                    "scope": "summary",
                    "fold_index": "",
                    "path": "",
                    "metric": key,
                    "value": "",
                    "mean": value["mean"],
                    "std": value["std"],
                }
            )
    return output_json
