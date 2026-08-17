from __future__ import annotations

import argparse
import json
from pathlib import Path

from taknet.evaluation import evaluate_prediction_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation masks.")
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="binary_eval")
    parser.add_argument("--ruler-csv", type=Path, default=None)
    args = parser.parse_args()
    output_json = evaluate_prediction_dir(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        file_list=args.file_list,
        output_dir=args.output_dir,
        run_name=args.run_name,
        ruler_csv=args.ruler_csv,
    )
    print(json.dumps(json.loads(output_json.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
