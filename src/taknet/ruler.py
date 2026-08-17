from __future__ import annotations

import csv
from pathlib import Path


def load_ruler_scale(csv_path: str | Path) -> dict[str, float]:
    """Map image basename -> px_per_cm from a ruler-scale CSV.

    Supports the v3 schema (``relative_path``, ``px_per_cm``) and older schemas that
    use ``file``. When both ``imgs`` and ``ceus`` rows share a basename, the ``imgs``
    row is preferred (their physical scale is identical, this only makes the source
    deterministic).
    """
    path = Path(csv_path)
    scale: dict[str, float] = {}
    preferred: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        path_key = "relative_path" if "relative_path" in fields else "file"
        modal_key = "modality" if "modality" in fields else ("modal" if "modal" in fields else None)
        for row in reader:
            name = Path(row.get(path_key, "")).name
            if not name:
                continue
            try:
                px = float(row["px_per_cm"])
            except (KeyError, TypeError, ValueError):
                continue
            if px <= 0:
                continue
            is_imgs = modal_key is not None and row.get(modal_key, "") == "imgs"
            if name in preferred and not is_imgs:
                continue
            scale[name] = px
            if is_imgs:
                preferred.add(name)
    return scale
