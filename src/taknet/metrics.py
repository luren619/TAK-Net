from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.spatial import cKDTree
from skimage.morphology import medial_axis


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask ^ binary_erosion(mask)


def boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    diagonal = float(math.hypot(gt.shape[0], gt.shape[1]))
    if not pred.any() or not gt.any():
        return diagonal, diagonal
    pred_surface = surface(pred)
    gt_surface = surface(gt)
    if not pred_surface.any() or not gt_surface.any():
        return diagonal, diagonal
    pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(distances, 95)), float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0)


def main_wall_components(
    mask: np.ndarray,
    min_component_area: int = 100,
    max_components: int = 2,
) -> list[dict]:
    """Return the largest wall components, ordered from upper to lower.

    The masks normally contain two long wall bands. Small isolated prediction
    components must not influence wall-thickness statistics.
    """
    labels, count = label(np.ascontiguousarray(mask.astype(bool)))
    components = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        ys, _ = np.where(component)
        area = int(ys.size)
        if area < int(min_component_area):
            continue
        components.append(
            {
                "mask": component,
                "area": area,
                "centroid_y": float(ys.mean()),
            }
        )
    components = sorted(components, key=lambda item: item["area"], reverse=True)[:max_components]
    return sorted(components, key=lambda item: item["centroid_y"])


def medial_axis_profile(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return medial-axis coordinates and local-thickness diameters in pixels.

    At each medial-axis point the value is ``2 * EDT``, i.e. the diameter of
    the maximal inscribed circle. This is direction-independent and is a useful
    local wall-thickness estimator for smooth bands, but it is not identical to
    an explicitly paired LI-to-MA normal distance at endpoints or branches.

    A one-pixel background pad makes image-border handling symmetric. ``rng=0``
    removes the random tie-breaking otherwise used by scikit-image.
    """
    mask = np.ascontiguousarray(mask.astype(bool))
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    skeleton, distance = medial_axis(padded, return_distance=True, rng=0)
    coords = np.argwhere(skeleton).astype(np.float64) - 1.0
    thickness = (2.0 * distance[skeleton]).astype(np.float64)
    inside = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < mask.shape[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < mask.shape[1])
    )
    return coords[inside], thickness[inside]


def match_local_thickness_profiles(
    gt_coords: np.ndarray,
    gt_thickness: np.ndarray,
    pred_coords: np.ndarray,
    pred_thickness: np.ndarray,
) -> dict[str, float]:
    """Compare local thickness at spatially corresponding medial-axis points.

    GT medial points are matched to the nearest predicted medial point from the
    corresponding upper/lower wall component. Matches farther than one median
    GT thickness are rejected and reported through ``match_fraction``.
    """
    if not gt_thickness.size:
        return {
            "points": 0,
            "match_fraction": float("nan"),
            "bias_pixels": float("nan"),
            "mae_pixels": float("nan"),
            "p95_abs_error_pixels": float("nan"),
        }
    if not pred_thickness.size:
        return {
            "points": 0,
            "match_fraction": 0.0,
            "bias_pixels": float("nan"),
            "mae_pixels": float("nan"),
            "p95_abs_error_pixels": float("nan"),
        }
    distances, indices = cKDTree(pred_coords).query(gt_coords, k=1)
    max_distance = max(3.0, float(np.median(gt_thickness)))
    matched = distances <= max_distance
    if not matched.any():
        return {
            "points": 0,
            "match_fraction": 0.0,
            "bias_pixels": float("nan"),
            "mae_pixels": float("nan"),
            "p95_abs_error_pixels": float("nan"),
        }
    error = pred_thickness[indices[matched]] - gt_thickness[matched]
    return {
        "points": int(matched.sum()),
        "match_fraction": float(matched.mean()),
        "bias_pixels": float(error.mean()),
        "mae_pixels": float(np.abs(error).mean()),
        "p95_abs_error_pixels": float(np.percentile(np.abs(error), 95)),
    }


def _empty_profile() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)


def _inside_mask(mask: np.ndarray, point_yx: np.ndarray) -> bool:
    y, x = np.rint(point_yx).astype(int)
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _ray_to_boundary(
    mask: np.ndarray,
    origin_yx: np.ndarray,
    direction_yx: np.ndarray,
    step: float = 0.25,
    max_distance: float = 80.0,
) -> tuple[float, np.ndarray] | None:
    last_inside = origin_yx.copy()
    distance = 0.0
    while distance <= max_distance:
        distance += step
        point = origin_yx + distance * direction_yx
        if not _inside_mask(mask, point):
            return max(0.0, distance - 0.5 * step), last_inside
        last_inside = point
    return None


def _empty_contour_profile() -> dict[str, np.ndarray]:
    coords = np.zeros((0, 2), dtype=np.float64)
    return {
        "coords": coords,
        "thickness": np.zeros(0, dtype=np.float64),
        "inner": coords.copy(),
        "outer": coords.copy(),
    }


def _refine_li_to_ma_profile(
    component_mask: np.ndarray,
    inner_points: np.ndarray,
    approximate_outer_points: np.ndarray,
    tangent_radius: float = 12.0,
    min_linearity: float = 3.0,
) -> dict[str, np.ndarray]:
    """Measure from the lumen-facing (LI) contour along its normal to the outer (MA) contour."""
    if inner_points.shape[0] < 5:
        return _empty_contour_profile()
    tree = cKDTree(inner_points)
    kept_inner = []
    outer_points = []
    thickness = []
    for index, inner in enumerate(inner_points):
        neighbors = tree.query_ball_point(inner, r=tangent_radius)
        if len(neighbors) < 5:
            continue
        neighborhood = inner_points[neighbors] - inner_points[neighbors].mean(axis=0, keepdims=True)
        covariance = neighborhood.T @ neighborhood / max(1, len(neighbors) - 1)
        values, vectors = np.linalg.eigh(covariance)
        linearity = float(values[-1] / max(values[0], 1e-6))
        if linearity < min_linearity:
            continue
        tangent = vectors[:, -1]
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-8)
        if float(np.dot(normal, approximate_outer_points[index] - inner)) < 0:
            normal = -normal
        crossing = _ray_to_boundary(component_mask, inner, normal)
        if crossing is None:
            continue
        width, outer = crossing
        if width <= 1.0:
            continue
        kept_inner.append(inner)
        outer_points.append(outer)
        thickness.append(width)
    if not kept_inner:
        return _empty_contour_profile()
    inner_array = np.asarray(kept_inner)
    return {
        "coords": inner_array,
        "thickness": np.asarray(thickness),
        "inner": inner_array,
        "outer": np.asarray(outer_points),
    }


def li_ma_contour_profile(
    component_mask: np.ndarray,
    opposite_wall_mask: np.ndarray | None,
    tangent_radius: float = 12.0,
    min_linearity: float = 3.0,
) -> dict[str, np.ndarray]:
    """Infer LI/MA contour sides, then measure LI-to-MA distance along the LI normal.

    Initial medial-axis normals are used only to locate the two contour sides. The
    side closer to the opposite wall is identified as lumen-facing LI. Final wall
    thickness is measured from LI along the normal of the LI contour itself until
    the outer MA contour is reached.
    """
    coords, _ = medial_axis_profile(component_mask)
    if coords.shape[0] < 5:
        return _empty_contour_profile()
    skeleton_tree = cKDTree(coords)
    opposite_tree = None
    if opposite_wall_mask is not None and opposite_wall_mask.any():
        opposite_tree = cKDTree(np.argwhere(opposite_wall_mask))
    inner_points = []
    outer_points = []
    for origin in coords:
        neighbors = skeleton_tree.query_ball_point(origin, r=tangent_radius)
        if len(neighbors) < 5:
            continue
        neighborhood = coords[neighbors] - coords[neighbors].mean(axis=0, keepdims=True)
        covariance = neighborhood.T @ neighborhood / max(1, len(neighbors) - 1)
        values, vectors = np.linalg.eigh(covariance)
        linearity = float(values[-1] / max(values[0], 1e-6))
        if linearity < min_linearity:
            continue
        tangent = vectors[:, -1]
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-8)
        positive = _ray_to_boundary(component_mask, origin, normal)
        negative = _ray_to_boundary(component_mask, origin, -normal)
        if positive is None or negative is None:
            continue
        _, positive_point = positive
        _, negative_point = negative
        if opposite_tree is None:
            inner, outer = positive_point, negative_point
        else:
            positive_gap = float(opposite_tree.query(positive_point, k=1)[0])
            negative_gap = float(opposite_tree.query(negative_point, k=1)[0])
            inner, outer = (
                (positive_point, negative_point)
                if positive_gap <= negative_gap
                else (negative_point, positive_point)
            )
        inner_points.append(inner)
        outer_points.append(outer)
    if not inner_points:
        return _empty_contour_profile()
    return _refine_li_to_ma_profile(
        component_mask,
        np.asarray(inner_points),
        np.asarray(outer_points),
        tangent_radius=tangent_radius,
        min_linearity=min_linearity,
    )


def _component_profiles(mask: np.ndarray) -> list[dict]:
    components = main_wall_components(mask)
    profiles = []
    for index, component in enumerate(components):
        opposite = components[1 - index]["mask"] if len(components) == 2 else None
        profile = li_ma_contour_profile(component["mask"], opposite)
        profiles.append({**component, **profile})
    return profiles


def _pair_components(gt_profiles: list[dict], pred_profiles: list[dict]) -> list[tuple[dict, dict | None]]:
    """Pair predicted walls to GT walls by vertical centroid without reuse."""
    remaining = list(pred_profiles)
    pairs = []
    for gt_profile in gt_profiles:
        if not remaining:
            pairs.append((gt_profile, None))
            continue
        index = min(
            range(len(remaining)),
            key=lambda idx: abs(remaining[idx]["centroid_y"] - gt_profile["centroid_y"]),
        )
        pairs.append((gt_profile, remaining.pop(index)))
    return pairs


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def wall_thickness_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Component-aware LI-to-MA contour-normal wall thickness and spatial MAE."""
    gt_profiles = _component_profiles(gt)
    pred_profiles = _component_profiles(pred)
    gt_lt = np.concatenate([item["thickness"] for item in gt_profiles]) if gt_profiles else np.zeros(0)
    pred_lt = np.concatenate([item["thickness"] for item in pred_profiles]) if pred_profiles else np.zeros(0)
    gt_mean = _mean_or_nan(gt_lt)
    pred_mean = _mean_or_nan(pred_lt)
    error = pred_mean - gt_mean
    result = {
        "wt_gt_mean_pixels": gt_mean,
        "wt_pred_mean_pixels": pred_mean,
        "wt_gt_median_pixels": _median_or_nan(gt_lt),
        "wt_pred_median_pixels": _median_or_nan(pred_lt),
        "wt_mean_error_pixels": error,
        "wt_abs_mean_difference_pixels": abs(error),
    }

    pairs = _pair_components(gt_profiles, pred_profiles)
    all_errors = []
    matched_points = 0
    total_gt_points = int(sum(item["thickness"].size for item in gt_profiles))
    for wall_index, wall_name in enumerate(("upper", "lower")):
        if wall_index < len(pairs):
            gt_profile, pred_profile = pairs[wall_index]
            gt_coords = gt_profile["coords"]
            gt_values = gt_profile["thickness"]
            if pred_profile is None:
                pred_coords, pred_values = _empty_profile()
            else:
                pred_coords = pred_profile["coords"]
                pred_values = pred_profile["thickness"]
            matched = match_local_thickness_profiles(gt_coords, gt_values, pred_coords, pred_values)
            if matched["points"]:
                distances, indices = cKDTree(pred_coords).query(gt_coords, k=1)
                valid_match = distances <= max(3.0, float(np.median(gt_values)))
                all_errors.append(pred_values[indices[valid_match]] - gt_values[valid_match])
                matched_points += int(valid_match.sum())
        else:
            gt_values = np.zeros(0, dtype=np.float64)
            pred_values = np.zeros(0, dtype=np.float64)
            matched = match_local_thickness_profiles(*_empty_profile(), *_empty_profile())

        gt_wall_mean = _mean_or_nan(gt_values)
        pred_wall_mean = _mean_or_nan(pred_values)
        wall_error = pred_wall_mean - gt_wall_mean
        result.update(
            {
                f"wt_{wall_name}_gt_mean_pixels": gt_wall_mean,
                f"wt_{wall_name}_pred_mean_pixels": pred_wall_mean,
                f"wt_{wall_name}_mean_error_pixels": wall_error,
                f"wt_{wall_name}_abs_mean_difference_pixels": abs(wall_error),
                f"wt_{wall_name}_profile_match_fraction": matched["match_fraction"],
                f"wt_{wall_name}_profile_mae_pixels": matched["mae_pixels"],
            }
        )

    if all_errors:
        local_error = np.concatenate(all_errors)
        result.update(
            {
                "wt_profile_points": int(local_error.size),
                "wt_profile_match_fraction": float(matched_points / max(1, total_gt_points)),
                "wt_profile_bias_pixels": float(local_error.mean()),
                "wt_profile_mae_pixels": float(np.abs(local_error).mean()),
                "wt_profile_p95_abs_error_pixels": float(np.percentile(np.abs(local_error), 95)),
            }
        )
    else:
        result.update(
            {
                "wt_profile_points": 0,
                "wt_profile_match_fraction": 0.0 if total_gt_points else float("nan"),
                "wt_profile_bias_pixels": float("nan"),
                "wt_profile_mae_pixels": float("nan"),
                "wt_profile_p95_abs_error_pixels": float("nan"),
            }
        )
    return result


MM_KEYS = [
    "hd95_mm",
    "assd_mm",
    "wt_gt_mean_mm",
    "wt_pred_mean_mm",
    "wt_gt_median_mm",
    "wt_pred_median_mm",
    "wt_mean_error_mm",
    "wt_abs_mean_difference_mm",
    "wt_profile_bias_mm",
    "wt_profile_mae_mm",
    "wt_profile_p95_abs_error_mm",
    "wt_upper_gt_mean_mm",
    "wt_upper_pred_mean_mm",
    "wt_upper_mean_error_mm",
    "wt_upper_abs_mean_difference_mm",
    "wt_upper_profile_mae_mm",
    "wt_lower_gt_mean_mm",
    "wt_lower_pred_mean_mm",
    "wt_lower_mean_error_mm",
    "wt_lower_abs_mean_difference_mm",
    "wt_lower_profile_mae_mm",
]


def to_mm_metrics(px_result: dict[str, float], px_per_cm: float | None) -> dict[str, float]:
    """Convert pixel distance and wall-thickness metrics using a per-image scale."""
    if not px_per_cm or float(px_per_cm) <= 0:
        return {"px_per_cm": float("nan"), **{key: float("nan") for key in MM_KEYS}}
    mm = 10.0 / float(px_per_cm)
    return {
        "px_per_cm": float(px_per_cm),
        "hd95_mm": px_result["hd95_pixels"] * mm,
        "assd_mm": px_result["assd_pixels"] * mm,
        "wt_gt_mean_mm": px_result["wt_gt_mean_pixels"] * mm,
        "wt_pred_mean_mm": px_result["wt_pred_mean_pixels"] * mm,
        "wt_gt_median_mm": px_result["wt_gt_median_pixels"] * mm,
        "wt_pred_median_mm": px_result["wt_pred_median_pixels"] * mm,
        "wt_mean_error_mm": px_result["wt_mean_error_pixels"] * mm,
        "wt_abs_mean_difference_mm": px_result["wt_abs_mean_difference_pixels"] * mm,
        "wt_profile_bias_mm": px_result["wt_profile_bias_pixels"] * mm,
        "wt_profile_mae_mm": px_result["wt_profile_mae_pixels"] * mm,
        "wt_profile_p95_abs_error_mm": px_result["wt_profile_p95_abs_error_pixels"] * mm,
        "wt_upper_gt_mean_mm": px_result["wt_upper_gt_mean_pixels"] * mm,
        "wt_upper_pred_mean_mm": px_result["wt_upper_pred_mean_pixels"] * mm,
        "wt_upper_mean_error_mm": px_result["wt_upper_mean_error_pixels"] * mm,
        "wt_upper_abs_mean_difference_mm": px_result["wt_upper_abs_mean_difference_pixels"] * mm,
        "wt_upper_profile_mae_mm": px_result["wt_upper_profile_mae_pixels"] * mm,
        "wt_lower_gt_mean_mm": px_result["wt_lower_gt_mean_pixels"] * mm,
        "wt_lower_pred_mean_mm": px_result["wt_lower_pred_mean_pixels"] * mm,
        "wt_lower_mean_error_mm": px_result["wt_lower_mean_error_pixels"] * mm,
        "wt_lower_abs_mean_difference_mm": px_result["wt_lower_abs_mean_difference_pixels"] * mm,
        "wt_lower_profile_mae_mm": px_result["wt_lower_profile_mae_pixels"] * mm,
    }


def compute_binary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray | None = None,
    px_per_cm: float | None = None,
) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    valid = np.ones_like(gt, dtype=bool) if valid is None else valid.astype(bool)
    pred_valid = np.logical_and(pred, valid)
    gt_valid = np.logical_and(gt, valid)
    tp = int(np.logical_and(pred_valid, gt_valid).sum())
    fp = int(np.logical_and(pred_valid, np.logical_and(~gt, valid)).sum())
    fn = int(np.logical_and(np.logical_and(~pred, valid), gt_valid).sum())
    tn = int(np.logical_and(np.logical_and(~pred, valid), np.logical_and(~gt, valid)).sum())
    eps = 1e-8
    hd95, assd = boundary_metrics(pred_valid, gt_valid)
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "specificity": float(tn / (tn + fp + eps)),
        "f1": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "hd95_pixels": hd95,
        "assd_pixels": assd,
        **wall_thickness_metrics(pred_valid, gt_valid),
    }
    result.update(to_mm_metrics(result, px_per_cm))
    return result


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }
