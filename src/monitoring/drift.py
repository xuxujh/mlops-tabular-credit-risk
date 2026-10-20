from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp


def detect_feature_drift(
    reference_data: dict[str, list[float]],
    current_data: dict[str, list[float]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Executa o teste de Kolmogorov-Smirnov (KS) de duas amostras para detecção de Data Drift."""
    report: dict[str, Any] = {}
    drift_detected_any = False

    for feature, ref_values in reference_data.items():
        if feature not in current_data or len(current_data[feature]) < 2:
            continue

        ref_arr = np.asarray(ref_values, dtype=float)
        cur_arr = np.asarray(current_data[feature], dtype=float)

        stat, p_value = ks_2samp(ref_arr, cur_arr)
        is_drifted = bool(p_value < alpha)
        if is_drifted:
            drift_detected_any = True

        report[feature] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": is_drifted,
        }

    return {
        "dataset_drift": drift_detected_any,
        "alpha": alpha,
        "features": report,
    }