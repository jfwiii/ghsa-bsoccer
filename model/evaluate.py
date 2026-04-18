"""
Model evaluation — M5 acceptance checks.

Metrics:
- Held-out log loss (W/L) on most-recent 15% of games.
- Brier score.
- Calibration by decile.
- Ablation comparisons (with/without duration offset, regulation correction, MaxPreps).
- Baselines: PSR rank, pure Massey, MaxPreps state rank, naive 50/50.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model import dixon_coles, massey

log = logging.getLogger(__name__)

DATA_DIR = Path("data")


def _log_loss_binary(y_true: np.ndarray, p_pred: np.ndarray,
                     eps: float = 1e-7) -> float:
    p = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((y_true - p_pred) ** 2))


def split_holdout(games_df: pd.DataFrame, holdout_frac: float = 0.15):
    """Time-ordered split: last holdout_frac of games held out."""
    df_sorted = games_df.sort_values("date").reset_index(drop=True)
    n_hold = max(1, int(len(df_sorted) * holdout_frac))
    train = df_sorted.iloc[:-n_hold]
    test = df_sorted.iloc[-n_hold:]
    return train, test


def evaluate(dc_result: dict, test_df: pd.DataFrame,
             massey_result: Optional[dict] = None,
             teams_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Evaluate Dixon-Coles predictions on held-out games.

    Returns eval report dict.
    """
    if test_df.empty:
        log.warning("evaluate: empty test set")
        return {}

    preds = []
    for _, row in test_df.iterrows():
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        neutral = bool(row.get("neutral_site", False))
        mp = dixon_coles.predict_matchup(h, a, neutral, dc_result)

        actual_home_win = int(row["home_goals_regulation"]) > int(row["away_goals_regulation"])
        preds.append({
            "p_home_win": mp["p_home_win"],
            "actual_home_win": int(actual_home_win),
        })

    pred_df = pd.DataFrame(preds)
    y = pred_df["actual_home_win"].values.astype(float)
    p = pred_df["p_home_win"].values

    dc_ll = _log_loss_binary(y, p)
    dc_brier = _brier_score(y, p)

    # Naive baseline
    naive_ll = _log_loss_binary(y, np.full_like(y, 0.5))
    naive_brier = _brier_score(y, np.full_like(y, 0.5))

    # Massey baseline (if provided)
    massey_ll = None
    massey_brier = None
    if massey_result and massey_result.get("ratings"):
        massey_preds = []
        for _, row in test_df.iterrows():
            h, a = int(row["home_team_id"]), int(row["away_team_id"])
            rh = massey_result["ratings"].get(h, 0.0)
            ra = massey_result["ratings"].get(a, 0.0)
            ha = massey_result.get("home_advantage", 0.0) if not row.get("neutral_site") else 0.0
            diff = rh - ra + ha
            # Sigmoid to get probability
            p_m = 1.0 / (1.0 + np.exp(-diff))
            massey_preds.append(p_m)
        pm_arr = np.array(massey_preds)
        massey_ll = _log_loss_binary(y, pm_arr)
        massey_brier = _brier_score(y, pm_arr)

    # Calibration by decile
    decile_bins = np.percentile(p, np.arange(0, 110, 10))
    calibration = []
    for i in range(len(decile_bins) - 1):
        mask = (p >= decile_bins[i]) & (p < decile_bins[i + 1])
        if mask.sum() > 0:
            calibration.append({
                "decile": i + 1,
                "mean_predicted": float(p[mask].mean()),
                "mean_actual": float(y[mask].mean()),
                "n": int(mask.sum()),
            })

    report = {
        "n_test_games": len(test_df),
        "dixon_coles": {
            "log_loss": round(dc_ll, 5),
            "brier_score": round(dc_brier, 5),
        },
        "naive_50_50": {
            "log_loss": round(naive_ll, 5),
            "brier_score": round(naive_brier, 5),
        },
        "massey": {
            "log_loss": round(massey_ll, 5) if massey_ll else None,
            "brier_score": round(massey_brier, 5) if massey_brier else None,
        },
        "calibration_by_decile": calibration,
        "improvements_vs_naive_pct": {
            "log_loss": round((naive_ll - dc_ll) / naive_ll * 100, 2) if naive_ll else None,
        },
    }

    if massey_ll:
        report["improvements_vs_massey_pct"] = {
            "log_loss": round((massey_ll - dc_ll) / massey_ll * 100, 2),
        }

    log.info(
        "Evaluation: DC log_loss=%.4f, Massey log_loss=%s, Naive log_loss=%.4f",
        dc_ll, f"{massey_ll:.4f}" if massey_ll else "N/A", naive_ll
    )

    return report


def save_eval_report(report: dict) -> None:
    path = DATA_DIR / "eval_report.json"
    path.write_text(json.dumps(report, indent=2))
    log.info("saved eval_report.json")
