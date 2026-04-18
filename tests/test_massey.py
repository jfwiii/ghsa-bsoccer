"""Tests for massey.py — log-MOV transform + synthetic rating recovery."""

import numpy as np
import pytest
import pandas as pd
from datetime import date

from model.massey import log_mov_transform, fit


# ---------------------------------------------------------------------------
# log_mov_transform — §15
# ---------------------------------------------------------------------------

def test_log_mov_zero():
    assert log_mov_transform(0) == 0.0


def test_log_mov_one():
    assert abs(log_mov_transform(1) - np.log(2)) < 1e-9


def test_log_mov_negative_five():
    assert abs(log_mov_transform(-5) - (-np.log(6))) < 1e-9


def test_log_mov_symmetry():
    for d in [1, 2, 3, 5, 7, 10]:
        assert abs(log_mov_transform(d) + log_mov_transform(-d)) < 1e-9


def test_log_mov_monotone():
    vals = [log_mov_transform(d) for d in range(0, 12)]
    for a, b in zip(vals, vals[1:]):
        assert b > a


# ---------------------------------------------------------------------------
# Synthetic rating recovery
# ---------------------------------------------------------------------------

def _make_synthetic_games(n_teams: int = 6, n_games: int = 60,
                           seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    """Generate synthetic game log with known team ratings."""
    rng = np.random.default_rng(seed)
    true_ratings = np.linspace(-1.5, 1.5, n_teams)
    teams = [{"team_id": i + 1, "name": f"Team{i+1}", "class": "AAA",
              "region_or_area": "1-AAA"} for i in range(n_teams)]

    rows = []
    for _ in range(n_games):
        i, j = rng.choice(n_teams, 2, replace=False)
        strength_diff = true_ratings[i] - true_ratings[j] + 0.15  # home adv
        d = int(np.round(np.clip(rng.normal(strength_diff * 2, 1.5), -8, 8)))
        hg = max(d, 0)
        ag = max(-d, 0)
        rows.append({
            "game_id": f"g{_}",
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(_)),
            "home_team_id": i + 1,
            "away_team_id": j + 1,
            "home_goals": hg,
            "away_goals": ag,
            "home_goals_regulation": hg,
            "away_goals_regulation": ag,
            "neutral_site": False,
            "is_non_ghsa": False,
            "duration_used_min": 80,
        })

    games_df = pd.DataFrame(rows)
    return games_df, teams


def test_massey_top_team_correct():
    """Best synthetic team should rank first."""
    games_df, teams = _make_synthetic_games(n_teams=6, n_games=120)
    teams_df = pd.DataFrame(teams)
    result = fit(games_df, teams_df)
    ratings = result["ratings"]
    # Team 6 has rating 1.5 (highest) → should rank first
    top = max(ratings, key=ratings.get)
    assert top == 6, f"Expected team 6 to be top, got {top}"


def test_massey_rating_sum_zero():
    """Identifiability: sum of ratings should be ≈ 0."""
    games_df, teams = _make_synthetic_games()
    teams_df = pd.DataFrame(teams)
    result = fit(games_df, teams_df)
    total = sum(result["ratings"].values())
    assert abs(total) < 0.1


def test_massey_home_advantage_positive():
    """Home advantage should be positive."""
    games_df, teams = _make_synthetic_games()
    teams_df = pd.DataFrame(teams)
    result = fit(games_df, teams_df)
    assert result["home_advantage"] >= 0
