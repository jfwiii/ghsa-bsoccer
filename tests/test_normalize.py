"""Tests for normalize.py — M3 acceptance and duration helpers."""

import pytest
from datetime import date

from pipeline.normalize import (
    impute_duration_minutes,
    observed_duration_minutes,
    Normalizer,
)


# ---------------------------------------------------------------------------
# impute_duration_minutes — §7.4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mov,expected", [
    (0, 80), (1, 80), (3, 80), (6, 80),
    (7, 70),
    (8, 65),
    (9, 60),
    (10, 50),
    (15, 50),
])
def test_impute_duration(mov, expected):
    assert impute_duration_minutes(mov) == expected


# ---------------------------------------------------------------------------
# observed_duration_minutes — §7.5
# ---------------------------------------------------------------------------

def test_observed_normal_game():
    # h1 MOV < 7 → 80 min
    assert observed_duration_minutes(2, 1, 4, 2) == 80


def test_observed_shortened_second_half():
    # h1 MOV = 7 → 60 min
    assert observed_duration_minutes(7, 0, 9, 1) == 60


def test_observed_terminated_at_halftime():
    # h1 MOV >= 10 → 40 min
    assert observed_duration_minutes(10, 0, 10, 0) == 40


def test_observed_exact_boundary():
    assert observed_duration_minutes(6, 0, 8, 1) == 80   # 6 → not shortened
    assert observed_duration_minutes(7, 0, 9, 2) == 60   # 7 → shortened
    assert observed_duration_minutes(9, 0, 9, 0) == 60   # 9 → shortened
    assert observed_duration_minutes(10, 0, 10, 0) == 40  # 10 → terminated


# ---------------------------------------------------------------------------
# Normalizer deduplication
# ---------------------------------------------------------------------------

def _make_game(date, home_id, away_id, hg, ag):
    from pipeline.normalize import _game_id
    return {
        "game_id": _game_id(date, home_id, away_id),
        "date": date,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_goals": hg,
        "away_goals": ag,
        "reporting_team_id": home_id,
        "opponent_name_raw": "Opponent",
        "opponent_team_id_raw": away_id,
        "is_home": True,
        "neutral_site": False,
        "is_non_ghsa": False,
    }


def test_dedup_identical_games():
    """Same game reported from both team pages should appear once."""
    d = date(2026, 2, 1)
    teams = [
        {"team_id": 1, "name": "Alpha", "class": "AAAAAA", "region_or_area": "1-AAAAAA"},
        {"team_id": 2, "name": "Beta", "class": "AAAAAA", "region_or_area": "1-AAAAAA"},
    ]
    raw_games = [
        _make_game(d, 1, 2, 3, 1),
        _make_game(d, 1, 2, 3, 1),  # duplicate
    ]
    n = Normalizer(teams, raw_games, no_maxpreps=True)
    _, games_df = n.run()
    assert len(games_df) == 1


def test_duration_used_never_null():
    """duration_used_min must always be populated."""
    d = date(2026, 2, 1)
    teams = [
        {"team_id": 10, "name": "A", "class": "AAA"},
        {"team_id": 20, "name": "B", "class": "AAA"},
    ]
    raw_games = [_make_game(d, 10, 20, 5, 0)]
    n = Normalizer(teams, raw_games, no_maxpreps=True)
    _, games_df = n.run()
    assert games_df["duration_used_min"].notna().all()


def test_regulation_equals_raw_without_maxpreps():
    """Without MaxPreps, regulation scores should equal GHSA raw scores."""
    d = date(2026, 2, 1)
    teams = [
        {"team_id": 1, "name": "A", "class": "AAA"},
        {"team_id": 2, "name": "B", "class": "AAA"},
    ]
    raw_games = [_make_game(d, 1, 2, 3, 2)]
    n = Normalizer(teams, raw_games, no_maxpreps=True)
    _, games_df = n.run()
    row = games_df.iloc[0]
    assert row["home_goals_regulation"] == 3
    assert row["away_goals_regulation"] == 2
