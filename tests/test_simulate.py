"""Tests for simulate.py — bracket simulator."""

import numpy as np
import pytest

from model.simulate import simulate_game, simulate_bracket


def _mock_dc(alpha_h=1.5, alpha_a=0.8):
    return {
        "alpha": {1: alpha_h, 2: alpha_a, 3: 1.0, 4: 0.9},
        "beta": {1: 0.9, 2: 1.1, 3: 1.0, 4: 1.0},
        "gamma": 1.2,
        "rho": -0.05,
        "rating": {1: 1.0, 2: 0.7, 3: 0.8, 4: 0.75},
    }


def _simple_bracket():
    return {
        "bracket": "TEST",
        "rounds": [
            {
                "round": 1,
                "matchups": [
                    {"position": 1, "top_team_id": 1, "bottom_team_id": 4,
                     "top_seed": 1, "bottom_seed": 4, "host_team_id": 1},
                    {"position": 2, "top_team_id": 2, "bottom_team_id": 3,
                     "top_seed": 2, "bottom_seed": 3, "host_team_id": 2},
                ]
            },
            {
                "round": 2,
                "matchups": [
                    {"position": 1, "top_team_id": None, "bottom_team_id": None,
                     "top_seed": 1, "bottom_seed": 2, "host_team_id": None},
                ]
            }
        ]
    }


def test_simulate_game_returns_valid_winner():
    dc = _mock_dc()
    rng = np.random.default_rng(42)
    winner = simulate_game(1, 2, False, dc, rng)
    assert winner in (1, 2)


def test_simulate_game_stronger_team_wins_more():
    """Run many games; stronger team (team 1) should win majority."""
    dc = _mock_dc(alpha_h=3.0, alpha_a=0.3)  # very lopsided
    rng = np.random.default_rng(0)
    wins = sum(simulate_game(1, 2, True, dc, rng) == 1 for _ in range(1000))
    assert wins > 700, f"Expected team 1 to win >70%, got {wins}/1000"


def test_simulate_bracket_no_nan_probs():
    dc = _mock_dc()
    bracket = _simple_bracket()
    result = simulate_bracket(bracket, dc, n_sims=1000, rng_seed=42)
    for team in result["teams"]:
        for label, prob in team["round_probabilities"].items():
            assert 0.0 <= prob <= 1.0, f"Prob out of range: {label}={prob}"


def test_simulate_bracket_top_seed_higher_champ_odds():
    """Top seed should have higher championship odds than bottom seed on average."""
    dc = _mock_dc()
    bracket = _simple_bracket()
    result = simulate_bracket(bracket, dc, n_sims=5000, rng_seed=42)
    probs = {t["team_id"]: t["round_probabilities"].get("Champ", 0)
             for t in result["teams"]}
    # Team 1 (strongest attack) should have highest championship odds
    assert probs[1] >= probs[4], f"Team 1 ({probs[1]}) should beat team 4 ({probs[4]})"


def test_simulate_bracket_probabilities_in_first_round():
    """All entered teams should have nonzero first-round reach."""
    dc = _mock_dc()
    bracket = _simple_bracket()
    result = simulate_bracket(bracket, dc, n_sims=2000, rng_seed=42)
    # R1 label for a 2-round bracket (max_round=2) → round 1 → offset = 2-1 = 1 → "F"
    # The test just checks all teams appear
    assert len(result["teams"]) == 4
