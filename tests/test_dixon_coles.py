"""Tests for dixon_coles.py — unit tests for likelihood and prediction."""

import numpy as np
import pytest

from model.dixon_coles import (
    _tau,
    _joint_pmf,
    log_likelihood_game,
    predict_matchup,
)


# ---------------------------------------------------------------------------
# tau correction
# ---------------------------------------------------------------------------

def test_tau_00():
    # tau(0,0) = 1 - lam*mu*rho
    lam, mu, rho = 2.0, 1.5, -0.1
    assert abs(_tau(0, 0, lam, mu, rho) - (1 - lam * mu * rho)) < 1e-9


def test_tau_01():
    lam, mu, rho = 2.0, 1.5, -0.1
    assert abs(_tau(0, 1, lam, mu, rho) - (1 + lam * rho)) < 1e-9


def test_tau_10():
    lam, mu, rho = 2.0, 1.5, -0.1
    assert abs(_tau(1, 0, lam, mu, rho) - (1 + mu * rho)) < 1e-9


def test_tau_11():
    lam, mu, rho = 2.0, 1.5, -0.1
    assert abs(_tau(1, 1, lam, mu, rho) - (1 - rho)) < 1e-9


def test_tau_other():
    assert _tau(3, 2, 2.0, 1.5, -0.1) == 1.0


# ---------------------------------------------------------------------------
# Joint PMF
# ---------------------------------------------------------------------------

def test_joint_pmf_sums_to_one():
    for lam, mu, rho in [(1.5, 1.2, -0.1), (3.0, 2.5, 0.0), (1.0, 1.0, 0.1)]:
        pmf = _joint_pmf(lam, mu, rho)
        assert abs(pmf.sum() - 1.0) < 1e-6, f"PMF sum={pmf.sum()} for lam={lam},mu={mu},rho={rho}"


def test_joint_pmf_non_negative():
    pmf = _joint_pmf(2.0, 1.5, -0.2)
    assert (pmf >= 0).all()


def test_joint_pmf_shape():
    pmf = _joint_pmf(2.0, 1.5, 0.0, max_goals=8)
    assert pmf.shape == (9, 9)


# ---------------------------------------------------------------------------
# Log likelihood
# ---------------------------------------------------------------------------

def test_log_likelihood_decreases_with_unlikely_score():
    """Higher-probability score should have better (less negative) LL."""
    alpha_h, beta_a, alpha_a, beta_h = 1.5, 0.8, 0.8, 1.2
    gamma, rho = 1.3, -0.05

    # Expected: lam ≈ 1.5*0.8*1.3=1.56, mu ≈ 0.8*1.2=0.96
    # Score 2-1 is near the mode; score 0-8 is very unlikely
    ll_likely = log_likelihood_game(alpha_h, beta_a, alpha_a, beta_h,
                                     gamma, rho, 2, 1, 80, False)
    ll_unlikely = log_likelihood_game(alpha_h, beta_a, alpha_a, beta_h,
                                       gamma, rho, 0, 8, 80, False)
    assert ll_likely > ll_unlikely


# ---------------------------------------------------------------------------
# predict_matchup
# ---------------------------------------------------------------------------

def _mock_dc_result(alpha_h=1.5, alpha_a=0.8, beta_h=0.9, beta_a=0.7,
                     gamma=1.2, rho=-0.05):
    return {
        "alpha": {1: alpha_h, 2: alpha_a},
        "beta": {1: beta_h, 2: beta_a},
        "gamma": gamma,
        "rho": rho,
        "rating": {1: 1.0, 2: 0.7},
    }


def test_predict_matchup_probs_sum_to_one():
    dc = _mock_dc_result()
    result = predict_matchup(1, 2, False, dc)
    total = result["p_home_win"] + result["p_draw"] + result["p_away_win"]
    assert abs(total - 1.0) < 1e-5


def test_predict_matchup_stronger_team_favored():
    dc = _mock_dc_result(alpha_h=2.0, alpha_a=0.5)  # team 1 much stronger
    result = predict_matchup(1, 2, False, dc)
    assert result["p_home_win"] > result["p_away_win"]


def test_predict_matchup_neutral_reduces_home_win():
    dc = _mock_dc_result()
    home = predict_matchup(1, 2, neutral=False, dc_result=dc)
    neutral = predict_matchup(1, 2, neutral=True, dc_result=dc)
    # Home advantage removed → home win prob should decrease
    assert neutral["p_home_win"] < home["p_home_win"]


def test_predict_matchup_duration_offset():
    """Shorter game → fewer goals → less spread in scores."""
    dc = _mock_dc_result()
    full = predict_matchup(1, 2, False, dc, duration_min=80)
    short = predict_matchup(1, 2, False, dc, duration_min=40)
    # Shorter game → higher draw probability
    assert short["p_draw"] > full["p_draw"]
