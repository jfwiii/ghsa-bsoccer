"""
Dixon-Coles bivariate Poisson model — M5.

Reference: Dixon & Coles (1997), Applied Statistics 46(2), 265-280.

Specification:
  lambda_raw = alpha_h * beta_a * gamma^is_home   (expected home goals per 80 min)
  mu_raw     = alpha_a * beta_h
  lambda     = lambda_raw * (duration / 80)
  mu         = mu_raw     * (duration / 80)

Joint probability with low-score correction tau (rho parameter).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

log = logging.getLogger(__name__)

MAX_GOALS = 10  # truncate joint distribution at 10 goals per side


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _joint_pmf(lam: float, mu: float, rho: float,
               max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Compute (max_goals+1) x (max_goals+1) joint probability matrix P(X=x, Y=y).
    """
    g = max_goals + 1
    px = poisson.pmf(np.arange(g), lam)
    py = poisson.pmf(np.arange(g), mu)
    joint = np.outer(px, py)

    # Apply tau correction for (0,0), (0,1), (1,0), (1,1).
    # Clip to 0: when lam*mu*rho > 1, tau(0,0) would go negative; treat as impossible.
    for x in range(min(2, g)):
        for y in range(min(2, g)):
            joint[x, y] *= max(0.0, _tau(x, y, lam, mu, rho))

    # Renormalize so probabilities sum to 1 (tau shifts mass slightly)
    total = joint.sum()
    if total > 0:
        joint /= total
    return joint


def log_likelihood_game(alpha_h: float, beta_a: float, alpha_a: float, beta_h: float,
                         gamma: float, rho: float,
                         home_goals: int, away_goals: int,
                         duration_min: float, neutral: bool,
                         weight: float = 1.0) -> float:
    """Log-likelihood contribution of a single game."""
    lam_raw = alpha_h * beta_a * (gamma if not neutral else 1.0)
    mu_raw = alpha_a * beta_h
    scale = duration_min / 80.0
    lam = lam_raw * scale
    mu = mu_raw * scale

    x, y = home_goals, away_goals
    px = poisson.pmf(x, lam)
    py = poisson.pmf(y, mu)
    t = _tau(x, y, lam, mu, rho)

    ll = np.log(max(px * py * t, 1e-300))
    return weight * ll


def fit(games_df,
        massey_result: dict,
        xi: float = 0.015,
        rho_init: float = 0.0,
        rho_prior_sigma: float = 0.03) -> dict:
    """
    Fit Dixon-Coles model via L-BFGS-B.

    Args:
        games_df: normalized games DataFrame.
        massey_result: output of model.massey.fit(), used for warm start.
        xi: time-decay half-life parameter (per day).
        rho_init: initial rho value.
        rho_prior_sigma: standard deviation for Gaussian prior on rho.

    Returns dict with:
        alpha: {team_id: attack_rate}
        beta:  {team_id: defense_rate}
        gamma: home_advantage multiplier
        rho:   correlation correction
        team_ids: ordered list
    """
    import pandas as pd

    df = games_df.dropna(subset=["home_team_id", "away_team_id",
                                  "home_goals_regulation", "away_goals_regulation"]).copy()
    df = df[~df.get("is_non_ghsa", pd.Series(False, index=df.index))]

    if df.empty:
        log.warning("Dixon-Coles: no games to fit")
        return {"alpha": {}, "beta": {}, "gamma": 1.0, "rho": 0.0, "team_ids": []}

    # Time-decay weights
    t_max = df["date"].max()
    dt = (t_max - df["date"]).dt.days.astype(float)
    weights = np.exp(-xi * dt.values)

    # Team index from Massey warm start
    massey_ratings = massey_result.get("ratings", {})
    all_ids = sorted(set(df["home_team_id"].astype(int)) | set(df["away_team_id"].astype(int)))
    idx = {tid: i for i, tid in enumerate(all_ids)}
    n = len(all_ids)

    # Precompute game arrays for fast likelihood
    home_idx = df["home_team_id"].astype(int).map(idx).values
    away_idx = df["away_team_id"].astype(int).map(idx).values
    home_goals = df["home_goals_regulation"].astype(int).values
    away_goals = df["away_goals_regulation"].astype(int).values
    durations = df["duration_used_min"].astype(float).values
    neutral = df["neutral_site"].fillna(False).values

    from scipy.special import gammaln

    # Precompute log-factorial terms (constant across optimizer calls)
    log_fact_h = gammaln(home_goals + 1)
    log_fact_a = gammaln(away_goals + 1)
    # Mask arrays for tau correction (low-score cells)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    scale = durations / 80.0
    gamma_mask = np.where(~neutral, 1.0, 0.0)  # 1 = apply gamma, 0 = neutral

    # Identifiability target: geometric_mean(alpha)*geometric_mean(beta) = league_avg_goals
    # so each geometric_mean = sqrt(league_avg_goals) → log_target = 0.5*log(league_avg_goals)
    safe_scale = np.maximum(scale, 1e-6)
    league_avg_goals = float(np.mean(np.concatenate([
        home_goals / safe_scale,
        away_goals / safe_scale,
    ])))
    log_target = 0.5 * np.log(max(league_avg_goals, 0.1))
    log.info("League avg goals/team/80min: %.3f, log_target=%.4f", league_avg_goals, log_target)

    # Warm-start from Massey (ratings are centered at 0, mean=0)
    massey_arr = np.array([massey_ratings.get(tid, 0.0) for tid in all_ids])
    massey_arr -= massey_arr.mean()  # ensure zero-centered
    # Shift to match league scale: geometric_mean(alpha)=geometric_mean(beta)=sqrt(league_avg)
    log_alpha_init = massey_arr + log_target
    log_beta_init = -massey_arr + log_target
    # home_advantage from Massey is log-MOV units; use a neutral warm start for log_gamma
    log_gamma_init = 0.1  # gamma ≈ 1.1, mild home advantage

    x0 = np.concatenate([log_alpha_init, log_beta_init, [log_gamma_init, rho_init]])

    def neg_log_likelihood(x: np.ndarray) -> float:
        log_alpha = x[:n]
        log_beta = x[n:2*n]
        log_gamma = x[2*n]
        rho = x[2*n + 1]
        gamma = np.exp(log_gamma)

        # Vectorized rate computation
        la_h = log_alpha[home_idx]
        lb_a = log_beta[away_idx]
        la_a = log_alpha[away_idx]
        lb_h = log_beta[home_idx]

        log_lam_raw = la_h + lb_a + log_gamma * gamma_mask
        log_mu_raw  = la_a + lb_h

        log_scale = np.log(np.maximum(scale, 1e-10))
        lam = np.exp(log_lam_raw + log_scale)
        mu  = np.exp(log_mu_raw  + log_scale)

        # Poisson log-PMF: x*log(lam) - lam - log(x!)
        log_px = home_goals * np.log(np.maximum(lam, 1e-300)) - lam - log_fact_h
        log_py = away_goals * np.log(np.maximum(mu,  1e-300)) - mu  - log_fact_a

        # Tau correction (vectorized over low-score cells only)
        log_tau = np.zeros(len(home_goals))
        if m00.any():
            log_tau[m00] = np.log(np.maximum(1 - lam[m00] * mu[m00] * rho, 1e-300))
        if m01.any():
            log_tau[m01] = np.log(np.maximum(1 + lam[m01] * rho, 1e-300))
        if m10.any():
            log_tau[m10] = np.log(np.maximum(1 + mu[m10] * rho, 1e-300))
        if m11.any():
            log_tau[m11] = np.log(np.maximum(1 - rho, 1e-300))

        ll = log_px + log_py + log_tau
        nll = -float(np.dot(weights, ll))

        # Gaussian prior on rho
        nll += 0.5 * (rho / rho_prior_sigma) ** 2

        # Identifiability: very soft constraint — just enough to break the alpha/beta scale
        # degeneracy (adding k to all alpha, subtracting k from all beta leaves likelihood
        # unchanged). Penalty must be << likelihood gradient to avoid distorting estimates.
        nll += 0.01 * (log_alpha.sum() - n * log_target) ** 2
        nll += 0.01 * (log_beta.sum() - n * log_target) ** 2

        return nll

    # Bounds: rho in [-0.3, 0.4] — allow some positive rho (PK phantom goals inflate
    # 1-0/0-1 outcomes) but cap it so tau stays valid: tau(0,0)=1-lam*mu*rho must stay >0
    # for typical lam≈mu≈1.7, rho < 1/1.7^2 ≈ 0.35.
    # log_gamma unbounded so gamma can settle naturally from the data.
    bounds = (
        [(None, None)] * n +   # log_alpha
        [(None, None)] * n +   # log_beta
        [(None, None)] +       # log_gamma: unconstrained
        [(-0.3, 0.35)]         # rho: bounded so tau(0,0) stays ≥ 0 for typical rates
    )

    log.info("Dixon-Coles: fitting %d params over %d games", len(x0), len(df))
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                      bounds=bounds,
                      options={"maxiter": 3000, "maxfun": 500000, "ftol": 1e-10})

    if not result.success:
        log.warning("Dixon-Coles optimizer did not converge: %s", result.message)

    x_opt = result.x
    log_alpha = x_opt[:n]
    log_beta = x_opt[n:2*n]

    # Re-center to enforce geometric_mean=sqrt(league_avg)
    log_alpha -= (log_alpha.mean() - log_target)
    log_beta -= (log_beta.mean() - log_target)

    log_gamma = x_opt[2*n]
    rho = x_opt[2*n + 1]

    alpha_dict = {all_ids[i]: float(np.exp(log_alpha[i])) for i in range(n)}
    beta_dict = {all_ids[i]: float(np.exp(log_beta[i])) for i in range(n)}
    gamma = float(np.exp(log_gamma))

    # Rating = attack / defense (geometric mean style: sqrt(alpha/beta))
    rating_dict = {tid: float(np.sqrt(alpha_dict[tid] / max(beta_dict[tid], 1e-6)))
                   for tid in all_ids}

    log.info(
        "Dixon-Coles fit: gamma=%.3f, rho=%.4f, converged=%s",
        gamma, rho, result.success
    )

    # Calibration check: geometric mean of alpha*beta should match league avg
    log_alpha_final = np.array([float(np.log(max(alpha_dict[tid], 1e-300))) for tid in all_ids])
    log_beta_final = np.array([float(np.log(max(beta_dict[tid], 1e-300))) for tid in all_ids])
    implied_geomean = float(np.exp(np.mean(log_alpha_final + log_beta_final)))
    log.info(
        "Calibration: geometric_mean(alpha*beta)=%.3f, league_avg=%.3f",
        implied_geomean, league_avg_goals,
    )
    if not (league_avg_goals * 0.85 <= implied_geomean <= league_avg_goals * 1.15):
        log.warning("Calibration off: geomean=%.3f vs league_avg=%.3f", implied_geomean, league_avg_goals)

    return {
        "alpha": alpha_dict,
        "beta": beta_dict,
        "gamma": gamma,
        "rho": float(rho),
        "rating": rating_dict,
        "team_ids": all_ids,
        "log_gamma": float(log_gamma),
    }


def predict_matchup(team_h: int, team_a: int, neutral: bool,
                    dc_result: dict,
                    duration_min: float = 80.0,
                    max_goals: int = MAX_GOALS) -> dict:
    """
    Compute win/draw/loss probabilities and joint PMF for a matchup.

    Returns dict with:
        p_home_win, p_draw, p_away_win: regulation outcome probabilities
        joint_pmf: (max_goals+1) x (max_goals+1) numpy array
        lambda: expected home goals
        mu: expected away goals
    """
    alpha = dc_result["alpha"]
    beta = dc_result["beta"]
    gamma = dc_result["gamma"]
    rho = dc_result["rho"]

    a_h = alpha.get(team_h, 1.0)
    b_a = beta.get(team_a, 1.0)
    a_a = alpha.get(team_a, 1.0)
    b_h = beta.get(team_h, 1.0)

    lam_raw = a_h * b_a * (gamma if not neutral else 1.0)
    mu_raw = a_a * b_h
    scale = duration_min / 80.0
    lam = lam_raw * scale
    mu = mu_raw * scale

    joint = _joint_pmf(lam, mu, rho, max_goals)

    p_home = float(np.tril(joint, -1).sum())  # home_goals > away_goals
    p_draw = float(np.diag(joint).sum())
    p_away = float(np.triu(joint, 1).sum())

    return {
        "p_home_win": p_home,
        "p_draw": p_draw,
        "p_away_win": p_away,
        "joint_pmf": joint,
        "lambda": lam,
        "mu": mu,
        "lambda_raw": lam_raw,
        "mu_raw": mu_raw,
    }
