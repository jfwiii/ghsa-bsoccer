"""
Bracket simulator — M6.

Monte Carlo championship odds for all 8 GHSA playoff brackets.
N = 200,000 simulations per bracket.

Simulation is fully vectorised: for each matchup in each round, all N
outcomes are sampled in a single NumPy call using a precomputed win
probability (regulation + OT + PK), giving ~50-100× speedup over the
scalar loop. Home advantage goes to the higher-rated team in rounds 1–4;
the championship round is on a neutral site.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from model import dixon_coles

log = logging.getLogger(__name__)

N_SIMS = 200_000
OT_DURATION = 20.0  # minutes per OT period
MAX_GOALS = 10
DATA_DIR = Path("data")
PUBLIC_DIR = Path("public")

RNG_SEED = 42

_BYE = np.int64(-1)  # sentinel for unresolved bracket slots


# ---------------------------------------------------------------------------
# PMF helpers
# ---------------------------------------------------------------------------

def _get_flat_pmf(team_h: int, team_a: int, neutral: bool,
                  dc_result: dict, scale: float,
                  cache: dict) -> np.ndarray:
    """Return cached flat joint PMF for (team_h, team_a, neutral, scale)."""
    key = (team_h, team_a, neutral, scale)
    if key not in cache:
        alpha = dc_result["alpha"]
        beta  = dc_result["beta"]
        gamma = dc_result["gamma"]
        rho   = dc_result["rho"]
        a_h = alpha.get(team_h, 1.0)
        b_a = beta.get(team_a, 1.0)
        a_a = alpha.get(team_a, 1.0)
        b_h = beta.get(team_h, 1.0)
        lam = a_h * b_a * (gamma if not neutral else 1.0) * scale
        mu  = a_a * b_h * scale
        cache[key] = dixon_coles._joint_pmf(lam, mu, rho, MAX_GOALS).ravel()
    return cache[key]


def _win_prob(team_h: int, team_a: int, neutral: bool,
              dc_result: dict, pmf_cache: dict) -> float:
    """
    Compute P(team_h wins the game) analytically, including OT and PK.
    team_h is the designated home team (ignored when neutral=True).
    """
    g = MAX_GOALS + 1

    flat = _get_flat_pmf(team_h, team_a, neutral, dc_result, 1.0, pmf_cache)
    mat  = flat.reshape(g, g)
    p_h_reg  = float(np.triu(mat, k=1).sum())
    p_a_reg  = float(np.tril(mat, k=-1).sum())
    p_draw   = max(0.0, 1.0 - p_h_reg - p_a_reg)

    flat_ot  = _get_flat_pmf(team_h, team_a, neutral, dc_result,
                              OT_DURATION / 80.0, pmf_cache)
    mat_ot   = flat_ot.reshape(g, g)
    p_h_ot   = float(np.triu(mat_ot, k=1).sum())
    p_draw_ot = float(np.diag(mat_ot).sum())

    alpha = dc_result["alpha"]
    a_h   = alpha.get(team_h, 1.0)
    a_a   = alpha.get(team_a, 1.0)
    p_h_pk = 1.0 / (1.0 + np.exp(
        -0.3 * (np.log(max(a_h, 1e-10)) - np.log(max(a_a, 1e-10)))
    ))

    return p_h_reg + p_draw * (p_h_ot + p_draw_ot * float(p_h_pk))


# ---------------------------------------------------------------------------
# Standalone game simulator (kept for external use)
# ---------------------------------------------------------------------------

def simulate_game(team_h: int, team_a: int, neutral: bool,
                  dc_result: dict,
                  rng: np.random.Generator,
                  is_championship: bool = False,
                  pmf_cache: Optional[dict] = None) -> int:
    """Simulate a single game. Returns the winner's team_id."""
    if pmf_cache is None:
        pmf_cache = {}
    g = MAX_GOALS + 1
    flat = _get_flat_pmf(team_h, team_a, neutral, dc_result, 1.0, pmf_cache)
    hg, ag = divmod(int(rng.choice(g * g, p=flat)), g)
    if hg != ag:
        return team_h if hg > ag else team_a

    flat_ot = _get_flat_pmf(team_h, team_a, neutral, dc_result,
                             OT_DURATION / 80.0, pmf_cache)
    hg_ot, ag_ot = divmod(int(rng.choice(g * g, p=flat_ot)), g)
    if hg_ot != ag_ot:
        return team_h if hg_ot > ag_ot else team_a

    alpha = dc_result["alpha"]
    a_h = alpha.get(team_h, 1.0)
    a_a = alpha.get(team_a, 1.0)
    p_h = 1.0 / (1.0 + np.exp(
        -0.3 * (np.log(max(a_h, 1e-10)) - np.log(max(a_a, 1e-10)))
    ))
    return team_h if rng.random() < p_h else team_a


# ---------------------------------------------------------------------------
# Vectorised bracket simulator
# ---------------------------------------------------------------------------

def simulate_bracket(bracket: dict, dc_result: dict,
                     n_sims: int = N_SIMS,
                     rng_seed: int = RNG_SEED) -> dict:
    """
    Run vectorised Monte Carlo simulation for one bracket.

    For each matchup in each round, all N outcomes are drawn simultaneously
    via a single rng.random() call — no Python loop over simulations.

    Home advantage: the higher-rated team (DC rating) hosts rounds 1–4.
    The championship round is neutral.

    Returns a dict formatted for bracket_odds.json.
    """
    rng = np.random.default_rng(rng_seed)
    bracket_name = bracket["bracket"]
    source_by_rn = {rnd["round"]: rnd["matchups"] for rnd in bracket["rounds"]}

    r1_matchups = source_by_rn.get(1, [])
    n_r1 = len(r1_matchups)
    if n_r1 == 0:
        log.warning("bracket %s has no Round 1 matchups; skipping", bracket_name)
        return {"bracket": bracket_name, "n_simulations": n_sims, "teams": []}

    total_rounds = max(1, int(math.log2(n_r1 * 2)))
    champ_rn = total_rounds + 1

    # Collect all participating team IDs
    team_ids: set[int] = set()
    for m in r1_matchups:
        if m.get("top_team_id"):
            team_ids.add(int(m["top_team_id"]))
        if m.get("bottom_team_id"):
            team_ids.add(int(m["bottom_team_id"]))
    team_ids_list = sorted(team_ids)

    rating   = dc_result.get("rating", {})
    pmf_cache: dict = {}
    wp_cache:  dict = {}

    def get_wp(h: int, a: int, neutral: bool) -> float:
        key = (h, a, neutral)
        if key not in wp_cache:
            wp_cache[key] = _win_prob(h, a, neutral, dc_result, pmf_cache)
        return wp_cache[key]

    reach_counts: dict[int, dict[int, int]] = {tid: {} for tid in team_ids_list}
    round_labels = _build_round_labels(total_rounds)

    # sim_slots: flat list of np.ndarray(n_sims, dtype=int64).
    # Initially holds pairs [top_arr_0, bot_arr_0, top_arr_1, bot_arr_1, ...]
    # for each R1 matchup.  After each round the list shrinks: only the
    # winners remain, paired consecutively for the next round.
    sim_slots: list[np.ndarray] = []
    for m in r1_matchups:
        top_id = m.get("top_team_id")
        bot_id = m.get("bottom_team_id")
        sim_slots.append(np.full(n_sims,
                                 np.int64(top_id) if top_id is not None else _BYE,
                                 dtype=np.int64))
        sim_slots.append(np.full(n_sims,
                                 np.int64(bot_id) if bot_id is not None else _BYE,
                                 dtype=np.int64))

    for rn in range(1, total_rounds + 1):
        is_champ   = (rn == total_rounds)
        n_matchups = len(sim_slots) // 2
        next_slots: list[np.ndarray] = []

        for mi in range(n_matchups):
            top_arr = sim_slots[2 * mi]
            bot_arr = sim_slots[2 * mi + 1]

            # Count teams entering this round
            for arr in (top_arr, bot_arr):
                valid = arr[arr != _BYE]
                if valid.size:
                    uids, ucnts = np.unique(valid, return_counts=True)
                    for tid, cnt in zip(uids.tolist(), ucnts.tolist()):
                        if tid in reach_counts:
                            reach_counts[tid][rn] = (
                                reach_counts[tid].get(rn, 0) + cnt
                            )

            # Resolve byes
            both_bye   = (top_arr == _BYE) & (bot_arr == _BYE)
            top_is_bye = (top_arr == _BYE) & ~both_bye
            bot_is_bye = (bot_arr == _BYE) & ~both_bye
            valid_mask = ~both_bye & ~top_is_bye & ~bot_is_bye

            winners = np.where(top_is_bye, bot_arr,
                      np.where(bot_is_bye, top_arr, _BYE)).copy()

            if valid_mask.any():
                top_v  = top_arr[valid_mask]
                bot_v  = bot_arr[valid_mask]

                top_wins = np.empty(int(valid_mask.sum()), dtype=bool)

                # Encode (top_id, bot_id) as a single int64 to find unique
                # ORDERED pairs without np.unique reordering the columns.
                # Team IDs are <1e6 so SCALE=1e9 keeps values within int64.
                _SCALE = np.int64(1_000_000_000)
                codes  = top_v.astype(np.int64) * _SCALE + bot_v.astype(np.int64)
                for code in np.unique(codes):
                    team_h = int(code // _SCALE)
                    team_a = int(code % _SCALE)
                    pm     = codes == code

                    if is_champ:
                        p_top = get_wp(team_h, team_a, True)
                    else:
                        r_h = rating.get(team_h, 0.0)
                        r_a = rating.get(team_a, 0.0)
                        if r_h >= r_a:
                            # top team is higher-rated → home
                            p_top = get_wp(team_h, team_a, False)
                        else:
                            # bottom team is higher-rated → home
                            p_top = 1.0 - get_wp(team_a, team_h, False)

                    top_wins[pm] = rng.random(int(pm.sum())) < p_top

                winners[valid_mask] = np.where(top_wins, top_v, bot_v)

            # Track championship winners
            if is_champ:
                valid_w = winners[winners != _BYE]
                if valid_w.size:
                    uids, ucnts = np.unique(valid_w, return_counts=True)
                    for tid, cnt in zip(uids.tolist(), ucnts.tolist()):
                        if tid in reach_counts:
                            reach_counts[tid][champ_rn] = (
                                reach_counts[tid].get(champ_rn, 0) + cnt
                            )

            next_slots.append(winners)

        sim_slots = next_slots

    # Build output
    teams_out = []
    for tid in team_ids_list:
        counts = reach_counts[tid]
        round_probs = {
            label: round(counts.get(rn, 0) / n_sims, 4)
            for rn, label in round_labels.items()
        }
        teams_out.append({"team_id": tid, "round_probabilities": round_probs})

    teams_out.sort(key=lambda t: -t["round_probabilities"].get("Champ", 0))

    return {
        "bracket": bracket_name,
        "n_simulations": n_sims,
        "teams": teams_out,
    }


# ---------------------------------------------------------------------------
# Round label builder
# ---------------------------------------------------------------------------

def _build_round_labels(total_rounds: int) -> dict[int, str]:
    """Map round numbers to human-readable labels."""
    n_start = 2 ** total_rounds
    labels: dict[int, str] = {}
    for rn in range(1, total_rounds + 1):
        n_teams = n_start // (2 ** (rn - 1))
        if n_teams >= 64:
            labels[rn] = f"R{n_teams}"
        elif n_teams == 32:
            labels[rn] = "R32"
        elif n_teams == 16:
            labels[rn] = "R16"
        elif n_teams == 8:
            labels[rn] = "QF"
        elif n_teams == 4:
            labels[rn] = "SF"
        elif n_teams == 2:
            labels[rn] = "F"
        else:
            labels[rn] = f"R{n_teams}"
    labels[total_rounds + 1] = "Champ"
    return labels


# ---------------------------------------------------------------------------
# Multi-bracket runner
# ---------------------------------------------------------------------------

def simulate_all_brackets(brackets: list[dict], dc_result: dict,
                           n_sims: int = N_SIMS) -> list[dict]:
    """Simulate all brackets and return results."""
    results = []
    for i, bracket in enumerate(brackets):
        log.info("simulating bracket %s (%d sims)...", bracket["bracket"], n_sims)
        result = simulate_bracket(bracket, dc_result,
                                  n_sims=n_sims, rng_seed=RNG_SEED + i)
        results.append(result)
        log.info("done: %s", bracket["bracket"])
    return results


def save_bracket_odds(odds: list[dict]) -> None:
    import json
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    path = PUBLIC_DIR / "bracket_odds.json"
    path.write_text(json.dumps(odds, indent=2))
    log.info("saved bracket_odds.json with %d brackets", len(odds))
