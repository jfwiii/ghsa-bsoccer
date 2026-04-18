"""
Bracket simulator — M6.

Monte Carlo championship odds for all 8 GHSA playoff brackets.
N = 200,000 simulations per bracket.

Single-game simulation follows design §11:
1. Sample (home_goals, away_goals) from joint Dixon-Coles PMF.
2. If tied: sample overtime (20-min scaled rates). If still tied: PK coin flip.
"""

from __future__ import annotations

import json
import logging
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


def _get_flat_pmf(team_h: int, team_a: int, neutral: bool,
                  dc_result: dict, scale: float,
                  cache: dict) -> np.ndarray:
    """Return cached flat joint PMF for (team_h, team_a, neutral, scale)."""
    key = (team_h, team_a, neutral, scale)
    if key not in cache:
        alpha = dc_result["alpha"]
        beta = dc_result["beta"]
        gamma = dc_result["gamma"]
        rho = dc_result["rho"]
        a_h = alpha.get(team_h, 1.0)
        b_a = beta.get(team_a, 1.0)
        a_a = alpha.get(team_a, 1.0)
        b_h = beta.get(team_h, 1.0)
        lam = a_h * b_a * (gamma if not neutral else 1.0) * scale
        mu = a_a * b_h * scale
        cache[key] = dixon_coles._joint_pmf(lam, mu, rho, MAX_GOALS).ravel()
    return cache[key]


def simulate_game(team_h: int, team_a: int, neutral: bool,
                  dc_result: dict,
                  rng: np.random.Generator,
                  is_championship: bool = False,
                  pmf_cache: Optional[dict] = None) -> int:
    """
    Simulate a single game. Returns the winner's team_id.

    Tie resolution (design §11.1):
    1. OT: sample 20-min period (lambda_ot = lambda * 20/80).
    2. PK: weighted coin flip P(h wins) = sigmoid(0.3 * (log_alpha_h - log_alpha_a)).
    """
    if pmf_cache is None:
        pmf_cache = {}

    g = MAX_GOALS + 1
    ot_scale = OT_DURATION / 80.0

    flat = _get_flat_pmf(team_h, team_a, neutral, dc_result, 1.0, pmf_cache)
    hg, ag = divmod(int(rng.choice(g * g, p=flat)), g)

    if hg != ag:
        return team_h if hg > ag else team_a

    # OT
    flat_ot = _get_flat_pmf(team_h, team_a, neutral, dc_result, ot_scale, pmf_cache)
    hg_ot, ag_ot = divmod(int(rng.choice(g * g, p=flat_ot)), g)
    if hg_ot != ag_ot:
        return team_h if hg_ot > ag_ot else team_a

    # PK coin flip
    alpha = dc_result["alpha"]
    a_h = alpha.get(team_h, 1.0)
    a_a = alpha.get(team_a, 1.0)
    log_alpha_h = np.log(max(a_h, 1e-10))
    log_alpha_a = np.log(max(a_a, 1e-10))
    p_h = 1.0 / (1.0 + np.exp(-0.3 * (log_alpha_h - log_alpha_a)))
    return team_h if rng.random() < p_h else team_a


def _host_for_round(round_num: int, top_team_id: Optional[int],
                    bottom_team_id: Optional[int],
                    top_seed: Optional[int], bottom_seed: Optional[int]) -> bool:
    """
    Returns True if top_team is hosting (i.e., top_team is home).
    Per design §11.2: higher seed hosts through semifinals; championship is neutral.
    """
    if round_num >= 5:  # championship round — neutral
        return False

    # Higher seed = lower seed number = hosts
    ts = top_seed or 99
    bs = bottom_seed or 99
    if ts == bs:  # seeding tie → treat as neutral
        return False
    return ts < bs  # top seed hosts if it has lower (better) seed number


def simulate_bracket(bracket: dict, dc_result: dict,
                     n_sims: int = N_SIMS,
                     rng_seed: int = RNG_SEED) -> dict:
    """
    Run Monte Carlo simulation for one bracket.

    Handles the case where only Round 1 matchups are seeded (future rounds empty):
    infers total rounds from Round 1 size and synthesizes empty matchup slots so
    _find_feeder can populate them from previous-round winners.

    Returns dict formatted for bracket_odds.json.
    """
    rng = np.random.default_rng(rng_seed)
    bracket_name = bracket["bracket"]
    source_rounds = bracket["rounds"]

    # Index source rounds by round number
    source_by_rn: dict[int, list] = {rnd["round"]: rnd["matchups"] for rnd in source_rounds}

    # Determine total rounds from Round 1 matchup count
    r1_matchups = source_by_rn.get(1, [])
    n_r1 = len(r1_matchups)
    if n_r1 == 0:
        log.warning("bracket %s has no Round 1 matchups; skipping", bracket_name)
        return {"bracket": bracket_name, "n_simulations": n_sims, "teams": []}

    import math
    total_rounds = max(1, int(math.log2(n_r1 * 2)))  # 16 matchups → 5 rounds

    # Build complete round list: seeded matchups for Round 1, synthetic slots for later rounds
    all_rounds: list[dict] = []
    for rn in range(1, total_rounds + 1):
        if rn in source_by_rn and source_by_rn[rn]:
            all_rounds.append({"round": rn, "matchups": source_by_rn[rn]})
        else:
            n_matchups = n_r1 // (2 ** (rn - 1))
            synthetic = [
                {"position": pos, "top_team_id": None, "bottom_team_id": None,
                 "top_seed": None, "bottom_seed": None}
                for pos in range(1, n_matchups + 1)
            ]
            all_rounds.append({"round": rn, "matchups": synthetic})

    # Collect all teams (from Round 1 only)
    team_ids: set[int] = set()
    for m in r1_matchups:
        if m.get("top_team_id"):
            team_ids.add(m["top_team_id"])
        if m.get("bottom_team_id"):
            team_ids.add(m["bottom_team_id"])
    team_ids_list = sorted(t for t in team_ids if t is not None)

    reach_counts: dict[int, dict[int, int]] = {tid: {} for tid in team_ids_list}
    round_labels = _build_round_labels(total_rounds)

    pmf_cache: dict = {}  # shared across all sims — keyed by (h, a, neutral, scale)

    for sim_idx in range(n_sims):
        winners: dict[tuple[int, int], int] = {}  # (round, position) → winner team_id

        for rnd in all_rounds:
            rn = rnd["round"]
            is_champ_round = (rn == total_rounds)
            neutral_round = is_champ_round  # championship on neutral site

            for matchup in rnd["matchups"]:
                pos = matchup["position"]
                top_id = matchup.get("top_team_id")
                bot_id = matchup.get("bottom_team_id")

                if top_id is None:
                    top_id = _find_feeder(winners, rn, pos, "top")
                if bot_id is None:
                    bot_id = _find_feeder(winners, rn, pos, "bottom")

                if top_id is None and bot_id is None:
                    continue  # both missing — hole in bracket, can't simulate

                if top_id is None or bot_id is None:
                    # One team unknown — grant bye to the known team
                    bye_winner = top_id if top_id is not None else bot_id
                    if bye_winner in reach_counts:
                        reach_counts[bye_winner][rn] = reach_counts[bye_winner].get(rn, 0) + 1
                    winners[(rn, pos)] = bye_winner
                    if is_champ_round and bye_winner in reach_counts:
                        champ_rn = total_rounds + 1
                        reach_counts[bye_winner][champ_rn] = reach_counts[bye_winner].get(champ_rn, 0) + 1
                    continue

                for tid in [top_id, bot_id]:
                    if tid in reach_counts:
                        reach_counts[tid][rn] = reach_counts[tid].get(rn, 0) + 1

                top_is_home = not neutral_round
                neutral = neutral_round

                home_id = top_id if top_is_home else bot_id
                away_id = bot_id if top_is_home else top_id

                winner = simulate_game(home_id, away_id, neutral, dc_result, rng,
                                       is_championship=is_champ_round,
                                       pmf_cache=pmf_cache)
                winners[(rn, pos)] = winner

                if is_champ_round and winner in reach_counts:
                    champ_rn = total_rounds + 1
                    reach_counts[winner][champ_rn] = reach_counts[winner].get(champ_rn, 0) + 1

    # Convert counts to probabilities
    champ_rn = total_rounds + 1
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


def _build_round_labels(total_rounds: int) -> dict[int, str]:
    """
    Map round numbers to human-readable labels based on total bracket rounds.

    Example for total_rounds=5 (32-team bracket):
      Round 1 → R32, Round 2 → R16, Round 3 → QF, Round 4 → SF,
      Round 5 → F, Round 6 → Champ (championship winner)
    """
    n_start = 2 ** total_rounds  # number of teams entering Round 1
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
    labels[total_rounds + 1] = "Champ"  # championship winner
    return labels


def _find_feeder(winners: dict, current_round: int, current_pos: int,
                 slot: str) -> Optional[int]:
    """
    Look up the winner from the previous round that feeds into this slot.
    Bracket structure: position in round r comes from positions 2*pos-1 and 2*pos in round r-1.
    """
    prev_round = current_round - 1
    if prev_round < 1:
        return None

    if slot == "top":
        feeder_pos = 2 * current_pos - 1
    else:
        feeder_pos = 2 * current_pos

    return winners.get((prev_round, feeder_pos))


def simulate_all_brackets(brackets: list[dict], dc_result: dict,
                           n_sims: int = N_SIMS) -> list[dict]:
    """Simulate all 8 brackets."""
    results = []
    for i, bracket in enumerate(brackets):
        log.info("simulating bracket %s (%d sims)...", bracket["bracket"], n_sims)
        result = simulate_bracket(bracket, dc_result, n_sims=n_sims, rng_seed=RNG_SEED + i)
        results.append(result)
        log.info("done: %s", bracket["bracket"])
    return results


def save_bracket_odds(odds: list[dict]) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    path = PUBLIC_DIR / "bracket_odds.json"
    path.write_text(json.dumps(odds, indent=2))
    log.info("saved bracket_odds.json with %d brackets", len(odds))
