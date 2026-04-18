"""
Full pipeline runner.

Usage:
  python scripts/run_full_pipeline.py
  python scripts/run_full_pipeline.py --mini            # small test run with cached fixtures
  python scripts/run_full_pipeline.py --no-maxpreps     # GHSA-only, graceful degradation
  python scripts/run_full_pipeline.py --playoffs        # shorter cache TTL
  python scripts/run_full_pipeline.py --skip-sim        # skip bracket simulation
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import normalize as norm_module
from pipeline.normalize import Normalizer, save_parquets
from pipeline.scrape_ghsa import scrape_all_teams
from pipeline.scrape_maxpreps import MaxPrepsAbort, MaxPrepsClient, discover_slugs, enrich_games
from pipeline.brackets import ingest_all_brackets, save_brackets, load_brackets
from model import massey as massey_module
from model import dixon_coles
from model import simulate as sim_module
from model import evaluate as eval_module

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

PUBLIC_DIR = Path("public")
DATA_DIR = Path("data")

# Mini-mode: a handful of team IDs for fast integration testing
MINI_TEAM_IDS = [76, 38, 611, 100, 200, 300, 400, 500, 600, 700]


def build_ratings_json(teams_df: pd.DataFrame, games_df: pd.DataFrame,
                       dc_result: dict, massey_result: dict,
                       enrichment_coverage: dict) -> dict:
    id_to_meta: dict[int, dict] = {
        int(r["team_id"]): r for r in teams_df.to_dict("records")
    }

    # Compute ranks
    alpha = dc_result.get("alpha", {})
    beta = dc_result.get("beta", {})
    rating = dc_result.get("rating", {})

    sorted_by_rating = sorted(rating.items(), key=lambda x: -x[1])
    overall_rank = {tid: i + 1 for i, (tid, _) in enumerate(sorted_by_rating)}

    # Per-class rank
    class_teams: dict[str, list[tuple[int, float]]] = {}
    for tid, r in sorted_by_rating:
        meta = id_to_meta.get(tid, {})
        cls = meta.get("class", "")
        class_teams.setdefault(cls, []).append((tid, r))
    class_rank: dict[int, int] = {}
    for cls, lst in class_teams.items():
        for i, (tid, _) in enumerate(lst):
            class_rank[tid] = i + 1

    # Per-region rank
    region_teams: dict[str, list[tuple[int, float]]] = {}
    for tid, r in sorted_by_rating:
        meta = id_to_meta.get(tid, {})
        region = meta.get("region_or_area", "") or ""
        if region:
            region_teams.setdefault(region, []).append((tid, r))
    region_rank: dict[int, int] = {}
    for region, lst in region_teams.items():
        for i, (tid, _) in enumerate(lst):
            region_rank[tid] = i + 1

    # Records from games_df
    records: dict[int, tuple[int, int, int]] = {}  # team_id → (W, L, T)
    for _, g in games_df.iterrows():
        h, a = int(g["home_team_id"]), int(g["away_team_id"])
        hg, ag = g["home_goals_regulation"], g["away_goals_regulation"]
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg, ag = int(hg), int(ag)
        for tid, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            w, l, t = records.get(tid, (0, 0, 0))
            if gf > ga:
                records[tid] = (w + 1, l, t)
            elif ga > gf:
                records[tid] = (w, l + 1, t)
            else:
                records[tid] = (w, l, t + 1)

    massey_ratings = massey_result.get("ratings", {})

    teams_out = []
    for tid, r in sorted_by_rating:
        meta = id_to_meta.get(tid, {})
        w, l, t = records.get(tid, (0, 0, 0))
        rec = f"{w}-{l}" + (f"-{t}" if t else "")

        # Playoff bracket assignment
        cls = meta.get("class", "")
        is_private = meta.get("is_private", False)
        bracket = "Private" if is_private else cls

        teams_out.append({
            "team_id": tid,
            "name": meta.get("name", str(tid)),
            "class": cls,
            "region_or_area": meta.get("region_or_area"),
            "playoff_bracket": bracket,
            "record": rec,
            "attack": round(alpha.get(tid, 1.0), 4),
            "defense": round(beta.get(tid, 1.0), 4),
            "rating": round(r, 4),
            "rating_rank_overall": overall_rank.get(tid),
            "rating_rank_class": class_rank.get(tid),
            "rating_rank_region": region_rank.get(tid),
            "maxpreps_state_rank": meta.get("maxpreps_ranking"),
            "psr_rank": None,  # PSR not ingested yet
        })

    return {
        "fit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hyperparams": {
            "xi": 0.015,
            "home_advantage_log": round(dc_result.get("log_gamma", 0.18), 4),
            "rho": round(dc_result.get("rho", -0.08), 4),
        },
        "enrichment_coverage": enrichment_coverage,
        "teams": teams_out,
    }


def main():
    parser = argparse.ArgumentParser(description="GHSA Soccer pipeline")
    parser.add_argument("--mini", action="store_true", help="Mini test run")
    parser.add_argument("--no-maxpreps", action="store_true", help="Skip MaxPreps")
    parser.add_argument("--playoffs", action="store_true", help="Shorter cache TTL")
    parser.add_argument("--skip-sim", action="store_true", help="Skip Monte Carlo")
    parser.add_argument("--n-sims", type=int, default=sim_module.N_SIMS)
    args = parser.parse_args()

    no_maxpreps = args.no_maxpreps
    playoffs = args.playoffs
    mini = args.mini
    team_ids = MINI_TEAM_IDS if mini else None

    # -----------------------------------------------------------------------
    # M1: GHSA scrape
    # -----------------------------------------------------------------------
    log.info("=== M1: GHSA scrape ===")
    teams_raw, games_raw = scrape_all_teams(playoffs=playoffs, team_ids=team_ids)
    log.info("GHSA: %d teams, %d raw game records", len(teams_raw), len(games_raw))

    # -----------------------------------------------------------------------
    # M2: MaxPreps slug discovery & enrichment
    # -----------------------------------------------------------------------
    enrichments = []
    maxpreps_rankings: dict[int, int] = {}

    if not no_maxpreps:
        log.info("=== M2: MaxPreps slug discovery ===")
        mp_client = MaxPrepsClient(playoffs=playoffs)
        try:
            slug_map = discover_slugs(teams_raw, mp_client)

            # Store slugs back on team dicts
            for t in teams_raw:
                t["maxpreps_url_slug"] = slug_map.get(t["team_id"])

            # M3 enrichment
            log.info("=== M2/M3: MaxPreps game enrichment ===")
            # Build preliminary games df for enrichment
            pre_norm = Normalizer(teams_raw, games_raw, no_maxpreps=True)
            _, pre_games = pre_norm.run()

            if not pre_games.empty:
                enrichments = enrich_games(pre_games, slug_map, mp_client)

        except MaxPrepsAbort as e:
            log.error("MaxPreps aborted: %s — continuing with GHSA-only data", e)
            no_maxpreps = True
    else:
        log.info("Skipping MaxPreps (--no-maxpreps)")

    # -----------------------------------------------------------------------
    # M3: Normalize & merge
    # -----------------------------------------------------------------------
    log.info("=== M3: Normalize & merge ===")
    normalizer = Normalizer(teams_raw, games_raw, enrichments=enrichments,
                            no_maxpreps=no_maxpreps)
    teams_df, games_df = normalizer.run()

    # Validation checks
    log.info("games.parquet: %d rows", len(games_df))
    assert len(games_df) > 0, "No games produced — check scraper"
    assert games_df["duration_used_min"].notna().all(), "duration_used_min has nulls"

    if not games_df.empty:
        mov = (games_df["home_goals"] - games_df["away_goals"]).abs()
        mean_mov = mov.mean()
        mean_goals = ((games_df["home_goals"] + games_df["away_goals"]) / 2).mean()
        log.info("mean MOV=%.2f (ref 3.12), mean goals/team=%.2f (ref 2.32)", mean_mov, mean_goals)

    # Enrichment coverage for ratings.json
    n_total = len(games_df)
    n_mp_matched = int(games_df["maxpreps_matched"].sum()) if "maxpreps_matched" in games_df.columns else 0
    n_obs_duration = int(games_df["observed_duration_min"].notna().sum())
    n_reg_correction = int(
        (games_df["home_goals_regulation"] != games_df["home_goals"]).sum()
        if not games_df.empty else 0
    )
    n_score_disc = int(games_df.get("score_discrepancy", pd.Series(False)).sum())

    enrichment_coverage = {
        "n_games_total": n_total,
        "n_games_maxpreps_matched": n_mp_matched,
        "n_games_with_observed_duration": n_obs_duration,
        "n_games_with_regulation_correction": n_reg_correction,
        "n_score_discrepancies_logged": n_score_disc,
    }

    save_parquets(teams_df, games_df)

    # -----------------------------------------------------------------------
    # M4: Massey warm start
    # -----------------------------------------------------------------------
    log.info("=== M4: Massey fit ===")
    massey_result = massey_module.fit(games_df, teams_df,
                                      maxpreps_rankings=maxpreps_rankings or None)

    # -----------------------------------------------------------------------
    # M5: Dixon-Coles fit
    # -----------------------------------------------------------------------
    log.info("=== M5: Dixon-Coles fit ===")
    # Train/test split for evaluation
    train_df, test_df = eval_module.split_holdout(games_df)

    dc_result = dixon_coles.fit(train_df, massey_result)

    # Evaluation
    eval_report = eval_module.evaluate(dc_result, test_df, massey_result, teams_df)
    eval_module.save_eval_report(eval_report)

    # Build and save ratings.json
    ratings_json = build_ratings_json(teams_df, games_df, dc_result, massey_result,
                                       enrichment_coverage)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / "ratings.json").write_text(json.dumps(ratings_json, indent=2))
    log.info("saved ratings.json with %d teams", len(ratings_json["teams"]))

    # -----------------------------------------------------------------------
    # M6: Bracket simulation
    # -----------------------------------------------------------------------
    if not args.skip_sim:
        log.info("=== M6: Bracket ingest & simulation ===")
        try:
            brackets = ingest_all_brackets(teams_df)
            save_brackets(brackets)

            odds = sim_module.simulate_all_brackets(brackets, dc_result,
                                                     n_sims=args.n_sims)
            sim_module.save_bracket_odds(odds)
        except Exception as e:
            log.error("Bracket simulation failed: %s", e)
    else:
        log.info("Skipping bracket simulation (--skip-sim)")

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
