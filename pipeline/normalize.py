"""
Normalizer — M3.

Takes raw GHSA game dicts + MaxPreps enrichment and produces:
  data/teams.parquet
  data/games.parquet

Key responsibilities:
- Resolve opponent name strings to team_ids (hyperlink first, fuzzy fallback).
- Deduplicate games (canonical home/away ordering by lower team_id).
- Merge MaxPreps enrichment fields.
- Compute duration (observed_duration_min, imputed_duration_min, duration_used_min).
- Compute home_goals_regulation / away_goals_regulation.
- Flag score discrepancies between GHSA and MaxPreps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

OVERRIDES_PATH = Path("pipeline/name_overrides.json")
DATA_DIR = Path("data")

FUZZY_THRESHOLD_OPPONENT = 92


# ---------------------------------------------------------------------------
# Duration helpers (§7.4, §7.5)
# ---------------------------------------------------------------------------

def impute_duration_minutes(mov: int) -> int:
    """Fallback duration when halftime scores unavailable. See design §7.4."""
    if mov <= 6:
        return 80
    if mov == 7:
        return 70
    if mov == 8:
        return 65
    if mov == 9:
        return 60
    return 50  # mov >= 10


def observed_duration_minutes(h1_home: int, h1_away: int,
                               final_home: int, final_away: int) -> int:
    """Derive game duration from halftime scores. See design §7.5."""
    h1_mov = abs(h1_home - h1_away)
    if h1_mov >= 10:
        return 40
    if h1_mov >= 7:
        return 60
    return 80


# ---------------------------------------------------------------------------
# Name override loading
# ---------------------------------------------------------------------------

def _load_name_overrides() -> dict[str, int]:
    if not OVERRIDES_PATH.exists():
        return {}
    data = json.loads(OVERRIDES_PATH.read_text())
    return data.get("overrides", {})


# ---------------------------------------------------------------------------
# Game ID
# ---------------------------------------------------------------------------

def _game_id(game_date: date, team_a: int, team_b: int) -> str:
    lo, hi = min(team_a, team_b), max(team_a, team_b)
    raw = f"{game_date.isoformat()}:{lo}:{hi}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main normalization
# ---------------------------------------------------------------------------

class Normalizer:
    def __init__(self, teams: list[dict], raw_games: list[dict],
                 enrichments: Optional[list[dict]] = None,
                 no_maxpreps: bool = False):
        self.teams = teams
        self.raw_games = raw_games
        self.enrichments = enrichments or []
        self.no_maxpreps = no_maxpreps
        self._name_overrides = _load_name_overrides()
        self._team_by_id: dict[int, dict] = {t["team_id"]: t for t in teams}
        self._team_names: list[str] = [t.get("name", "") for t in teams]
        self._team_ids: list[int] = [t["team_id"] for t in teams]

    def _resolve_team_id(self, name: str, known_id: Optional[int]) -> Optional[int]:
        """Resolve opponent name to a team_id."""
        if known_id is not None:
            return known_id

        # Manual override
        if name in self._name_overrides:
            return self._name_overrides[name]

        # Fuzzy match
        if self._team_names:
            match = process.extractOne(name, self._team_names, scorer=fuzz.WRatio)
            if match and match[1] >= FUZZY_THRESHOLD_OPPONENT:
                idx = self._team_names.index(match[0])
                return self._team_ids[idx]

        return None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run full normalization. Returns (teams_df, games_df)."""
        teams_df = self._build_teams_df()
        games_df = self._build_games_df()
        return teams_df, games_df

    def _build_teams_df(self) -> pd.DataFrame:
        rows = []
        for t in self.teams:
            row = {
                "team_id": t["team_id"],
                "name": t.get("name", ""),
                "class": t.get("class", ""),
                "region_or_area": t.get("region_or_area"),
                "structure_type": "area" if "Area" in (t.get("region_or_area") or "") else "region",
                "is_private": t.get("class", "") == "Private",
                "is_ghsa": t["team_id"] > 0,
                "maxpreps_url_slug": t.get("maxpreps_url_slug"),
                "maxpreps_ranking": t.get("maxpreps_ranking"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_games_df(self) -> pd.DataFrame:
        # Build enrichment lookup by game_id
        enrich_map: dict[str, dict] = {}
        for e in self.enrichments:
            enrich_map[e["game_id"]] = e

        # Deduplicate raw games
        seen: dict[str, dict] = {}
        discrepancies: list[dict] = []

        for g in self.raw_games:
            # Resolve opponent
            opp_id = self._resolve_team_id(
                g.get("opponent_name_raw", ""),
                g.get("opponent_team_id_raw")
            )

            # Determine canonical home/away ordering
            home_id = g.get("home_team_id")
            away_id = g.get("away_team_id")

            # If either is None, use reporting team + resolved opponent
            reporter = g.get("reporting_team_id")
            if home_id is None or away_id is None:
                if g.get("is_home", True):
                    home_id = reporter
                    away_id = opp_id
                else:
                    home_id = opp_id
                    away_id = reporter

            if home_id is None or away_id is None:
                continue  # cannot resolve; skip

            game_date = g["date"]
            gid = _game_id(game_date, home_id, away_id)

            if gid not in seen:
                seen[gid] = {
                    "game_id": gid,
                    "date": game_date,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_goals": g["home_goals"],
                    "away_goals": g["away_goals"],
                    "neutral_site": g.get("neutral_site", False),
                    "is_non_ghsa": g.get("is_non_ghsa", False),
                    "_sources": 1,
                }
            else:
                # Cross-check for within-GHSA discrepancy
                existing = seen[gid]
                if (existing["home_goals"] != g["home_goals"] or
                        existing["away_goals"] != g["away_goals"]):
                    discrepancies.append({
                        "game_id": gid,
                        "source_a": (existing["home_goals"], existing["away_goals"]),
                        "source_b": (g["home_goals"], g["away_goals"]),
                        "type": "within_ghsa",
                    })
                    log.warning(
                        "within-GHSA score discrepancy %s: %s vs %s",
                        gid, existing["home_goals"], g["home_goals"]
                    )
                seen[gid]["_sources"] = existing.get("_sources", 1) + 1

        # Build final game rows with enrichment
        rows = []
        score_discrepancy_count = 0
        for gid, g in seen.items():
            e = enrich_map.get(gid, {}) if not self.no_maxpreps else {}

            home_goals = g["home_goals"]
            away_goals = g["away_goals"]
            mov = abs(home_goals - away_goals)

            # Regulation scores
            went_to_so = e.get("went_to_shootout") or False
            if went_to_so and e.get("reg_home") is not None:
                home_goals_reg = e["reg_home"]
                away_goals_reg = e["reg_away"]
            else:
                home_goals_reg = home_goals
                away_goals_reg = away_goals

            reg_mov = abs(home_goals_reg - away_goals_reg)

            # Halftime scores
            h1_home = e.get("h1_home") if not self.no_maxpreps else None
            h1_away = e.get("h1_away") if not self.no_maxpreps else None

            # Duration
            obs_dur = None
            if h1_home is not None and h1_away is not None:
                obs_dur = observed_duration_minutes(h1_home, h1_away, home_goals_reg, away_goals_reg)
            imp_dur = impute_duration_minutes(mov)
            dur_used = obs_dur if obs_dur is not None else imp_dur

            # MaxPreps score discrepancy check
            mp_matched = e.get("maxpreps_matched", False) and not self.no_maxpreps
            score_disc = False
            if mp_matched and e.get("mp_home_goals") is not None:
                if (e["mp_home_goals"] != home_goals or e["mp_away_goals"] != away_goals):
                    score_disc = True
                    score_discrepancy_count += 1
                    log.warning("GHSA/MaxPreps score discrepancy game %s", gid)

            rows.append({
                "game_id": gid,
                "date": g["date"],
                "home_team_id": g["home_team_id"],
                "away_team_id": g["away_team_id"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_goals_regulation": home_goals_reg,
                "away_goals_regulation": away_goals_reg,
                "went_to_overtime": e.get("went_to_overtime"),
                "went_to_shootout": e.get("went_to_shootout"),
                "home_goals_half1": h1_home,
                "away_goals_half1": h1_away,
                "neutral_site": g.get("neutral_site", False),
                "is_non_ghsa": g.get("is_non_ghsa", False),
                "imputed_duration_min": imp_dur,
                "observed_duration_min": obs_dur,
                "duration_used_min": dur_used,
                "maxpreps_matched": mp_matched,
                "score_discrepancy": score_disc,
            })

        if discrepancies:
            log.warning("within-GHSA discrepancies: %d", len(discrepancies))

        log.info(
            "normalize: %d deduplicated games; %d score discrepancies logged",
            len(rows), score_discrepancy_count
        )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def save_parquets(teams_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    teams_df.to_parquet(DATA_DIR / "teams.parquet", index=False)
    games_df.to_parquet(DATA_DIR / "games.parquet", index=False)
    log.info("saved teams.parquet (%d rows) and games.parquet (%d rows)",
             len(teams_df), len(games_df))


def load_parquets() -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = pd.read_parquet(DATA_DIR / "teams.parquet")
    games = pd.read_parquet(DATA_DIR / "games.parquet")
    return teams, games
