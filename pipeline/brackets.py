"""
Bracket ingest — M6 prep.

Fetches all 8 GHSA playoff bracket CSVs and normalizes them to brackets.json.
Team names in the CSV are resolved to team_ids via the normalizer's fuzzy matching.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

BASE_URL = "https://www.ghsa.net"
DATA_DIR = Path("data")

HEADERS = {
    "User-Agent": (
        "GHSABSoccerRatings/0.1 (jfwright3i@gmail.com; research/ratings; respectful-bot)"
    )
}

# All 8 bracket HTML paths (relative to BASE_URL)
BRACKET_PATHS = {
    "AAAAAA":  "/2025-2026-ghsa-class-aaaaaa-boys-state-soccer-championship-bracket",
    "AAAAA":   "/2025-2026-ghsa-class-aaaaa-boys-state-soccer-championship-bracket",
    "AAAA":    "/2025-2026-ghsa-class-aaaa-boys-state-soccer-championship-bracket",
    "AAA":     "/2025-2026-ghsa-class-aaa-boys-state-soccer-championship-bracket",
    "AA":      "/2025-2026-ghsa-class-aa-boys-state-soccer-championship-bracket",
    "A DI":    "/2025-2026-ghsa-class-a-division-i-boys-state-soccer-championship-bracket",
    "A DII":   "/2025-2026-ghsa-class-a-division-ii-boys-state-soccer-championship-bracket",
    "Private": "/2025-2026-ghsa-private-boys-state-soccer-championship-bracket",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _fetch(session: requests.Session, url: str, delay: float = 1.0) -> str:
    time.sleep(delay)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _extract_node_id(html: str) -> Optional[str]:
    """Extract node_id from the bracket CSV export link."""
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=re.compile(r"/node/(\d+)/bracket-export-csv")):
        m = re.search(r"/node/(\d+)/", a["href"])
        if m:
            return m.group(1)
    # Also try extracting from page path metadata
    m = re.search(r"/node/(\d+)/bracket-export-csv", html)
    if m:
        return m.group(1)
    return None


def _fetch_bracket_text(session: requests.Session, node_id: str) -> str:
    """Fetch raw bracket export text (GHSA uses plaintext, not a real CSV)."""
    url = f"{BASE_URL}/node/{node_id}/bracket-export-csv"
    return _fetch(session, url)


# Round header → sequential round number. Named rounds come after numbered rounds.
_NAMED_ROUND_MAP = {
    "quarter-finals": "QF", "quarterfinals": "QF", "quarter finals": "QF",
    "semi-finals": "SF", "semifinals": "SF", "semi finals": "SF",
    "finals": "F", "final": "F", "championship": "F",
}


def _parse_bracket_text(text: str, bracket_name: str,
                         team_lookup: dict[str, int]) -> dict:
    """
    Parse GHSA bracket export text.

    Actual format (confirmed from live pages):
      Title line
      [blank]
      Round 1
      TeamA, TeamB
      TeamC, TeamD
      ...
      [blank]
      Round 2
      [blank or filled matchups]
      ...
      Quarter-Finals
      Semi-Finals
      Finals
    """
    rounds: dict = {}     # round_num → list of matchup dicts
    named_order: list[str] = []  # order in which named rounds appear
    current_round: Optional[int | str] = None  # int for numbered, str tag for named

    for line_raw in text.splitlines():
        line = line_raw.strip()
        if not line:
            continue

        lower = line.lower()

        # Numbered round: "Round N"
        m = re.match(r"^round\s+(\d+)$", lower)
        if m:
            current_round = int(m.group(1))
            if current_round not in rounds:
                rounds[current_round] = []
            continue

        # Named round
        tag = _NAMED_ROUND_MAP.get(lower.rstrip("-").strip())
        if tag:
            if tag not in rounds:
                rounds[tag] = []
                named_order.append(tag)
            current_round = tag
            continue

        # Skip title line (no comma → likely title or empty round header)
        if current_round is None:
            continue

        # Parse "TeamA, TeamB" — use csv.reader to handle quoted commas in names
        try:
            parts = next(csv.reader([line]))
        except Exception:
            continue

        if len(parts) < 2:
            continue

        top_name = parts[0].strip()
        bot_name = parts[1].strip()
        if not top_name or not bot_name:
            continue

        top_id = _resolve_name(top_name, team_lookup)
        bot_id = _resolve_name(bot_name, team_lookup)

        rnd = rounds[current_round]
        rnd.append({
            "position": len(rnd) + 1,
            "top_team_id": top_id,
            "bottom_team_id": bot_id,
            "top_seed": None,
            "bottom_seed": None,
            "host_team_id": top_id,  # first-listed team is home
            "top_name_raw": top_name,
            "bot_name_raw": bot_name,
        })

    # Assign sequential integers to named rounds (after any numbered rounds)
    max_numbered = max((k for k in rounds if isinstance(k, int)), default=0)
    name_to_num = {tag: max_numbered + i + 1 for i, tag in enumerate(named_order)}
    normalized: dict[int, list] = {}
    for k, v in rounds.items():
        num = k if isinstance(k, int) else name_to_num[k]
        normalized[num] = v

    r1_matchups = normalized.get(1, [])
    n_r1 = len(r1_matchups)
    seeds = _seed_table(n_r1)
    for m in r1_matchups:
        top_s, bot_s = seeds.get(m["position"], (None, None))
        m["top_seed"] = top_s
        m["bottom_seed"] = bot_s

    all_rounds = _build_bracket_tree(r1_matchups, n_r1)

    # Merge in actual results from later rounds in GHSA export
    for rn in sorted(normalized):
        if rn == 1 or not normalized[rn]:
            continue
        for rd in all_rounds:
            if rd["round"] == rn:
                for i, src in enumerate(normalized[rn]):
                    if i < len(rd["matchups"]):
                        stub = rd["matchups"][i]
                        stub["top_team_id"] = src.get("top_team_id")
                        stub["bottom_team_id"] = src.get("bottom_team_id")
                        stub["top_name_raw"] = src.get("top_name_raw")
                        stub["bot_name_raw"] = src.get("bot_name_raw")
                break

    return {"bracket": bracket_name, "rounds": all_rounds}


def _parse_int(s) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


# GHSA bracket export uses consistent abbreviations that differ from full team names.
# Expand them before fuzzy matching so exact/near-exact matches dominate.
_ABBREV_EXPANSIONS: list[tuple[str, str]] = [
    (r'^"',          ""),          # strip leading curly/straight quote artifact
    (r'\bCo\b',      "County"),    # "Colquitt Co" → "Colquitt County"
    (r'\bMtn\b',     "Mountain"),  # "Kennesaw Mtn" → "Kennesaw Mountain"
    (r'\bBr\b',      "Branch"),    # "Flowery Br" → "Flowery Branch"
    (r'\bRdg\b',     "Ridge"),     # "Peachtree Rdg" → "Peachtree Ridge"
    (r'^E\s+',       "East "),     # "E Coweta" → "East Coweta"
    (r'^N\s+',       "North "),    # "N Clayton" → "North Clayton"
    (r'^S\s+',       "South "),    # "S Forsyth" → "South Forsyth"
    (r'^W\s+',       "West "),     # "W Hall" → "West Hall"
    (r',Con\b',      ", Conyers"), # "Heritage,Con" → "Heritage, Conyers"
    (r',Cat\b',      ", Catoosa"), # "Heritage,Cat" → "Heritage, Catoosa"
    (r',Aug\b',      ", Augusta"), # "Johnson,Aug" → "Johnson, Augusta"
    (r',Sav\b',      ", Savannah"),# "Johnson,Sav" → "Johnson, Savannah"
    (r',Gain\b',     ", Gainesville"),
    (r'\bWash\b',    "Washington"),  # "Wash-Wilkes" → "Washington-Wilkes"
    (r'^GACS$',       "Greater Atlanta Christian"),  # Private bracket abbreviation
    # Private school abbreviations used in GHSA bracket CSV
    (r'\bChr\b',      "Christian"),  # "Providence Chr" → "Providence Christian"
    (r'\bAcad\b',     "Academy"),    # "Whitefield Acad" → "Whitefield Academy"
    (r'\bAve\b',      "Avenue"),     # "Prince Ave Chr" → "Prince Avenue Christian"
    (r'^Mt\b',        "Mt."),        # "Mt Bethel" → "Mt. Bethel"
    (r'^N Cobb\b',    "North Cobb"), # "N Cobb Chr" → "North Cobb Christian"
    (r'^Sav Cntry Day$', "Savannah Country Day"),
    (r'^Sav Chr\b',   "Savannah Christian"),
    (r'^Atlanta Int\b', "Atlanta International"),
    (r'^Christian Her\b', "Christian Heritage"),
    (r'^Mt Vernon\b', "Mount Vernon"),
]

# Standard bracket seeding pairings: position → (top/away seed, bottom/home seed).
# Top team (first in CSV) is away (lower seed); bottom is home (higher seed).
_R1_SEEDS_32: dict[int, tuple] = {
    1: (32, 1),  2: (17, 16), 3: (24, 9),  4: (25, 8),
    5: (28, 5),  6: (21, 12), 7: (29, 4),  8: (20, 13),
    9: (31, 2),  10: (18, 15), 11: (23, 10), 12: (26, 7),
    13: (30, 3), 14: (19, 14), 15: (27, 6), 16: (22, 11),
}
_R1_SEEDS_16: dict[int, tuple] = {
    1: (16, 1), 2: (9, 8),  3: (12, 5), 4: (13, 4),
    5: (15, 2), 6: (10, 7), 7: (14, 3), 8: (11, 6),
}
# Private (29 teams): seeds 1–3 have byes; 13 actual R1 games map to
# full-bracket positions 2,3,4,5,6,7,8,10,11,12,14,15,16.
_R1_SEEDS_PRIVATE: dict[int, tuple] = {
    1: (17, 16), 2: (24, 9),  3: (25, 8),  4: (28, 5),
    5: (21, 12), 6: (29, 4),  7: (20, 13), 8: (18, 15),
    9: (23, 10), 10: (26, 7), 11: (19, 14), 12: (27, 6),
    13: (22, 11),
}

_ROUND_NAMES = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]


def _seed_table(n_r1: int) -> dict:
    if n_r1 == 16:
        return _R1_SEEDS_32
    if n_r1 == 8:
        return _R1_SEEDS_16
    if n_r1 == 13:
        return _R1_SEEDS_PRIVATE
    return {}


def _build_bracket_tree(r1_matchups: list, n_r1: int) -> list:
    """Generate full bracket tree from R1 matchups: returns list of round dicts."""
    r1_label_idx = {16: 0, 13: 0, 8: 1}.get(n_r1, 0)

    # Assign game numbers and feeds_into for R1
    for m in r1_matchups:
        m["game"] = m["position"]
    # R2 game offset: first R2 game = n_r1 + 1
    r2_base = n_r1 + 1
    for m in r1_matchups:
        m["feeds_into"] = r2_base + (m["position"] - 1) // 2

    rounds = [{"round": 1, "round_name": _ROUND_NAMES[r1_label_idx], "matchups": list(r1_matchups)}]

    prev = r1_matchups
    game_counter = n_r1 + 1
    round_num = 2
    name_idx = r1_label_idx + 1

    while len(prev) > 1:
        next_matchups = []
        for i in range(0, len(prev), 2):
            top_g, bot_g = prev[i], prev[i + 1] if i + 1 < len(prev) else prev[i]
            stub = {
                "game": game_counter,
                "position": len(next_matchups) + 1,
                "top_from_game": top_g["game"],
                "bottom_from_game": bot_g["game"],
                "top_team_id": None, "bottom_team_id": None,
                "top_seed": None, "bottom_seed": None,
                "top_name_raw": None, "bot_name_raw": None,
            }
            next_matchups.append(stub)
            game_counter += 1

        if len(next_matchups) > 1:
            next_base = game_counter
            for k, m in enumerate(next_matchups):
                m["feeds_into"] = next_base + k // 2

        label = _ROUND_NAMES[name_idx] if name_idx < len(_ROUND_NAMES) else f"Round {round_num}"
        rounds.append({"round": round_num, "round_name": label, "matchups": next_matchups})
        prev = next_matchups
        round_num += 1
        name_idx += 1

    return rounds


def _expand_bracket_name(name: str) -> str:
    for pattern, replacement in _ABBREV_EXPANSIONS:
        name = re.sub(pattern, replacement, name)
    return name.strip()


def _resolve_name(name: str, lookup: dict[str, int]) -> Optional[int]:
    """Resolve a bracket team name to team_id.

    Strategy:
    1. Exact match on raw name.
    2. Expand known GHSA bracket abbreviations, exact match on expanded name.
    3. Fuzzy match on expanded name (threshold 80 — safe after expansion).
    """
    name = name.strip()
    if not name:
        return None
    if name in lookup:
        return lookup[name]

    expanded = _expand_bracket_name(name)
    if expanded in lookup:
        return lookup[expanded]

    names = list(lookup.keys())
    match = process.extractOne(expanded, names, scorer=fuzz.WRatio)
    if match and match[1] >= 80:
        return lookup[match[0]]

    log.warning("bracket: could not resolve team name %r (expanded: %r)", name, expanded)
    return None


def _build_team_lookup(teams_df, class_filter: Optional[str] = None) -> dict[str, int]:
    """Build name → team_id lookup from teams DataFrame.

    class_filter restricts matches to teams in that classification, preventing
    same-named teams in different classes from cross-contaminating brackets.
    """
    lookup = {}
    for _, row in teams_df.iterrows():
        if class_filter:
            if class_filter == "Private":
                if "(Private)" not in row.get("name", ""):
                    continue
            elif row.get("class", "") != class_filter:
                continue
        name = row.get("name", "")
        if name:
            tid = int(row["team_id"])
            lookup[name] = tid
            # Also index without parenthetical suffix, e.g. "Foo (Private)" → "Foo"
            stripped = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
            if stripped and stripped != name:
                lookup.setdefault(stripped, tid)
    return lookup


def ingest_all_brackets(teams_df, session: Optional[requests.Session] = None) -> list[dict]:
    """
    Fetch and parse all 8 GHSA playoff brackets.
    Returns list of bracket dicts for brackets.json.
    """
    if session is None:
        session = _session()

    brackets = []
    needs_manual = []

    for bracket_name, path in BRACKET_PATHS.items():
        # Scope name resolution to this bracket's class only — prevents teams
        # with identical short names (Drew, Jackson) from landing in the wrong bracket.
        team_lookup = _build_team_lookup(teams_df, class_filter=bracket_name)
        url = BASE_URL + path
        try:
            log.info("fetching bracket page: %s", bracket_name)
            html = _fetch(session, url)
            node_id = _extract_node_id(html)
            if not node_id:
                log.warning("could not find node_id for bracket %s at %s", bracket_name, url)
                needs_manual.append(bracket_name)
                continue

            text = _fetch_bracket_text(session, node_id)
            if not text.strip():
                log.warning("empty export for bracket %s (node %s)", bracket_name, node_id)
                needs_manual.append(bracket_name)
                continue

            bracket = _parse_bracket_text(text, bracket_name, team_lookup)

            # Flag unresolved teams
            for rnd in bracket["rounds"]:
                for m in rnd["matchups"]:
                    if m["top_team_id"] is None and m["top_name_raw"]:
                        needs_manual.append(f"{bracket_name}: {m['top_name_raw']}")
                    if m["bottom_team_id"] is None and m["bot_name_raw"]:
                        needs_manual.append(f"{bracket_name}: {m['bot_name_raw']}")

            brackets.append(bracket)
            log.info("bracket %s: %d rounds", bracket_name, len(bracket["rounds"]))

        except requests.HTTPError as e:
            log.error("HTTP error fetching bracket %s: %s", bracket_name, e)
        except Exception as e:
            log.error("error ingesting bracket %s: %s", bracket_name, e, exc_info=True)

    if needs_manual:
        log.warning("bracket entries needing manual reconciliation:\n%s",
                    "\n".join(f"  {x}" for x in needs_manual))

    return brackets


def save_brackets(brackets: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "brackets.json"
    path.write_text(json.dumps(brackets, indent=2, default=str))
    log.info("saved brackets.json with %d brackets", len(brackets))
    pub = Path("public")
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "brackets.json").write_text(json.dumps(brackets, indent=2, default=str))
    log.info("saved public/brackets.json")


def load_brackets() -> list[dict]:
    path = DATA_DIR / "brackets.json"
    return json.loads(path.read_text())
