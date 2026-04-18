"""
MaxPreps scraper — M2.

Responsibilities:
1. Slug discovery: build team_id → maxpreps_url_slug via fuzzy match.
2. Per-team schedule fetch: extract OT/shootout indicators and game hashes.
3. Per-game detail fetch for PK-decided games: extract regulation scores.
4. Non-GHSA opponent enrichment.
5. Graceful degradation: any sustained failure aborts MaxPreps enrichment
   cleanly, allowing pipeline to continue with GHSA-only data.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

BASE_URL = "https://www.maxpreps.com"
GA_SOCCER_RANKINGS_URL = f"{BASE_URL}/ga/soccer/rankings/"

CACHE_DIR = Path("data/raw/maxpreps")
CACHE_TTL_REGULAR = timedelta(hours=24)
CACHE_TTL_PLAYOFFS = timedelta(hours=4)

OVERRIDES_PATH = Path("pipeline/maxpreps_slug_overrides.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FUZZY_THRESHOLD = 90
MAX_CONSECUTIVE_FAILURES = 5
REQUEST_DELAY = 2.0  # seconds


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", key)
    return CACHE_DIR / f"{safe}.html"


def _is_stale(path: Path, playoffs: bool = False) -> bool:
    if not path.exists():
        return True
    ttl = CACHE_TTL_PLAYOFFS if playoffs else CACHE_TTL_REGULAR
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > ttl


class MaxPrepsAbort(Exception):
    """Raised when we hit sustained MaxPreps failures and must abort enrichment."""


class MaxPrepsClient:
    def __init__(self, playoffs: bool = False):
        self.session = _session()
        self.playoffs = playoffs
        self._consecutive_failures = 0

    def _fetch(self, url: str, cache_key: str) -> Optional[str]:
        path = _cache_path(cache_key)
        if not _is_stale(path, self.playoffs):
            return path.read_text(encoding="utf-8")

        time.sleep(REQUEST_DELAY)
        delay = REQUEST_DELAY
        for attempt in range(5):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    self._consecutive_failures = 0
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_text(resp.text, encoding="utf-8")
                    return resp.text
                else:
                    log.warning("MaxPreps HTTP %d for %s (attempt %d)", resp.status_code, url, attempt + 1)
                    self._consecutive_failures += 1
            except requests.RequestException as e:
                log.warning("MaxPreps request error %s (attempt %d): %s", url, attempt + 1, e)
                self._consecutive_failures += 1

            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise MaxPrepsAbort(
                    f"MaxPreps aborted after {self._consecutive_failures} consecutive failures"
                )

            time.sleep(delay)
            delay = min(delay * 2, 60)

        return None

    def fetch_ga_rankings_page(self, page: int = 1) -> Optional[str]:
        url = f"{GA_SOCCER_RANKINGS_URL}?page={page}"
        return self._fetch(url, f"ga_rankings_p{page}")

    def fetch_team_schedule(self, slug: str) -> Optional[str]:
        url = f"{BASE_URL}/ga/{slug}/soccer/spring/schedule/"
        return self._fetch(url, f"sched_{slug.replace('/', '_')}")

    def fetch_game_detail(self, game_url: str, game_hash: str) -> Optional[str]:
        return self._fetch(game_url, f"game_{game_hash}")


def _load_overrides() -> tuple[dict[str, Optional[str]], list[int]]:
    if not OVERRIDES_PATH.exists():
        return {}, []
    data = json.loads(OVERRIDES_PATH.read_text())
    return data.get("overrides", {}), data.get("unmatched", [])


def _save_overrides(overrides: dict, unmatched: list[int]) -> None:
    existing = {}
    if OVERRIDES_PATH.exists():
        existing = json.loads(OVERRIDES_PATH.read_text())
    existing["overrides"] = overrides
    existing["unmatched"] = unmatched
    OVERRIDES_PATH.write_text(json.dumps(existing, indent=2))


def discover_slugs(teams: list[dict], client: MaxPrepsClient) -> dict[int, Optional[str]]:
    """
    Build team_id → maxpreps_url_slug mapping.
    Uses manual overrides first, then fuzzy match from MaxPreps GA rankings pages.
    """
    str_overrides, _ = _load_overrides()
    # Convert string keys to int
    overrides: dict[int, Optional[str]] = {int(k): v for k, v in str_overrides.items()}

    # Collect (name, slug) from MaxPreps GA rankings pages
    mp_entries: list[tuple[str, str]] = []  # (display_name, slug)
    page = 1
    while True:
        html = client.fetch_ga_rankings_page(page)
        if not html:
            break
        entries, has_next = _parse_ga_rankings_page(html)
        mp_entries.extend(entries)
        if not has_next or page >= 20:
            break
        page += 1

    log.info("MaxPreps GA rankings: collected %d team entries", len(mp_entries))

    mp_names = [e[0] for e in mp_entries]
    result: dict[int, Optional[str]] = {}
    unmatched: list[int] = []

    for team in teams:
        tid = team["team_id"]
        if tid in overrides:
            result[tid] = overrides[tid]
            continue

        name = team.get("name", "")
        if not name:
            unmatched.append(tid)
            continue

        match = process.extractOne(name, mp_names, scorer=fuzz.WRatio)
        if match and match[1] >= FUZZY_THRESHOLD:
            idx = mp_names.index(match[0])
            result[tid] = mp_entries[idx][1]
            log.debug("matched %r → %r (score=%d)", name, mp_entries[idx][1], match[1])
        else:
            log.info("no MaxPreps match for team %d %r (best=%s)", tid, name, match)
            unmatched.append(tid)
            result[tid] = None

    _save_overrides(
        {str(k): v for k, v in {**overrides, **result}.items()},
        unmatched
    )

    matched_count = sum(1 for v in result.values() if v is not None)
    log.info("slug discovery: %d/%d GHSA teams matched", matched_count, len(teams))
    return result


def _parse_ga_rankings_page(html: str) -> tuple[list[tuple[str, str]], bool]:
    """
    Parse MaxPreps GA state rankings page.
    Returns ([(display_name, city/school-slug), ...], has_next_page).
    """
    soup = BeautifulSoup(html, "lxml")
    entries = []

    for a in soup.find_all("a", href=re.compile(r"/ga/[\w-]+/[\w-]+/soccer/spring/")):
        href = a["href"]
        m = re.match(r"/ga/([\w-]+/[\w-]+)/soccer/spring/", href)
        if not m:
            continue
        slug = m.group(1)
        name = a.get_text(strip=True)
        if name and slug:
            entries.append((name, slug))

    has_next = bool(soup.find("a", string=re.compile(r"next", re.I)))
    return entries, has_next


def parse_schedule_page(html: str) -> list[dict]:
    """
    Parse a MaxPreps team schedule page.
    Returns list of game dicts with fields relevant for enrichment.
    """
    soup = BeautifulSoup(html, "lxml")
    games = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue

        game = _parse_schedule_row(tr, cells)
        if game:
            games.append(game)

    return games


def _parse_schedule_row(tr, cells) -> Optional[dict]:
    row_text = tr.get_text(" ", strip=True)

    # Date — look for M/D/YYYY pattern
    date_m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", row_text)
    if not date_m:
        return None
    try:
        game_date = datetime.strptime(date_m.group(1), "%m/%d/%Y").date()
    except ValueError:
        return None

    # Detect OT/shootout
    went_to_ot = "(OT)" in row_text or "(2OT)" in row_text
    went_to_so = "SO" in row_text or re.search(r"\bSO\b", row_text) is not None

    # Score — "W X-Y" or "L X-Y" possibly with "(OT)"
    score_m = re.search(r"([WL])\s+(\d+)-(\d+)(?:\s*\(OT\))?(?:\s*\(SO\))?", row_text)
    if not score_m:
        return None
    result = score_m.group(1)
    score_a, score_b = int(score_m.group(2)), int(score_m.group(3))

    # Home/Away
    is_home = "@" not in row_text.split(date_m.group(1))[0]

    # Game detail URL
    detail_url = None
    detail_hash = None
    for a in tr.find_all("a", href=re.compile(r"/games/")):
        detail_url = a["href"]
        if not detail_url.startswith("http"):
            detail_url = f"https://www.maxpreps.com{detail_url}"
        h_m = re.search(r"c=([a-f0-9\-]+)", detail_url)
        if h_m:
            detail_hash = h_m.group(1)
        break

    # Opponent name
    opp_name = ""
    for cell in cells:
        text = cell.get_text(strip=True)
        if text and not re.search(r"^\d|^[WL]\s", text) and "/" not in text:
            opp_name = text
            break

    return {
        "date": game_date,
        "opponent_name": opp_name,
        "result": result,
        "score_a": score_a,  # winner's score
        "score_b": score_b,  # loser's score
        "is_home": is_home,
        "went_to_overtime": went_to_ot or went_to_so,
        "went_to_shootout": went_to_so,
        "detail_url": detail_url,
        "detail_hash": detail_hash,
    }


def parse_game_detail(html: str) -> dict:
    """
    Parse a MaxPreps game detail page to extract:
    - Halftime scores (h1_home, h1_away)
    - Regulation scores for PK games
    - Overtime / shootout confirmation
    """
    soup = BeautifulSoup(html, "lxml")
    result = {
        "h1_home": None,
        "h1_away": None,
        "reg_home": None,
        "reg_away": None,
        "went_to_overtime": False,
        "went_to_shootout": False,
    }

    # Look for score table with columns: 1 | 2 | OT1 | OT2 | Final | SO
    # or simpler: 1 | 2 | Final
    tables = soup.find_all("table")
    for table in tables:
        headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]
        if not any(h in ("1", "FINAL", "SO") for h in headers):
            continue

        rows = table.find_all("tr")
        if len(rows) < 3:
            continue

        # Two data rows: home team, away team
        home_cells = rows[1].find_all("td")
        away_cells = rows[2].find_all("td") if len(rows) > 2 else []

        def cell_val(cells, idx) -> Optional[int]:
            if idx < len(cells):
                t = cells[idx].get_text(strip=True)
                if t.isdigit():
                    return int(t)
            return None

        col_map = {h: i for i, h in enumerate(headers)}

        if "1" in col_map:
            result["h1_home"] = cell_val(home_cells, col_map["1"])
            result["h1_away"] = cell_val(away_cells, col_map["1"])

        if "SO" in col_map:
            result["went_to_shootout"] = True
            # Regulation score = Final - phantom goal for winner
            # MaxPreps shows reg score directly in some layouts
            final_col = col_map.get("FINAL", col_map.get("F"))
            if final_col is not None:
                fh = cell_val(home_cells, final_col)
                fa = cell_val(away_cells, final_col)
                if fh is not None and fa is not None:
                    # The SO column shows who won the shootout (1 = winner, 0 = loser)
                    so_col = col_map["SO"]
                    so_home = cell_val(home_cells, so_col)
                    # Regulation: subtract the phantom goal from winner
                    if so_home == 1:
                        result["reg_home"] = fh - 1
                        result["reg_away"] = fa
                    else:
                        result["reg_home"] = fh
                        result["reg_away"] = fa - 1

        if "OT1" in col_map or "OT" in col_map:
            result["went_to_overtime"] = True

        break

    # Also check for explicit "L X-X" regulation notation
    # MaxPreps sometimes shows "L 2-2" in result cells for PK losses
    for td in soup.find_all("td"):
        t = td.get_text(strip=True)
        m = re.match(r"[LW]\s+(\d+)-(\d+)", t)
        if m and result["reg_home"] is None:
            # This is a regulation-score indicator
            pass  # handled by table parsing above

    return result


def enrich_games(
    games_df,  # pandas DataFrame
    slug_map: dict[int, Optional[str]],
    client: MaxPrepsClient,
) -> list[dict]:
    """
    For each game in games_df, fetch MaxPreps enrichment where useful.
    Returns list of enrichment dicts keyed by game_id.
    """
    import pandas as pd

    enrichments: list[dict] = []

    # Group games by team to minimize fetches
    team_schedules: dict[str, list[dict]] = {}  # slug → parsed schedule rows

    def get_schedule(slug: str) -> list[dict]:
        if slug not in team_schedules:
            html = client.fetch_team_schedule(slug)
            team_schedules[slug] = parse_schedule_page(html) if html else []
        return team_schedules[slug]

    fetched_details: set[str] = set()

    for _, row in games_df.iterrows():
        game_id = row["game_id"]
        enrichment = {"game_id": game_id, "maxpreps_matched": False}

        # Try home team slug first, then away
        for tid in [row["home_team_id"], row["away_team_id"]]:
            slug = slug_map.get(int(tid))
            if not slug:
                continue

            schedule = get_schedule(slug)
            mp_game = _match_mp_game(row, schedule, int(tid))
            if not mp_game:
                continue

            enrichment["maxpreps_matched"] = True
            enrichment["went_to_overtime"] = mp_game.get("went_to_overtime")
            enrichment["went_to_shootout"] = mp_game.get("went_to_shootout")

            # Fetch detail page for shootout games or if we need halftime
            detail_url = mp_game.get("detail_url")
            detail_hash = mp_game.get("detail_hash")
            needs_detail = (
                mp_game.get("went_to_shootout") or
                (detail_url and game_id not in fetched_details)
            )

            if needs_detail and detail_url and detail_hash:
                detail = client.fetch_game_detail(detail_url, detail_hash)
                if detail:
                    fetched_details.add(game_id)
                    parsed = parse_game_detail(detail)
                    enrichment.update({
                        "h1_home": parsed["h1_home"],
                        "h1_away": parsed["h1_away"],
                        "went_to_overtime": parsed["went_to_overtime"] or enrichment.get("went_to_overtime"),
                        "went_to_shootout": parsed["went_to_shootout"] or enrichment.get("went_to_shootout"),
                    })
                    if parsed["reg_home"] is not None:
                        # Regulation scores — orient to home/away perspective of games_df
                        if int(tid) == int(row["home_team_id"]):
                            enrichment["reg_home"] = parsed["reg_home"]
                            enrichment["reg_away"] = parsed["reg_away"]
                        else:
                            enrichment["reg_home"] = parsed["reg_away"]
                            enrichment["reg_away"] = parsed["reg_home"]
            break

        enrichments.append(enrichment)

    return enrichments


def _match_mp_game(row, mp_schedule: list[dict], team_id: int) -> Optional[dict]:
    """Match a games_df row to a MaxPreps schedule entry by date."""
    game_date = row["date"]
    if hasattr(game_date, "date"):
        game_date = game_date.date()

    for mp in mp_schedule:
        mp_date = mp.get("date")
        if mp_date == game_date:
            return mp
    return None
