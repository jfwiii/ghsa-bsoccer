"""
MaxPreps scraper — M3.

URL structure as of 2026:
  Search:   https://www.maxpreps.com/search/?q={name}&sport=soccer&state=ga
  Schedule: https://www.maxpreps.com/ga/{city}/{school-slug}/soccer/spring/
  Detail:   https://www.maxpreps.com/games/{date}/{sport-slug}/{team1}-vs-{team2}.htm?c={id}

Responsibilities:
1. Slug discovery: build team_id → maxpreps_url_slug via search API + fuzzy match.
2. Per-team schedule fetch: parse __NEXT_DATA__ JSON for game records.
3. Per-game detail fetch for OT/PK games: parse SSR HTML boxscore table.
4. Non-GHSA opponent enrichment.
5. Graceful degradation: sustained failures abort MaxPreps enrichment cleanly.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

BASE_URL = "https://www.maxpreps.com"
GA_SCHOOL_SEARCH_URL = f"{BASE_URL}/search/?q={{query}}&sport=soccer&state=ga"
# /schedule/ path has full season; base path only has last ~3 games in wallCards
SCHEDULE_URL_TEMPLATE = f"{BASE_URL}/ga/{{slug}}/soccer/spring/schedule/"

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

FUZZY_THRESHOLD = 85
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

    def fetch_team_search(self, name: str) -> Optional[str]:
        query = urllib.parse.quote(name)
        url = GA_SCHOOL_SEARCH_URL.format(query=query)
        safe_name = re.sub(r'[^\w]', '_', name)
        cache_key = f"search_{safe_name}"
        return self._fetch(url, cache_key)

    def fetch_team_schedule(self, slug: str) -> Optional[str]:
        url = SCHEDULE_URL_TEMPLATE.format(slug=slug)
        return self._fetch(url, f"sched_{slug.replace('/', '_')}")

    def fetch_game_detail(self, game_url: str, contest_id: str) -> Optional[str]:
        safe_id = re.sub(r"[^\w\-]", "_", contest_id)
        return self._fetch(game_url, f"game_{safe_id}")


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


def _parse_search_results(html: str) -> list[tuple[str, str]]:
    """
    Parse __NEXT_DATA__ from a MaxPreps search page.
    Returns [(display_name, city/school-slug), ...].
    """
    soup = BeautifulSoup(html, "lxml")
    nd_tag = soup.find("script", id="__NEXT_DATA__")
    if not nd_tag:
        return []
    try:
        data = json.loads(nd_tag.string)
    except (json.JSONDecodeError, TypeError):
        return []

    results = data.get("props", {}).get("pageProps", {}).get("initialSchoolResults") or []
    entries = []
    for r in results:
        canonical = r.get("canonicalUrl", "")
        m = re.match(r"https?://www\.maxpreps\.com/ga/([\w-]+/[\w-]+)/?", canonical)
        if not m:
            continue
        slug = m.group(1)
        display = f"{r.get('name', '')} {r.get('mascot', '')}".strip()
        entries.append((display, slug))
    return entries


def discover_slugs(teams: list[dict], client: MaxPrepsClient) -> dict[int, Optional[str]]:
    """
    Build team_id → maxpreps_url_slug mapping.
    Uses manual overrides first, then search API with fuzzy matching per team.
    """
    str_overrides, _ = _load_overrides()
    overrides: dict[int, Optional[str]] = {int(k): v for k, v in str_overrides.items()}

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

        html = client.fetch_team_search(name)
        if not html:
            unmatched.append(tid)
            result[tid] = None
            continue

        entries = _parse_search_results(html)
        if not entries:
            log.info("no MaxPreps search results for team %d %r", tid, name)
            unmatched.append(tid)
            result[tid] = None
            continue

        if len(entries) == 1:
            result[tid] = entries[0][1]
            log.debug("search single match %r → %r", name, entries[0][1])
        else:
            mp_names = [e[0] for e in entries]
            match = process.extractOne(name, mp_names, scorer=fuzz.WRatio)
            if match and match[1] >= FUZZY_THRESHOLD:
                idx = mp_names.index(match[0])
                result[tid] = entries[idx][1]
                log.debug("search fuzzy match %r → %r (score=%d)", name, entries[idx][1], match[1])
            else:
                log.info("no confident match for team %d %r (best=%s)", tid, name, match)
                unmatched.append(tid)
                result[tid] = None

    _save_overrides(
        {str(k): v for k, v in {**overrides, **result}.items()},
        unmatched,
    )

    matched_count = sum(1 for v in result.values() if v is not None)
    log.info("slug discovery: %d/%d GHSA teams matched", matched_count, len(teams))
    return result


def parse_schedule_page(html: str) -> list[dict]:
    """
    Parse a MaxPreps team schedule page (/soccer/spring/schedule/).
    Extracts games from __NEXT_DATA__ JSON (props.pageProps.contests).

    Contest structure: each entry is a JSON array (Python list). Integer indices:
      c[1]  = contestId (UUID str)
      c[4]  = hasResult (bool)
      c[11] = timestamp (ISO str)
      c[35] = detail URL (str, optional)
      c[37] = page's team data array
      c[38] = opponent's team data array

    Team array positional layout:
      [3]  = score string "W 1-0" or "L 2-2" (regulation score; SO games show tied scores)
      [5]  = result "W" | "L" | None
      [6]  = goals scored (int)
      [11] = homeAwayType 0=home, 1=away
      [13] = teamCanonicalUrl (used to extract opponent slug)
      [14] = schoolName

    SO detection: MaxPreps shows regulation scores, so equal scores with a W/L result → shootout.
    OT (without SO) is not reliably detectable from schedule data.
    """
    soup = BeautifulSoup(html, "lxml")
    nd_tag = soup.find("script", id="__NEXT_DATA__")
    if not nd_tag:
        log.warning("parse_schedule_page: no __NEXT_DATA__ found")
        return []
    try:
        data = json.loads(nd_tag.string)
    except (json.JSONDecodeError, TypeError):
        return []

    contests = (
        data.get("props", {})
        .get("pageProps", {})
        .get("contests") or []
    )

    games = []
    for c in contests:
        if not isinstance(c, list) or len(c) < 39:
            continue
        if not c[4]:  # hasResult
            continue

        contest_id = c[1] or ""
        timestamp = c[11] or ""

        try:
            game_date = datetime.fromisoformat(timestamp).date()
        except (ValueError, TypeError):
            continue

        my_team = c[37] if isinstance(c[37], list) else []
        opp_team = c[38] if isinstance(c[38], list) else []
        if len(my_team) < 15 or len(opp_team) < 15:
            continue

        my_result = my_team[5] if len(my_team) > 5 else None   # "W" | "L"
        my_goals = my_team[6] if len(my_team) > 6 else None
        my_ha = my_team[11] if len(my_team) > 11 else -1        # 0=home, 1=away
        opp_goals = opp_team[6] if len(opp_team) > 6 else None
        opp_name = opp_team[14] if len(opp_team) > 14 else ""
        opp_canonical = opp_team[13] if len(opp_team) > 13 else ""

        is_home = (my_ha == 0)

        # MaxPreps stores regulation scores. Equal scores with a result = SO game.
        went_to_shootout = (
            my_result in ("W", "L")
            and my_goals is not None
            and opp_goals is not None
            and my_goals == opp_goals
        )

        opp_slug = None
        if isinstance(opp_canonical, str):
            m = re.match(r"https?://www\.maxpreps\.com/ga/([\w-]+/[\w-]+)/", opp_canonical)
            if m:
                opp_slug = m.group(1)

        # Game detail URL at contest index 35
        detail_url = None
        for detail_idx in (35, 18):
            val = c[detail_idx] if len(c) > detail_idx else None
            if val and isinstance(val, str) and "/games/" in val:
                detail_url = val
                break

        games.append({
            "date": game_date,
            "opponent_name": opp_name,
            "opponent_slug": opp_slug,
            "result": my_result,
            "my_score": my_goals,
            "opp_score": opp_goals,
            "is_home": is_home,
            "went_to_overtime": False,
            "went_to_shootout": went_to_shootout,
            "detail_url": detail_url,
            "contest_id": contest_id,
        })

    return games


def parse_game_detail(html: str) -> dict:
    """
    Parse a MaxPreps game detail page (SSR HTML, no __NEXT_DATA__).
    Finds the boxscore table (class='boxscore') to extract period scores.

    Table row order: row 0 = away team, row 1 = home team.
    Column classes: 'firsthalf', 'secondhalf', 'overtime*', 'shootout*', 'total score'.
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

    table = soup.find("table", class_="boxscore")
    if not table:
        return result

    # Detect OT/SO from header cell classes
    for th in table.find_all("th"):
        cls = " ".join(th.get("class") or []).lower()
        if "overtime" in cls:
            result["went_to_overtime"] = True
        if "shootout" in cls:
            result["went_to_shootout"] = True

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    if len(rows) < 2:
        return result

    away_row, home_row = rows[0], rows[1]

    def _cell_score(row, class_substr: str) -> Optional[int]:
        for cell in row.find_all("td"):
            classes_str = " ".join(cell.get("class") or [])
            if class_substr in classes_str:
                t = cell.get_text(strip=True)
                if t.isdigit():
                    return int(t)
        return None

    result["h1_away"] = _cell_score(away_row, "firsthalf")
    result["h1_home"] = _cell_score(home_row, "firsthalf")

    if result["went_to_shootout"]:
        final_away = _cell_score(away_row, "total score")
        final_home = _cell_score(home_row, "total score")
        if final_away is not None and final_home is not None:
            # Winner has phantom SO goal in final; subtract it to get regulation score
            if final_home > final_away:
                result["reg_home"] = final_home - 1
                result["reg_away"] = final_away
            else:
                result["reg_home"] = final_home
                result["reg_away"] = final_away - 1

    return result


def enrich_games(
    games_df,  # pandas DataFrame
    slug_map: dict[int, Optional[str]],
    client: MaxPrepsClient,
    fetch_details: bool = False,
) -> list[dict]:
    """
    For each game in games_df, fetch MaxPreps enrichment where useful.
    Returns list of enrichment dicts keyed by game_id.

    fetch_details=False (default): schedule-only enrichment — fast, identifies
    non-GHSA opponents and provides OT/SO flags from schedule data.
    fetch_details=True: also fetches per-game detail pages for halftime scores
    and regulation-score correction on SO games. Much slower (~2s per game).
    """
    enrichments: list[dict] = []

    # Cache schedule fetches to avoid re-fetching the same team
    team_schedules: dict[str, list[dict]] = {}

    def get_schedule(slug: str) -> list[dict]:
        if slug not in team_schedules:
            html = client.fetch_team_schedule(slug)
            team_schedules[slug] = parse_schedule_page(html) if html else []
        return team_schedules[slug]

    fetched_details: set[str] = set()

    for _, row in games_df.iterrows():
        game_id = row["game_id"]
        enrichment = {"game_id": game_id, "maxpreps_matched": False}

        for tid in [row["home_team_id"], row["away_team_id"]]:
            slug = slug_map.get(int(tid))
            if not slug:
                continue

            schedule = get_schedule(slug)
            mp_game = _match_mp_game(row, schedule)
            if not mp_game:
                continue

            enrichment["maxpreps_matched"] = True
            enrichment["went_to_overtime"] = mp_game.get("went_to_overtime", False)
            enrichment["went_to_shootout"] = mp_game.get("went_to_shootout", False)

            if fetch_details:
                detail_url = mp_game.get("detail_url")
                contest_id = mp_game.get("contest_id", "")
                if detail_url and contest_id and contest_id not in fetched_details:
                    detail_html = client.fetch_game_detail(detail_url, contest_id)
                    if detail_html:
                        fetched_details.add(contest_id)
                        parsed = parse_game_detail(detail_html)
                        enrichment["went_to_overtime"] = (
                            parsed["went_to_overtime"] or enrichment.get("went_to_overtime", False)
                        )
                        enrichment["went_to_shootout"] = (
                            parsed["went_to_shootout"] or enrichment.get("went_to_shootout", False)
                        )
                        enrichment["h1_home"] = parsed["h1_home"]
                        enrichment["h1_away"] = parsed["h1_away"]
                        if parsed["reg_home"] is not None:
                            enrichment["reg_home"] = parsed["reg_home"]
                            enrichment["reg_away"] = parsed["reg_away"]
            break

        enrichments.append(enrichment)

    return enrichments


def _match_mp_game(row, mp_schedule: list[dict]) -> Optional[dict]:
    """Match a games_df row to a MaxPreps schedule entry by date."""
    game_date = row["date"]
    if hasattr(game_date, "date"):
        game_date = game_date.date()

    for mp in mp_schedule:
        if mp.get("date") == game_date:
            return mp
    return None
