"""
GHSA scraper — M1.

Actual page structure (confirmed from live pages):

Index page (https://www.ghsa.net/schedule-ranking-view/boys-soccer):
  - Single <table class="table tabled-bordered">
  - Section headers: <td colspan="7"><strong><u>AAAAAA</u></strong></td>
  - Team rows:  PSR_rank | School | Record | WP | OWP | OOWP | PSR | <a>explain</a>
    The link "explain" points to /rankings/{team_id}.

Team "Explain" page (/rankings/{team_id}):
  - Multiple schedule tables, each with:
      <tr class="schedule-head"><td><strong><a href="/rankings/{id}">Name</a><span>CLASS</span></strong></td></tr>
      <tr class="schedule-detail"><td class="schedule-view-date">M/D</td>
                                   <td class="schedule-view-opponent">At? Opp (Neutral)?</td>
                                   <td>X - Y</td>
                                   <td>●W or ●L</td></tr>
  - The FIRST schedule table is the main team's own schedule.
  - Subsequent tables are opponents' full schedules (for OWP calculation).
  - schedule-head links give name → team_id for each embedded opponent.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE_URL = "https://www.ghsa.net"
INDEX_URL = f"{BASE_URL}/schedule-ranking-view/boys-soccer"
TEAM_URL = f"{BASE_URL}/schedule-ranking-view/boys-soccer/rankings/{{team_id}}"

CACHE_DIR = Path("data/raw/ghsa/teams")
CACHE_TTL_REGULAR = timedelta(hours=24)
CACHE_TTL_PLAYOFFS = timedelta(hours=4)

HEADERS = {
    "User-Agent": (
        "GHSABSoccerRatings/0.1 (jfwright3i@gmail.com; "
        "research/ratings; respectful-bot)"
    )
}

# Canonical class names
CLASS_SECTION_MAP = {
    "aaaaaa": "AAAAAA",
    "aaaaa": "AAAAA",
    "aaaa": "AAAA",
    "aaa": "AAA",
    "aa": "AA",
    "a division i": "A DI",
    "a division ii": "A DII",
    "private boys": "Private",
    "private": "Private",
}

# Approximate soccer season year (all regular season games are in this year)
SEASON_YEAR = 2026


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _cache_path(team_id: int, fetch_date: date) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{team_id}_{fetch_date.strftime('%Y%m%d')}.html"


def _is_stale(path: Path, playoffs: bool = False) -> bool:
    if not path.exists():
        return True
    ttl = CACHE_TTL_PLAYOFFS if playoffs else CACHE_TTL_REGULAR
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > ttl


def _fetch_html(session: requests.Session, url: str, delay: float = 1.0) -> str:
    time.sleep(delay)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _get_team_html(session: requests.Session, team_id: int,
                   playoffs: bool = False) -> str:
    today = date.today()
    cache = _cache_path(team_id, today)
    if _is_stale(cache, playoffs):
        url = TEAM_URL.format(team_id=team_id)
        html = _fetch_html(session, url)
        cache.write_text(html, encoding="utf-8")
        log.debug("fetched team %d", team_id)
    else:
        html = cache.read_text(encoding="utf-8")
        log.debug("cache hit team %d", team_id)
    return html


def _normalize_class(raw: str) -> str:
    s = raw.strip()
    key = s.lower()
    if key in CLASS_SECTION_MAP:
        return CLASS_SECTION_MAP[key]
    # Try without spaces/hyphens
    key2 = key.replace("-", " ").replace("_", " ")
    return CLASS_SECTION_MAP.get(key2, s)


def _parse_date_md(s: str) -> Optional[date]:
    """Parse 'M/D' format into a date using the season year."""
    s = s.strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})$", s)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    # Boys soccer: Feb–May → all 2026
    # Preseason (Aug–Dec) would be 2025 — uncommon for GHSA soccer
    year = SEASON_YEAR if month <= 7 else SEASON_YEAR - 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _game_id(game_date: date, team_a: int, team_b: int) -> str:
    lo, hi = min(team_a, team_b), max(team_a, team_b)
    raw = f"{game_date.isoformat()}:{lo}:{hi}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def scrape_index(session: requests.Session) -> list[dict]:
    """
    Fetch the PSR index page.
    Returns list of {team_id, name, class, psr_rank, record}.
    """
    html = _fetch_html(session, INDEX_URL)
    soup = BeautifulSoup(html, "lxml")
    teams = []
    current_class = ""

    table = soup.find("table", class_="table")
    if not table:
        log.error("index: could not find main table")
        return teams

    for tr in table.find_all("tr"):
        # Section header row: <td colspan="7"><strong><u>AAAAAA</u></strong></td>
        tds = tr.find_all("td")
        if len(tds) == 1:
            colspan = tds[0].get("colspan", "1")
            if int(colspan) >= 6:
                text = tds[0].get_text(strip=True)
                current_class = _normalize_class(text)
                continue

        # Team row: PSR_rank | School | Record | WP | OWP | OOWP | PSR | explain
        if len(tds) < 7:
            continue

        # Find the "explain" link → team_id
        link = tr.find("a", href=re.compile(r"/rankings/\d+"))
        if not link:
            continue
        m = re.search(r"/rankings/(\d+)", link["href"])
        if not m:
            continue
        team_id = int(m.group(1))

        try:
            psr_rank = int(tds[0].get_text(strip=True))
        except ValueError:
            psr_rank = None

        name = tds[1].get_text(strip=True)
        record = tds[2].get_text(strip=True)

        teams.append({
            "team_id": team_id,
            "name": name,
            "class": current_class,
            "psr_rank": psr_rank,
            "record": record,
        })

    log.info("index: found %d teams", len(teams))
    return teams


# ---------------------------------------------------------------------------
# Team page
# ---------------------------------------------------------------------------

def parse_team_page(html: str, home_team_id: int,
                    index_id_map: Optional[dict[str, int]] = None) -> tuple[dict, list[dict]]:
    """
    Parse a GHSA "explain ranking" team page.

    Returns (team_meta, raw_games).

    team_meta keys: team_id, name, class, region_or_area
    raw_game keys: game_id, date, home_team_id, away_team_id, home_goals, away_goals,
                   reporting_team_id, opponent_name_raw, opponent_team_id_raw,
                   neutral_site, is_non_ghsa
    """
    soup = BeautifulSoup(html, "lxml")
    team_meta: dict = {"team_id": home_team_id}

    # Build a local name → team_id mapping from schedule-head links on this page
    local_id_map: dict[str, int] = dict(index_id_map or {})
    for a in soup.find_all("a", href=re.compile(r"/rankings/\d+")):
        m = re.search(r"/rankings/(\d+)", a["href"])
        if m:
            opp_name = a.get_text(strip=True)
            if opp_name:
                local_id_map[opp_name] = int(m.group(1))

    # Find schedule tables by locating each schedule-head and walking up to its table.
    # This avoids double-counting when opponent tables are wrapped in outer <table> elements.
    all_games: list[dict] = []
    first_table = True
    processed_table_ids: set[int] = set()

    for head in soup.find_all("tr", class_="schedule-head"):
        table = head.find_parent("table")
        if table is None or id(table) in processed_table_ids:
            continue
        processed_table_ids.add(id(table))

        # Only include schedule-detail rows that belong directly to this table
        # (not from nested tables inside it)
        detail_rows = [
            tr for tr in table.find_all("tr", class_="schedule-detail")
            if tr.find_parent("table") is table
        ]
        if not detail_rows:
            first_table = False
            continue

        # Get team identity from schedule-head
        if head:
            strong = head.find("strong")
            if strong:
                # Class is in a float:right span
                span = strong.find("span")
                cls_text = span.get_text(strip=True) if span else ""
                if span:
                    span.extract()
                name_text = strong.get_text(strip=True)

                if first_table:
                    # This is the main team's table
                    team_meta["name"] = name_text
                    team_meta["class"] = _normalize_class(cls_text)
                    # Check for region in surrounding context
                    region = _extract_region(soup)
                    if region:
                        team_meta["region_or_area"] = region
                    reporting_id = home_team_id
                else:
                    # Opponent's table — get their team_id
                    link = head.find("a", href=re.compile(r"/rankings/\d+"))
                    if link:
                        m = re.search(r"/rankings/(\d+)", link["href"])
                        reporting_id = int(m.group(1)) if m else None
                    else:
                        reporting_id = local_id_map.get(name_text)
                    if reporting_id is None:
                        first_table = False
                        continue
        else:
            first_table = False
            continue

        games = _parse_schedule_rows(detail_rows, reporting_id, local_id_map)
        all_games.extend(games)
        first_table = False

    return team_meta, all_games


def _extract_region(soup: BeautifulSoup) -> Optional[str]:
    """Try to extract region/area from page text."""
    for text in soup.stripped_strings:
        m = re.match(r"^(\d+-[A-Z]{2,}(?:\s+Area\s+\d+)?)$", text)
        if m:
            return m.group(1)
    return None


def _parse_schedule_rows(rows, reporting_id: int,
                          local_id_map: dict[str, int]) -> list[dict]:
    games = []
    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue

        # Date: schedule-view-date cell
        date_td = tr.find("td", class_="schedule-view-date")
        date_text = date_td.get_text(strip=True) if date_td else (cells[0].get_text(strip=True))
        game_date = _parse_date_md(date_text)
        if not game_date:
            continue

        # Opponent: schedule-view-opponent cell
        opp_td = tr.find("td", class_="schedule-view-opponent")
        opp_text = opp_td.get_text(strip=True) if opp_td else cells[1].get_text(strip=True)

        # Home/away/neutral from opponent text prefix
        is_away = opp_text.startswith("At ") or opp_text.startswith("at ")
        neutral = re.search(r"\(neutral\)", opp_text, re.I) is not None
        opp_name = re.sub(r"^[Aa]t\s+", "", opp_text)
        opp_name = re.sub(r"\s*\(neutral\)", "", opp_name, flags=re.I).strip()

        # Non-GHSA detection
        is_non_ghsa = re.search(r"\(non-ghsa\)", opp_text, re.I) is not None

        # Score: "X - Y" where X = reporting team, Y = opponent
        score_td = cells[2]
        score_text = score_td.get_text(strip=True)
        score_m = re.search(r"(\d+)\s*[-–]\s*(\d+)", score_text)
        if not score_m:
            continue
        my_goals, opp_goals = int(score_m.group(1)), int(score_m.group(2))

        # Resolve opponent team_id
        opp_id = local_id_map.get(opp_name)

        # Determine home/away (from reporting team's perspective)
        if neutral:
            # Canonical ordering: lower team_id is "home" so both reporters agree
            if opp_id is not None and reporting_id > opp_id:
                home_id = opp_id
                away_id = reporting_id
                home_goals = opp_goals
                away_goals = my_goals
            else:
                home_id = reporting_id
                away_id = opp_id
                home_goals = my_goals
                away_goals = opp_goals
        elif is_away:
            # Reporting team is away, opponent is home
            home_id = opp_id
            away_id = reporting_id
            home_goals = opp_goals
            away_goals = my_goals
        else:
            # Reporting team is home
            home_id = reporting_id
            away_id = opp_id
            home_goals = my_goals
            away_goals = opp_goals

        gid = _game_id(game_date, reporting_id, opp_id or 0)

        games.append({
            "game_id": gid,
            "date": game_date,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "reporting_team_id": reporting_id,
            "opponent_name_raw": opp_name,
            "opponent_team_id_raw": opp_id,
            "neutral_site": neutral,
            "is_non_ghsa": is_non_ghsa,
        })

    return games


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scrape_all_teams(playoffs: bool = False,
                     team_ids: Optional[list[int]] = None) -> tuple[list[dict], list[dict]]:
    """
    Fetch index + all team pages.
    Returns (teams, raw_games).
    """
    session = _session()

    index_teams = scrape_index(session)
    # Build name → team_id for fuzzy resolution during page parsing
    index_id_map: dict[str, int] = {t["name"]: t["team_id"] for t in index_teams}

    if team_ids is not None:
        fetch_list = [t for t in index_teams if t["team_id"] in team_ids]
        found_ids = {t["team_id"] for t in fetch_list}
        for tid in team_ids:
            if tid not in found_ids:
                fetch_list.append({"team_id": tid, "name": "", "class": "", "psr_rank": None})
    else:
        fetch_list = index_teams

    all_teams: dict[int, dict] = {t["team_id"]: t for t in index_teams}
    all_games: list[dict] = []

    for i, entry in enumerate(fetch_list):
        tid = entry["team_id"]
        try:
            html = _get_team_html(session, tid, playoffs=playoffs)
            meta, games = parse_team_page(html, tid, index_id_map)

            # Merge index metadata with page metadata (page wins on name/class if present)
            merged = {**entry}
            for k, v in meta.items():
                if v:
                    merged[k] = v
            all_teams[tid] = merged
            all_games.extend(games)

            if (i + 1) % 50 == 0:
                log.info("progress: %d/%d teams fetched, %d game records so far",
                         i + 1, len(fetch_list), len(all_games))

        except requests.HTTPError as e:
            log.warning("HTTP error fetching team %d: %s", tid, e)
        except Exception as e:
            log.warning("error parsing team %d: %s", tid, e, exc_info=True)

    log.info("scrape complete: %d teams, %d raw game records", len(all_teams), len(all_games))
    return list(all_teams.values()), all_games
