"""Snapshot-parse tests for MaxPreps scraper (M3 format)."""

import json
from datetime import date
from pipeline.scrape_maxpreps import parse_schedule_page, parse_game_detail


def _team_array(
    name: str,
    city: str,
    goals: int | None,
    result: str | None,
    home_away: int,  # 0=home, 1=away
    slug: str,
    slot: int = 1,
) -> list:
    """Build a minimal positional team array matching MaxPreps /schedule/ format."""
    canonical = f"https://www.maxpreps.com/ga/{slug}/soccer/spring/"
    score_str = None
    if result and goals is not None:
        score_str = f"{result} {goals}-?"  # opponent goals don't matter for this field
    # Pad to at least 15 elements; key positions:
    #  0=teamSeasonId, 1=schoolId, 2=sportSeasonId, 3=score_str, 4=slot,
    #  5=result, 6=goals, 7-10=booleans, 11=homeAwayType, 12=num, 13=canonicalUrl, 14=name
    arr = [
        "00000000-0000-0000-0000-000000000000",  # 0 teamSeasonId
        "00000000-0000-0000-0000-000000000000",  # 1 schoolId
        "00000000-0000-0000-0000-000000000000",  # 2 sportSeasonId
        score_str,   # 3 score string
        slot,        # 4 slot
        result,      # 5 result
        goals,       # 6 goals
        False,       # 7
        False,       # 8
        False,       # 9
        False,       # 10
        home_away,   # 11 homeAwayType
        1,           # 12
        canonical,   # 13 teamCanonicalUrl
        name,        # 14 schoolName
        city,        # 15 city
    ]
    return arr


def _contest(
    contest_id: str,
    timestamp: str,
    has_result: bool,
    my_name: str,
    my_city: str,
    my_slug: str,
    my_goals: int | None,
    my_result: str | None,
    my_ha: int,
    opp_name: str,
    opp_city: str,
    opp_slug: str,
    opp_goals: int | None,
    opp_result: str | None,
    detail_url: str | None = None,
) -> list:
    """Build a minimal contest list matching MaxPreps /schedule/ __NEXT_DATA__ format.

    Contests are JSON arrays; integer indices matter:
      [1]=contestId, [4]=hasResult, [11]=timestamp,
      [35]=detailUrl, [37]=my_team, [38]=opp_team.
    """
    my_slot = 1
    opp_slot = 2
    my_team = _team_array(my_name, my_city, my_goals, my_result, my_ha, my_slug, my_slot)
    opp_ha = 1 - my_ha
    opp_team = _team_array(opp_name, opp_city, opp_goals, opp_result, opp_ha, opp_slug, opp_slot)
    c: list = [None] * 39
    c[1] = contest_id
    c[4] = has_result
    c[11] = timestamp
    c[35] = detail_url
    c[37] = my_team
    c[38] = opp_team
    return c


_CONTESTS = [
    _contest(
        "abc123-001", "2026-02-01T20:00:00", True,
        "Walton", "Marietta", "marietta/walton-raiders", 3, "W", 0,
        "Lambert", "Suwanee", "suwanee/lambert-longhorns", 1, "L",
        detail_url="https://www.maxpreps.com/games/02-01-2026/soccer-spring-26/lambert-vs-walton.htm?c=abc",
    ),
    _contest(
        "def456-002", "2026-02-08T19:00:00", True,
        "Walton", "Marietta", "marietta/walton-raiders", 2, "L", 1,
        "Lassiter", "Marietta", "marietta/lassiter-trojans", 3, "W",
    ),
    _contest(
        "ghi789-003", "2026-03-15T18:00:00", True,
        "Walton", "Marietta", "marietta/walton-raiders", 1, "W", 0,
        "Pebblebrook", "Mableton", "mableton/pebblebrook-knights", 0, "L",
    ),
    # SO game: equal scores (1-1) with a W result
    _contest(
        "so-004", "2026-03-20T18:00:00", True,
        "Walton", "Marietta", "marietta/walton-raiders", 1, "W", 0,
        "North Cobb", "Kennesaw", "kennesaw/north-cobb-warriors", 1, "L",
    ),
    # Future game: hasResult=False
    _contest(
        "future-005", "2026-04-01T19:00:00", False,
        "Walton", "Marietta", "marietta/walton-raiders", None, None, 1,
        "Woodstock", "Woodstock", "woodstock/woodstock-wolverines", None, None,
    ),
]

FIXTURE_SCHEDULE = (
    f'<html><body><script id="__NEXT_DATA__" type="application/json">'
    f'{json.dumps({"props": {"pageProps": {"contests": _CONTESTS}}})}'
    f"</script></body></html>"
)


# Game detail fixtures use boxscore table with CSS class structure
FIXTURE_GAME_DETAIL_PK = """
<html><body>
<table class="mx-grid boxscore d-b-s post soccer">
<thead>
<tr class="primary-header-row">
  <th class="team string first" scope="col"><span></span></th>
  <th class="firsthalf string score dw" scope="col" title="First Half"><span>1</span></th>
  <th class="secondhalf string score dw" scope="col" title="Second Half"><span>2</span></th>
  <th class="overtime string score dw" scope="col" title="Overtime"><span>OT</span></th>
  <th class="shootout string score dw" scope="col" title="Shootout"><span>SO</span></th>
  <th class="score string total score" scope="col" title="Final"><span>Final</span></th>
</tr>
</thead>
<tbody>
<tr class="first">
  <th class="team first" scope="row"><a href="/ga/mableton/pebblebrook-knights/soccer/spring/">Pebblebrook</a></th>
  <td class="firsthalf score dw">1</td>
  <td class="secondhalf score dw">1</td>
  <td class="overtime score dw">0</td>
  <td class="shootout score dw">0</td>
  <td class="score total score">2</td>
</tr>
<tr class="last alternate">
  <th class="team first" scope="row"><a href="/ga/marietta/walton-raiders/soccer/spring/">Walton</a></th>
  <td class="firsthalf score dw">1</td>
  <td class="secondhalf score dw">1</td>
  <td class="overtime score dw">0</td>
  <td class="shootout score dw">1</td>
  <td class="score total score">3</td>
</tr>
</tbody>
</table>
</body></html>
"""

FIXTURE_GAME_DETAIL_NORMAL = """
<html><body>
<table class="mx-grid boxscore d-b-s post soccer">
<thead>
<tr class="primary-header-row">
  <th class="team string first" scope="col"><span></span></th>
  <th class="firsthalf string score dw" scope="col" title="First Half"><span>1</span></th>
  <th class="secondhalf string score dw" scope="col" title="Second Half"><span>2</span></th>
  <th class="score string total score" scope="col" title="Final"><span>Final</span></th>
</tr>
</thead>
<tbody>
<tr class="first">
  <th class="team first" scope="row"><a href="/ga/suwanee/lambert-longhorns/soccer/spring/">Lambert</a></th>
  <td class="firsthalf score dw">0</td>
  <td class="secondhalf score dw">1</td>
  <td class="score total score">1</td>
</tr>
<tr class="last alternate">
  <th class="team first" scope="row"><a href="/ga/marietta/walton-raiders/soccer/spring/">Walton</a></th>
  <td class="firsthalf score dw">2</td>
  <td class="secondhalf score dw">1</td>
  <td class="score total score">3</td>
</tr>
</tbody>
</table>
</body></html>
"""


def test_parse_schedule_extracts_completed_games():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    # 5 entries, 1 has hasResult=False → 4 completed games
    assert len(games) == 4


def test_parse_schedule_date_parsed():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    assert games[0]["date"] == date(2026, 2, 1)


def test_parse_schedule_home_away_flag():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    home_game = next(g for g in games if g["date"] == date(2026, 2, 1))
    away_game = next(g for g in games if g["date"] == date(2026, 2, 8))
    assert home_game["is_home"] is True   # my_ha=0 → home
    assert away_game["is_home"] is False  # my_ha=1 → away


def test_parse_schedule_opponent_name():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    first = next(g for g in games if g["date"] == date(2026, 2, 1))
    assert first["opponent_name"] == "Lambert"


def test_parse_schedule_opponent_slug_extracted():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    first = next(g for g in games if g["date"] == date(2026, 2, 1))
    assert first["opponent_slug"] == "suwanee/lambert-longhorns"


def test_parse_schedule_detail_url_and_contest_id():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    first = next(g for g in games if g["date"] == date(2026, 2, 1))
    assert first["detail_url"] is not None
    assert "lambert-vs-walton" in first["detail_url"]
    assert first["contest_id"] == "abc123-001"


def test_parse_schedule_shootout_detected():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    so_game = next(g for g in games if g["date"] == date(2026, 3, 20))
    # Equal scores (1-1) with a W result → shootout
    assert so_game["went_to_shootout"] is True
    regular = next(g for g in games if g["date"] == date(2026, 2, 1))
    assert regular["went_to_shootout"] is False


def test_parse_game_detail_pk():
    detail = parse_game_detail(FIXTURE_GAME_DETAIL_PK)
    assert detail["went_to_shootout"] is True
    assert detail["went_to_overtime"] is True
    # away=Pebblebrook row0, home=Walton row1
    assert detail["h1_away"] == 1
    assert detail["h1_home"] == 1
    # Walton (home, final=3) won the shootout → reg_home = 3-1 = 2, reg_away = 2
    assert detail["reg_home"] == 2
    assert detail["reg_away"] == 2


def test_parse_game_detail_normal():
    detail = parse_game_detail(FIXTURE_GAME_DETAIL_NORMAL)
    assert detail["went_to_shootout"] is False
    assert detail["went_to_overtime"] is False
    # away=Lambert row0, home=Walton row1
    assert detail["h1_away"] == 0
    assert detail["h1_home"] == 2
