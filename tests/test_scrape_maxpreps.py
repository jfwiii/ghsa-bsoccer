"""Snapshot-parse tests for MaxPreps scraper (M3 format)."""

import json
from datetime import date
from pipeline.scrape_maxpreps import parse_schedule_page, parse_game_detail


# Schedule fixture uses __NEXT_DATA__ JSON (M3 format)
_SCHEDULE_DATA = {
    "props": {
        "pageProps": {
            "wallCards": {
                "schedule": {
                    "data": [
                        {
                            "contestId": "abc123-001",
                            "hasResult": True,
                            "canonicalUrl": "https://www.maxpreps.com/games/02-01-2026/soccer-spring-26/lambert-vs-walton.htm?c=abc",
                            "timestamp": "2026-02-01T20:00:00",
                            "homeAwayType": 0,
                            "teams": [
                                {
                                    "schoolName": "Walton",
                                    "score": 3,
                                    "result": "W",
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/marietta/walton-raiders/soccer/spring/",
                                },
                                {
                                    "schoolName": "Lambert",
                                    "score": 1,
                                    "result": "L",
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/suwanee/lambert-longhorns/soccer/spring/",
                                },
                            ],
                        },
                        {
                            "contestId": "def456-002",
                            "hasResult": True,
                            "canonicalUrl": "https://www.maxpreps.com/games/02-08-2026/soccer-spring-26/walton-vs-lassiter.htm?c=def",
                            "timestamp": "2026-02-08T19:00:00",
                            "homeAwayType": 1,
                            "teams": [
                                {
                                    "schoolName": "Walton",
                                    "score": 2,
                                    "result": "L",
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/marietta/walton-raiders/soccer/spring/",
                                },
                                {
                                    "schoolName": "Lassiter",
                                    "score": 3,
                                    "result": "W",
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/marietta/lassiter-trojans/soccer/spring/",
                                },
                            ],
                        },
                        {
                            "contestId": "ghi789-003",
                            "hasResult": True,
                            "canonicalUrl": "https://www.maxpreps.com/games/03-15-2026/soccer-spring-26/pebblebrook-vs-walton.htm?c=ghi",
                            "timestamp": "2026-03-15T18:00:00",
                            "homeAwayType": 0,
                            "teams": [
                                {
                                    "schoolName": "Walton",
                                    "score": 1,
                                    "result": "W",
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/marietta/walton-raiders/soccer/spring/",
                                },
                                {
                                    "schoolName": "Pebblebrook",
                                    "score": 0,
                                    "result": "L",
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/mableton/pebblebrook-knights/soccer/spring/",
                                },
                            ],
                        },
                        {
                            "contestId": "future-004",
                            "hasResult": False,
                            "canonicalUrl": "https://www.maxpreps.com/games/04-01-2026/soccer-spring-26/walton-vs-north-cobb.htm?c=xyz",
                            "timestamp": "2026-04-01T19:00:00",
                            "homeAwayType": 1,
                            "teams": [
                                {
                                    "schoolName": "Walton",
                                    "score": None,
                                    "result": None,
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/marietta/walton-raiders/soccer/spring/",
                                },
                                {
                                    "schoolName": "North Cobb",
                                    "score": None,
                                    "result": None,
                                    "teamCanonicalUrl": "https://www.maxpreps.com/ga/kennesaw/north-cobb-warriors/soccer/spring/",
                                },
                            ],
                        },
                    ]
                }
            }
        }
    }
}

FIXTURE_SCHEDULE = (
    f'<html><body><script id="__NEXT_DATA__" type="application/json">'
    f"{json.dumps(_SCHEDULE_DATA)}"
    f"</script></body></html>"
)


# Game detail fixtures use boxscore table with CSS class structure (M3 format)
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
    # 4 entries but 1 has hasResult=False → 3 completed games
    assert len(games) == 3


def test_parse_schedule_date_parsed():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    assert games[0]["date"] == date(2026, 2, 1)


def test_parse_schedule_home_away_flag():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    home_game = next(g for g in games if g["date"] == date(2026, 2, 1))
    away_game = next(g for g in games if g["date"] == date(2026, 2, 8))
    assert home_game["is_home"] is True   # homeAwayType 0 = home
    assert away_game["is_home"] is False  # homeAwayType 1 = away


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
