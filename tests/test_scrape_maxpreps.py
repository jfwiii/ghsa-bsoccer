"""Snapshot-parse tests for MaxPreps scraper."""

from datetime import date
from pipeline.scrape_maxpreps import parse_schedule_page, parse_game_detail


FIXTURE_SCHEDULE = """
<html><body>
<table>
<tr><td>02/01/2026</td><td>vs Lambert</td><td>W 3-1</td><td><a href="/games/02-01-2026/soccer-spring-25/walton-vs-lambert.htm?c=abc123">box</a></td></tr>
<tr><td>02/08/2026</td><td>@ Lassiter</td><td>L 2-3 (OT)</td><td><a href="/games/02-08-2026/soccer-spring-25/lassiter-vs-walton.htm?c=def456">box</a></td></tr>
<tr><td>03/15/2026</td><td>vs Pebblebrook</td><td>W 3-2 SO</td><td><a href="/games/03-15-2026/soccer-spring-25/walton-vs-pebblebrook.htm?c=ghi789">box</a></td></tr>
</table>
</body></html>
"""


FIXTURE_GAME_DETAIL_PK = """
<html><body>
<table>
<tr><th>Team</th><th>1</th><th>2</th><th>OT1</th><th>Final</th><th>SO</th></tr>
<tr><td>Walton</td><td>1</td><td>1</td><td>0</td><td>3</td><td>1</td></tr>
<tr><td>Pebblebrook</td><td>1</td><td>1</td><td>0</td><td>2</td><td>0</td></tr>
</table>
</body></html>
"""

FIXTURE_GAME_DETAIL_NORMAL = """
<html><body>
<table>
<tr><th>Team</th><th>1</th><th>2</th><th>Final</th></tr>
<tr><td>Walton</td><td>2</td><td>1</td><td>3</td></tr>
<tr><td>Lambert</td><td>0</td><td>1</td><td>1</td></tr>
</table>
</body></html>
"""


def test_parse_schedule_extracts_games():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    assert len(games) == 3


def test_parse_schedule_ot_flagged():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    ot_game = next(g for g in games if g["date"] == date(2026, 2, 8))
    assert ot_game["went_to_overtime"] is True


def test_parse_schedule_shootout_flagged():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    so_game = next(g for g in games if g["date"] == date(2026, 3, 15))
    assert so_game["went_to_shootout"] is True


def test_parse_schedule_detail_url_extracted():
    games = parse_schedule_page(FIXTURE_SCHEDULE)
    first = next(g for g in games if g["date"] == date(2026, 2, 1))
    assert first["detail_url"] is not None
    assert "abc123" in first["detail_url"]


def test_parse_game_detail_pk():
    detail = parse_game_detail(FIXTURE_GAME_DETAIL_PK)
    assert detail["went_to_shootout"] is True
    assert detail["h1_home"] == 1
    assert detail["h1_away"] == 1
    # Walton won SO → reg_home = Final - 1 = 2
    assert detail["reg_home"] == 2
    assert detail["reg_away"] == 2


def test_parse_game_detail_halftime():
    detail = parse_game_detail(FIXTURE_GAME_DETAIL_NORMAL)
    assert detail["h1_home"] == 2
    assert detail["h1_away"] == 0
