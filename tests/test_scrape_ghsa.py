"""Snapshot-parse tests for GHSA scraper."""

from datetime import date

from pipeline.scrape_ghsa import (
    parse_team_page,
    _parse_date_md,
    _normalize_class,
    scrape_index,
)


# Fixture matches real GHSA "explain ranking" page structure
FIXTURE_TEAM_PAGE = """
<html><body>
<table style="margin: 10px; width: auto;">
  <tr class="schedule-head">
    <td colspan="5"><strong>Walton<span style="float:right;">AAAAAA</span></strong></td>
  </tr>
  <tr class="schedule-detail schedule-winner">
    <td class="schedule-view-date">2/01</td>
    <td class="schedule-view-opponent">Lambert</td>
    <td>3 - 1</td>
    <td>&#9679;W</td>
  </tr>
  <tr class="schedule-detail">
    <td class="schedule-view-date">2/08</td>
    <td class="schedule-view-opponent">At Lassiter</td>
    <td>0 - 2</td>
    <td>&#9679;L</td>
  </tr>
  <tr class="schedule-detail schedule-winner">
    <td class="schedule-view-date">2/15</td>
    <td class="schedule-view-opponent">Pebblebrook (Neutral)</td>
    <td>2 - 1</td>
    <td>&#9679;W</td>
  </tr>
</table>

<table>
  <tr><td valign="top">
    <table style="margin: 10px; width: auto;">
      <tr class="schedule-head">
        <td colspan="5"><strong><a href="/schedule-ranking-view/boys-soccer/rankings/574">Lambert</a>
          <span style="float:right;">AAAAAA</span></strong></td>
      </tr>
      <tr class="schedule-detail schedule-winner">
        <td class="schedule-view-date">2/01</td>
        <td class="schedule-view-opponent">At Walton</td>
        <td>1 - 3</td>
        <td>&#9679;L</td>
      </tr>
    </table>
  </td></tr>
</table>
</body></html>
"""


def test_parse_team_page_game_count():
    _, games = parse_team_page(FIXTURE_TEAM_PAGE, 76)
    # 3 from Walton's table + 1 from Lambert's table
    assert len(games) == 4


def test_parse_team_page_home_game():
    _, games = parse_team_page(FIXTURE_TEAM_PAGE, 76)
    home_game = next(g for g in games if g["date"] == date(2026, 2, 1)
                     and g["reporting_team_id"] == 76)
    assert home_game["home_team_id"] == 76
    assert home_game["home_goals"] == 3
    assert home_game["away_goals"] == 1
    assert home_game["neutral_site"] is False


def test_parse_team_page_away_game():
    _, games = parse_team_page(FIXTURE_TEAM_PAGE, 76)
    away_game = next(g for g in games if g["date"] == date(2026, 2, 8))
    # Walton is away → Lassiter is home (opp_id unknown → None)
    assert away_game["away_team_id"] == 76
    assert away_game["away_goals"] == 0
    assert away_game["home_goals"] == 2


def test_parse_team_page_neutral():
    _, games = parse_team_page(FIXTURE_TEAM_PAGE, 76)
    neutral_game = next(g for g in games if g["date"] == date(2026, 2, 15))
    assert neutral_game["neutral_site"] is True


def test_parse_team_page_meta():
    meta, _ = parse_team_page(FIXTURE_TEAM_PAGE, 76)
    assert meta["team_id"] == 76
    assert meta["name"] == "Walton"
    assert meta["class"] == "AAAAAA"


def test_parse_team_page_opponent_id_resolved_from_link():
    """Lambert has a link → should be resolved to team_id 574."""
    _, games = parse_team_page(FIXTURE_TEAM_PAGE, 76)
    lambert_game = next(g for g in games if g["date"] == date(2026, 2, 1)
                        and g["reporting_team_id"] == 76)
    assert lambert_game["away_team_id"] == 574


def test_parse_date_md():
    assert _parse_date_md("2/09") == date(2026, 2, 9)
    assert _parse_date_md("4/17") == date(2026, 4, 17)
    assert _parse_date_md("12/01") == date(2025, 12, 1)
    assert _parse_date_md("garbage") is None


def test_normalize_class():
    assert _normalize_class("AAAAAA") == "AAAAAA"
    assert _normalize_class("A Division II") == "A DII"
    assert _normalize_class("A Division I") == "A DI"
    assert _normalize_class("Private") == "Private"
    assert _normalize_class("Private Boys") == "Private"
