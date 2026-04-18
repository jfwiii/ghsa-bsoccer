# GHSA Boys Soccer — Development Log

## Recently Completed

| Item | Description |
|------|-------------|
| Score discrepancy noise (bad9ce1f) | Root cause: GHSA page lists Glynn vs Effingham 3/27 twice with scores 8–2 and 5–2. Fix: `_parse_schedule_rows` deduplicates by game pair within each section; normalizer suppresses repeat warnings for the same game ID. Stored score (8–2, first-seen) is correct. |
| Netlify build hook | Secret `NETLIFY_BUILD_HOOK` configured in GitHub → pipeline-triggered deploys now auto-redeploy Netlify. |
| Rankings contextual rank | When class/region filter is active, far-left rank column shows rank-within-subset; OVR rank shown as small secondary text. Header updates to "Rank" vs "OVR Rank". |
| Brackets tab actual draw | Fetches `public/brackets.json`; shows R1 matchup cards with team names, region·seed, R1 win probability (= R16 probability from simulation), and championship odds. Falls back to probability table if bracket structure unavailable. |
| R32 hidden from team pop-out | Bracket odds section no longer shows R32 = 100% row (filter in `openTeamPanel`). |
| Region rank fix | `team_class_region` is now built before `region_rank`, so the reconciled region from the alignment scrape is used. Teams previously showing `—` for region rank should now populate. |
| `brackets.json` written to `public/` | `save_brackets` now writes to both `data/brackets.json` and `public/brackets.json` so the UI can fetch it. |
| NW Whitfield name override | Added `"NW Whitfield": 211` to `pipeline/name_overrides.json` — was incorrectly resolving to Southeast Whitfield (243). |
| Methodology tab refresh | Added sections for Non-GHSA games, Region Seeding, updated Bracket Simulation (vectorized MC, analytical win prob, home advantage), and Impact Score formula. |
| Simulation odds fix | `triu`/`tril` were swapped in `_win_prob` — every win probability was returning P(away wins) as P(home wins), fully inverting all bracket odds |
| Simulation pair-ordering fix | `np.unique` on 2D arrays sorts rows lexicographically, reordering `(top_id, bot_id)` pairs; replaced with int64 scalar encoding to preserve slot semantics |
| Vectorized simulation | Replaced 200k-iteration Python loop with per-matchup `rng.random(n_sims) < p_win`; 8 brackets now complete in ~2s vs ~90s |
| Home/away assignment | Simulation now assigns home advantage to the higher-rated team (rounds 1–4); championship is neutral. Was: always gave home to top bracket slot regardless of rating |
| Non-GHSA records | Normalizer was skipping games where opponent couldn't be resolved. Now assigns stable synthetic negative IDs so non-GHSA results count in team records |
| Region standings sort | Seeding and display now sort by region W-L record first, PSR rank second, DC rating third (was: PSR rank only) |
| Impact score redesign | Now shows `(outcome − p_win) × 10` in range [−10, +10]. Positive = exceeded expectation, negative = underperformed. Was: raw opponent quality [0, 10] |
| Team panel UX | Widened to `min(660px, 95vw)` on desktop; added H/A/N column to schedule table |
| Netlify deploy reliability | Added build hook trigger step to CI workflow. |
| Cache-Control headers | `index.html` and JSON data files set to `no-cache, must-revalidate` in `netlify.toml` |

---

## Pending — Priority Order

### 1. MaxPreps URL update (blocker)
**Detail:** MaxPreps changed their URL structure — the old `/ga/{slug}/soccer/spring/schedule/` and GA rankings paths now return 404. The main site loads (200), so the site is up but the scraper's URL patterns need to be discovered and updated. Once fixed, Option A (committed cache) is implemented: the pipeline saves enrichment output to `pipeline/maxpreps_enrichment.json` on local runs, CI loads it without re-scraping.  
**Files:** `pipeline/scrape_maxpreps.py` — `GA_SOCCER_RANKINGS_URL`, `fetch_team_schedule()`, `_parse_ga_rankings_page()`; `scripts/run_full_pipeline.py` — Option A save/load already in place.

### 2. Region tiebreaker rules (bug)
**Detail:** Current seeding uses win%, then win-loss diff, then PSR rank, then DC rating. GHSA's actual tiebreaker order ([source](https://www.ghsa.net/constitution-section-2025-2026-soccer)) is:
1. Region W-L record
2. Head-to-head record between tied teams
3. Goals allowed in head-to-head games (tied teams only)
4. Goal differential in head-to-head games, capped at 3 goals per game
5. Goals allowed in all region games
6. Goal differential in all region games, capped at 3 goals per game
7. If still tied, revert to step 2 (head-to-head) and continue

Steps 2–7 require comparing subsets of tied teams iteratively, which is a non-trivial sort. The current fallback to PSR rank / DC rating is a reasonable approximation but does not match official GHSA seeding.  
**Note:** PK wins count as a one-goal win for the winner; loser receives no additional goal.  
**Files:** `scripts/run_full_pipeline.py` — `_region_sort()` and surrounding region seeding logic.

### 3. Brackets tab: visual bracket tree
**Detail:** The brackets tab currently shows R1 matchup cards (pairs), but not the connected bracket structure showing which winners would meet in R2, QF, SF, and F. Need to render a proper bracket tree: left column = R1 matchups, right columns = winner slots advancing toward the championship. Win probabilities per round already exist in `bracket_odds.json`.  
**Files:** `public/index.html` — `renderBrackets()`; `public/brackets.json` already has full round structure.

---

## Known Issues / Bugs

_(No open bugs.)_

---

## Future Season (2026–2027 school year)

### 7A Classification and New Region Alignments
**Detail:** GHSA is adding a 7A (AAAAAAA) classification and reshuffling all regions for 2026–2028. New alignments are published at the [GHSA reclassification PDF](https://www.ghsa.net/sites/default/files/documents/reclassification/GHSA_Proposed_Region_Alignment_for_2026-2028_11-10-2025.pdf).  
**Impact:** Region alignment scraper URL, class normalization map, bracket structure, and all display labels will need updating. The PDF should be reviewed before the 2026–2027 season to confirm final assignments.

### Prior-Season Ratings as Bayesian Prior
**Detail:** Early in the 2026–2027 season, Dixon-Coles will have few games to fit on. Using the 2025–2026 final ratings as an informative prior (e.g., shrinking new α/β estimates toward last year's values, weighted by a prior strength that decays as new games accumulate) would produce more stable early-season ratings.  
**Files:** `model/dixon_coles.py` — model fitting; `public/ratings.json` from the prior season would need to be stored and loaded.

### Playoff Probability Tab (New Format)
**Detail:** Starting 2026–2027, only region champions are guaranteed a state playoff seed; remaining spots are filled by PSR ranking across non-champions. A new "Playoff Probability" tab should simulate the remaining regular season to estimate each team's probability of: (a) winning their region, (b) qualifying via PSR wild card, (c) reaching each playoff round.  
**Requirements:**
- Full remaining-schedule data (scrape future game dates and opponents)
- Simulate remaining games using current DC ratings to get a distribution of final region standings and PSR ranks
- Integrate with existing bracket simulation for playoff-round probabilities
- Display as a sortable table: team, P(region champ), P(PSR qualifier), P(make playoffs), P(each round)  
**Files:** New module `model/playoff_prob.py`; new tab in `public/index.html`; new output file `public/playoff_odds.json`.
