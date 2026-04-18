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
| Netlify deploy reliability | Added build hook trigger step to CI workflow. **Requires user setup** — see Known Issues |
| Cache-Control headers | `index.html` and JSON data files set to `no-cache, must-revalidate` in `netlify.toml` |

---

## Pending — Priority Order

### 1. MaxPreps in CI
**Detail:** MaxPreps enrichment (halftime scores, regulation scores for OT games) currently only runs locally. It's skipped in CI (`--no-maxpreps`) because the IP changes needed to avoid blocks haven't been solved. Halftime data improves duration estimation; regulation score separation fixes OT game distortion.  
**Files:** `scripts/refresh.py`, `pipeline/scrape_maxpreps.py`, `.github/workflows/refresh.yml`.

---

## Known Issues / Bugs

_(No open bugs.)_
