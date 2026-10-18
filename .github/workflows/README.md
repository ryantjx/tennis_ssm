# GitHub Workflows

This repository separates model data generation from GitHub Pages deployment.

## Daily prediction generation

`generate-predictions.yml` runs once per day at `06:17 UTC` and can also be
started manually with `workflow_dispatch`.

The job:

1. Checks out `main`.
2. Installs Python dependencies.
3. Runs the Python test suite (`python -m unittest discover`). Set the
   `skip_tests` dispatch input to bypass it during incident recovery only.
4. Runs `python main.py`.
5. Reloads the latest available completed WTA historical results.
6. Appends final singles results from the WTA API that are not yet present in
   the historical spreadsheet.
7. Reruns the sequential filter over all completed observations.
8. Uses the latest filtered state to generate unplayed fixture predictions.
9. Writes:
   - `outputs/latest/predictions.json`
   - `outputs/latest/results.json`
   - `outputs/daily/YYYY-MM-DD/predictions.json`
   - `outputs/daily/YYYY-MM-DD/results.json`
10. Prunes dated snapshots older than 30 days.
11. Pushes the generated artifacts to the `data` branch.

Generation retries up to three times so a transient upstream data error does
not leave the daily snapshot stale. Retries only help against transient faults:
a deterministic upstream change fails all three attempts, and the run then fails
loudly so it can be investigated.

### Why generated data goes to a `data` branch

`main` is protected by a ruleset that only accepts merged pull requests. A
write-level Actions token cannot bypass that rule on a user-owned repository:
GitHub rejects the Actions integration as a bypass actor, and granting the
write role a bypass would also open direct pushes to everyone with write
access. Weakening the rule to let a bot through would remove the review gate
for humans too.

The job therefore publishes the generated artifacts to a dedicated, unprotected
`data` branch, and the frontend reads `outputs/latest` from there. Source code
stays PR-gated on `main`; only derived prediction data lives on `data`. The push
is forced because the branch holds nothing but generated snapshots and each run
writes a complete copy, which keeps its history from growing by one
multi-megabyte payload per day.

### Freshness reporting

`scripts/report_freshness.py` prints the date of the newest completed
observation, its age, and the number of future fixtures. Recency problems are
emitted as workflow warnings, not errors, because blocking publication would
hide the last known-good forecast from the site.

### Failure alerting

A failed run opens or updates an issue labelled `pipeline-failure`; a subsequent
successful run comments and closes it. This exists because the workflow failed
on all 52 daily runs between 25 August and 17 October 2026 with no notification.

Concurrency queues rather than cancels: a cancelled run reports `cancelled`
rather than `failed`, which hides problems and can discard a completed but
unpushed run.

`results.json` is historical-only. It does not include current or unplayed
fixtures. Unplayed matches live in `predictions.json` as `future_matches`.

## GitHub Pages deployment

`deploy.yml` deploys the React app shell to GitHub Pages. It does not run the
model and does not regenerate prediction data.

The deploy job runs only when frontend code or the deploy workflow changes:

- `frontend/**`
- `.github/workflows/deploy.yml`

It runs `npm ci`, the frontend test suite, and `npm run build` before uploading
the artifact, and declares the `github-pages` environment so the deployment URL
is recorded on the run.

The production frontend fetches the latest data directly from:

- `outputs/latest/predictions.json`
- `outputs/latest/results.json`

on the `data` branch, so daily prediction updates do not require a Pages
redeploy. The base URL comes from the `VITE_DATA_BASE_URL` repository variable
and falls back to this repository's `data` branch. Set the variable if the
owner, repository, or branch is renamed.

## Site audit

`audit.yml` runs daily at 08:37 UTC and on dispatch. It asserts that the
deployed site returns HTTP 200, that its bundled assets resolve, that the
published JSON parses, and that the data is at most three days old. A failure
opens an issue labelled `site-audit`, which is closed by the next passing run.

## Saved model parameters

`outputs/tennis_factorial_state.json` is intentionally tracked. It provides the
saved `tau`, `s`, and `init_var` parameters used by daily generation before the
filter is rerun over the latest historical data.

## Historical workbook cache

`src/data/data.py` caches downloaded workbooks under `local_data/raw`. Past
seasons are immutable and reused indefinitely; the current season is refreshed
after `TENNIS_DATA_CACHE_TTL_HOURS` (default 6).

Caching alone is not sufficient in CI. `tennis-data.co.uk` returns HTTP 403 to
datacenter IP ranges, so a runner can never download a workbook. The committed
seed under `data/historical` is therefore the base of the historical dataset in
CI, with recently completed results appended from the WTA API on top. Refresh
the seed locally with `python scripts/refresh_data_seed.py` when the
spreadsheets are updated; the workflow itself never modifies it.

The WTA API backfill covers the last 30 days, so refresh the seed if the current
season spreadsheet falls further behind than that.
