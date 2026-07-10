# GitHub Workflows

This repository separates model data generation from GitHub Pages deployment.

## Daily prediction generation

`generate-predictions.yml` runs once per day at `08:00 UTC` and can also be
started manually with `workflow_dispatch`.

The job:

1. Checks out `main`.
2. Installs Python dependencies.
3. Runs `python main.py`.
4. Reloads the latest available completed WTA historical results.
5. Reruns the sequential filter over all completed historical observations.
6. Uses the latest filtered state to generate unplayed fixture predictions.
7. Writes:
   - `outputs/latest/predictions.json`
   - `outputs/latest/results.json`
   - `outputs/daily/YYYY-MM-DD/predictions.json`
   - `outputs/daily/YYYY-MM-DD/results.json`
8. Commits the updated output files back to `main` when they changed.

`results.json` is historical-only. It does not include current or unplayed
fixtures. Unplayed matches live in `predictions.json` as `future_matches`.

## GitHub Pages deployment

`deploy.yml` deploys the React app shell to GitHub Pages. It does not run the
model and does not regenerate prediction data.

The deploy job runs only when frontend code or the deploy workflow changes:

- `frontend/**`
- `.github/workflows/deploy.yml`

The production frontend fetches the latest data directly from:

- `outputs/latest/predictions.json`
- `outputs/latest/results.json`

on the `main` branch, so daily prediction updates do not require a Pages
redeploy.

## Saved model parameters

`outputs/tennis_factorial_state.json` is intentionally tracked. It provides the
saved `tau`, `s`, and `init_var` parameters used by daily generation before the
filter is rerun over the latest historical data.
