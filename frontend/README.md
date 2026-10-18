# Tennis SSM Frontend

Vite + React frontend for the WTA match prediction state-space model.

## Recent changes

- Removed the "Recent match results" reference table from the Completed results section; completed forecast rows already display the same data.
- Removed the match count label from the filter bar.
- Removed the "Only incorrect" filter toggle.
- Fixed overflow in the model results summary card by allowing long values to wrap within stat cells.

## Available scripts

```bash
npm install
npm test
npm run build
```

The app is deployed to GitHub Pages at <https://ryantjx.github.io/tennis_ssm/>.

## Data

The production bundle fetches `predictions.json` and `results.json` at runtime
from `VITE_DATA_BASE_URL`, which defaults to this repository's `outputs/latest`
directory on `main`. Set the variable at build time to point elsewhere:

```bash
VITE_DATA_BASE_URL=https://example.com/outputs/latest npm run build
```

In development the app reads `public/data/*.json` instead, because those files
are not committed. Copy the current artifacts before starting the dev server:

```bash
npm run sync-data   # copies ../outputs/latest/*.json into public/data
npm run dev         # open http://localhost:5173/tennis_ssm/
```

The dev URL includes the `/tennis_ssm/` base path configured in
`vite.config.ts`.

`results.json` is optional at runtime: it supplies only the completed-results
subtitle and footer attribution, so the dashboard still renders if it is
unavailable. A staleness banner appears when the published data is older than
the threshold recorded in its `data_freshness` block.
