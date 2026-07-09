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

The app is deployed to GitHub Pages at https://ryantjx.github.io/tennis_ssm/.
