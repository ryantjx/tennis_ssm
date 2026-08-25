# Tennis Data Exploration Prompt

Use this prompt to explore WTA tennis match data from the tennis-match-data repository.

## Prompt

```
I want to perform exploratory data analysis on WTA tennis match data from https://github.com/ryantjx/tennis-match-data

Please:
1. Load the WTA data from: https://github.com/ryantjx/tennis-match-data/releases/download/data-latest/wta.parquet using Polars
2. Process dates correctly by using `event_start_date` to extract year (the `year` column has null values)
3. Generate the following analyses with Altair visualizations:
   - Match distribution by year (with rolling average)
   - Surface type distribution (pie chart)
   - Tournament level distribution
   - Top 15 players by match count
   - Top 15 tournaments by match count (use match_date for year calculations)
   - Surface evolution by decade (stacked area chart)
   - Player win rate distribution
   - Surface vs tournament level heatmap
   - Tournament level vs surface bubble chart
   - Future fixtures summary (if available)

4. Create a comprehensive summary dashboard showing:
   - Total records, unique players, unique tournaments
   - Year range coverage
   - Surface distribution percentages
   - Tournament level breakdown
   - Top 5 players by match count

Important notes:
- Use `.show()` for all Altair charts to ensure they display
- Filter out null dates before calculating years_held, first_year, last_year
- The dataset has 776,976+ records spanning 1967-2026
- Completed matches have record_type='completed', fixtures have record_type='fixture'
```

## Quick Start

Copy the prompt above and paste it into your AI assistant to generate a complete data exploration notebook.
