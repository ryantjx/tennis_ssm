# Data Methodology

This package keeps historical match results and future fixtures as separate data
flows because they answer different questions.

## Historical Results

Historical WTA match results are loaded by `data.py` from yearly Excel files on
`tennis-data.co.uk`, using URLs of the form:

```text
http://www.tennis-data.co.uk/{year}w/{year}.xlsx
```

These files are used for training and evaluation because they contain completed
matches with known winners and losers. The loader:

- downloads the yearly WTA files for the requested date range;
- keeps match date, winner, loser, tournament, location, tier, surface, and
  round;
- normalizes player names into the same key format used throughout the model;
- builds `name_to_id` and `id_to_name` mappings from all observed players;
- computes integer timestamps as days since `origin_date`;
- computes each player's previous-match timestamp for the Wiener-process skill
  dynamics;
- stores winners as `player1`, losers as `player2`, and sets `winner = 1.0`.

The resulting `TennisData` object contains both a Polars dataframe for
inspection and a `WTATennisResults` JAX tuple for model filtering and
evaluation.

## Future Fixtures

Future fixtures are loaded by `fixtures_womens.py` and `fixtures_men.py` from
public website APIs used by the tour sites. They are intentionally separate
from historical tennis-data rows because future fixtures do not have winners.

### WTA Fixtures

WTA fixtures come from the public WTA website API used by `wtatennis.com`. The
current implementation queries:

```text
https://api.wtatennis.com/tennis/tournaments/
https://api.wtatennis.com/tennis/tournaments/{tournament_id}/{year}/matches
```

with the same `account: wta` header used by the WTA site. This source is used
only for scheduled matchups where both players are already known. It does not
provide or require future winners.

The fixture loader:

- finds WTA tournaments active in the requested date range;
- fetches tournament match rows for each tournament;
- keeps only singles matches by default;
- keeps only upcoming matches by default (`MatchState == "U"`);
- converts WTA first/last names into tennis-data-compatible player keys such as
  `Muchova K` and `Gauff C`;
- preserves full display names such as `Karolina Muchova` and `Coco Gauff`;
- computes the same integer timestamp convention used by historical results;
- emits source metadata including `source`, `source_match_id`, `source_event`,
  `date_source`, and `match_state`.

### ATP Fixtures

ATP fixtures come from public Tennis TV / ATP Media endpoints used by
`tennistv.com`. Tennis TV is used instead of direct ATP official-site scraping
for v1 because direct non-browser requests to `atptour.com` can hit Cloudflare
challenge pages. The implementation queries:

```text
https://api.tennistv.com/tennis/v1/tournaments?from={date}&to={date}
https://api.tennistv.com/tennis/v1/matches?tournamentId={id}&year={year}
https://api.tennistv.com/tennis/v1/tournaments/{id}/{year}/draws
```

The Tennis TV match endpoint is preferred because rows can include exact
`MatchDate` values. The draw endpoint is a fallback for known future ATP
matchups when both opponents are already known but the exact order-of-play
timestamp is not yet exposed.

The ATP loader:

- finds ATP or joint tournaments active in the requested date range;
- fetches tournament match rows and keeps only unplayed singles rows with a
  reliable `MatchDate`;
- optionally fetches tournament draw rows and keeps only unplayed men's singles
  matchups where both opponents are known;
- excludes doubles, completed results, and draw placeholders;
- converts ATP first/last names into tennis-data-compatible player keys such as
  `Sinner J` and `Djokovic N`;
- marks exact dated match rows with `date_source = "match_date"`;
- marks opt-in draw-only future matchups with `date_source = "draw_unknown"`
  and null `date` / `timestamp`.

Draw-only ATP rows are useful for research and manual inspection, but they are
not prediction-ready in v1 because the model pipeline needs a timestamp and the
implementation does not fabricate dates. The default ATP command excludes these
rows.

#### ATP date resolution issue

Tennis TV draws can expose a future matchup before Tennis TV or ATP exposes a
dated match row. This caused confusion for the 2026 Wimbledon men's semifinals:
Tennis TV draws showed `Jannik Sinner vs Novak Djokovic` and `Arthur Fery vs
Alexander Zverev`, but those rows had no `MatchDate`. The date was available on
Sky Sports' tennis schedule instead, where both matches appeared under Friday,
2026-07-10, with time still listed as `TBD`.

The general ATP fix is to keep Tennis TV as the draw/matchup source, then add a
date resolver from a date-bearing schedule source. The current no-key candidate
is Sky Sports:

```text
https://www.skysports.com/tennis/scores-schedule/{DD-MM-YYYY}
```

For ATP rows, parse sections whose heading contains `ATP World Tour / Men's
Singles`, then extract tournament, round, players, status, and time if present.
Use the URL date as `date`, compute `timestamp` from `origin_date`, and set
`date_source = "sky_sports_schedule"`. If a requested date window is
2026-07-08 to 2026-07-09, the Wimbledon men's semifinals should not appear; if
the window includes 2026-07-10, they should appear with date `2026-07-10`.

API-Tennis is a more structured fallback if an API key is acceptable. Its
`get_fixtures` endpoint supports `date_start`, `date_stop`, ATP event-type
filters, and returns `event_date`, `event_time`, players, tournament, and
round.

### Normalized Schema

The normalized fixture schema is:

```text
tour, date, timestamp,
player1, player2,
player1_full_name, player2_full_name,
player1_alt, player2_alt,
tournament, location, tier, surface, round,
source, source_match_id, source_tournament_id,
source_event, date_source, match_state
```

Before prediction, fixtures are filtered with the trained model's `name_to_id`
mapping. For v1, a fixture is skipped if either player cannot be resolved to a
known model player. This avoids silently assigning a new prior skill to an
unseen player in a future prediction.

## Prediction Usage

`main.py` now prefers real WTA scheduled fixtures for future predictions. If
fixture loading fails or no fixture rows map to known model players, it falls
back to the older synthetic top-player matchups.

Historical completed matches remain the only rows with:

```text
actual_winner, correct, log_score
```

Future fixture predictions intentionally export:

```text
actual_winner = null
correct = null
log_score = null
```

## Running The Fixture Pull

Print the next seven days of scheduled WTA singles fixtures:

```bash
.venv/bin/python -m src.data.fixtures_womens
```

Print a specific date window:

```bash
.venv/bin/python -m src.data.fixtures_womens --start-date 2026-07-08 --end-date 2026-07-09
```

Print ATP future matchups from Tennis TV:

```bash
.venv/bin/python -m src.data.fixtures_men --start-date 2026-07-08 --end-date 2026-07-09
```

Include ATP draw-only research rows that lack exact scheduled dates:

```bash
.venv/bin/python -m src.data.fixtures_men --include-draw-unknown-dates
```

Each command prints the normalized Polars table and writes a tour-specific CSV
for inspection. The legacy `src.data.fixtures` module remains as a compatibility
wrapper with a combined `--tour` option.

## Known Limitations

- `tennis-data.co.uk` is excellent for historical results but does not expose
  future scheduled matches.
- The WTA API is a public endpoint used by the website, not a formal contracted
  data feed; endpoint or schema changes may require maintenance.
- WTA fixture names and tennis-data historical names are not identical, so the
  loader creates both primary and alternate player keys for matching.
- ATP fixture research support is included, but the current model and
  historical loader remain WTA-specific. ATP rows should not be merged into the
  WTA prediction output until a separate ATP historical training path exists.
- Tennis TV draw rows can identify known future ATP matchups before an exact
  scheduled date/time is exposed. Those rows are opt-in and intentionally have
  null `date` / `timestamp`.
- ATP official schedule pages can lag the draw state. For the 2026 Wimbledon
  men's semifinals, ATP showed no daily schedule while Sky Sports carried the
  date-bearing schedule.
- The fixture feed is for known scheduled matches only, not full bracket
  simulation with unknown future opponents.
