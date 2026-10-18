"""
Data collection module for WTA tennis match data.

Downloads historical match results from tennis-data.co.uk, cleans player names,
builds player ID mappings, computes timestamps, and computes the previous match
timestamp for each player (needed for the Wiener-process dynamics).

This module remains the historical WTA training/backfill path. Future fixture
ingestion lives in ``src.data.fixtures_womens`` and
``src.data.fixtures_men`` because future rows have no winners. ATP fixture
research should not be merged into this WTA model pipeline until a separate
men's historical loader and model state are added.

The structure follows abile's ``datasets/tennis.py`` (name consolidation, ID
mapping, timestamp computation) but uses polars for data processing, matching
the pattern in ``references/process_data.py`` (previous-match-timestamp via
self-join).
"""

import io
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import polars as pl
from jax import numpy as jnp

from src.data.data_types import WTATennisResults, TennisMatchMetadata

# tennis-data.co.uk now serves workbook links from a hashed directory that is
# only advertised on its data index page, and it rejects the default Python
# user agent with HTTP 403. Links are therefore discovered from the index page
# and fetched with browser-like headers. The legacy yearly path is retained as
# a fallback in case the index page changes shape again.
DATA_INDEX_URL = "https://tennis-data.co.uk/data.php"
DATA_URL_TEMPLATE = "https://www.tennis-data.co.uk/{year}w/{year}.xlsx"
DATA_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}
# Downloaded workbooks are cached locally so a transient upstream outage does not
# leave the daily snapshot stale. Past seasons are immutable and cached
# indefinitely; the current season is refreshed according to a TTL because
# results are appended throughout the season.
DATA_CACHE_DIR = Path(os.environ.get("TENNIS_DATA_CACHE_DIR", "local_data/raw"))
CURRENT_SEASON_CACHE_TTL_HOURS = float(
    os.environ.get("TENNIS_DATA_CACHE_TTL_HOURS", "6")
)
WORKBOOK_LINK_PATTERN = re.compile(
    r"[\"']([^\"']*?/(\d{4})w/\2\.xlsx)[\"']", re.IGNORECASE
)


class TennisData(NamedTuple):
    """Container for loaded tennis data.

    Attributes:
        jax_data: ``WTATennisResults`` NamedTuple with JAX arrays, ready for
            filtering.
        polars_data: The cleaned polars DataFrame (for inspection / plotting).
        id_to_name: Dict mapping integer player IDs to player name strings.
        name_to_id: Dict mapping player name strings to integer player IDs.
        num_players: Total number of unique players.
        num_matches: Total number of matches.
        match_metadata: ``TennisMatchMetadata`` with string data for each match.
    """

    jax_data: WTATennisResults
    polars_data: pl.DataFrame
    id_to_name: dict
    name_to_id: dict
    num_players: int
    num_matches: int
    match_metadata: TennisMatchMetadata


def _consolidate_name_string(s: str | None) -> str | None:
    """Normalise a single player name string: strip accents, remove initials.

    Mirrors abile's ``consolidate_name_strings`` but for a single string.
    """
    import unicodedata

    if s is None:
        return s
    # NFKD normalise, encode to ASCII dropping accents, decode back
    nfkd = unicodedata.normalize("NFKD", str(s))
    ascii_str = nfkd.encode("ascii", errors="ignore").decode("utf-8")
    # Strip leading initials like "A." in "A. Smith"
    if "." in ascii_str:
        ascii_str = ascii_str.split(".")[0].strip()
    return ascii_str


def _consolidate_name_expr(col_name: str) -> pl.Expr:
    """Polars expression to consolidate player name strings in a column."""
    return pl.col(col_name).map_elements(_consolidate_name_string, return_dtype=pl.String)


def load_wta(
    start_date: str = "2018-12-31",
    end_date: str = "2024-01-01",
    origin_date: str = "2018-12-31",
    years: list[int] | None = None,
) -> TennisData:
    """Download and process WTA tennis match data.

    Args:
        start_date: Start date for filtering matches (inclusive).
        end_date: End date for filtering matches (exclusive).
        origin_date: Reference date for computing integer timestamps
            (days since origin).
        years: List of years to download. If None, inferred from
            ``start_date`` and ``end_date``.

    Returns:
        ``TennisData`` containing JAX arrays, polars DataFrame, and player
        ID mappings.
    """
    if years is None:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        # tennis-data.co.uk files start from 2019 onwards for WTA
        years = list(range(max(start_year, 2019), end_year + 1))

    discovered_urls = discover_wta_workbook_urls()

    # Download and concatenate yearly Excel files (with retry for flaky server)
    frames = []
    for year in years:
        df = None
        try:
            df = load_wta_workbook(year, discovered_urls.get(year))
        except Exception as e:
            print(f"Warning: could not load WTA {year}: {e}")
        if df is not None:
            frames.append(df)
            print(f"  Loaded {year}: {df.height} matches")

    if not frames:
        raise RuntimeError(
            "No data could be downloaded. Check network / URLs "
            f"(index page: {DATA_INDEX_URL})."
        )

    data = pl.concat(frames, how="diagonal_relaxed")

    # --- Clean and prepare ---
    # tennis-data.co.uk columns: Date, Winner, Loser, Tournament, Location, Tier, Surface, Round
    # Select tournament info along with match data
    data = data.select(["Date", "Winner", "Loser", "Tournament", "Location", "Tier", "Surface", "Round"]).drop_nulls(subset=["Date", "Winner", "Loser"])

    # Parse dates — the Excel reader may return Date as string or date
    if data.schema["Date"] == pl.String:
        data = data.with_columns(
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        ).drop_nulls(subset=["Date"])
    else:
        data = data.drop_nulls(subset=["Date"])

    # Consolidate player names
    data = data.with_columns(
        _consolidate_name_expr("Winner").alias("Winner"),
        _consolidate_name_expr("Loser").alias("Loser"),
    ).drop_nulls(subset=["Winner", "Loser"])
    
    # Fill null tournament info
    data = data.with_columns(
        pl.col("Tournament").fill_null("Unknown"),
        pl.col("Location").fill_null("Unknown"),
        pl.col("Tier").fill_null("Unknown"),
        pl.col("Surface").fill_null("Unknown"),
        pl.col("Round").fill_null("Unknown"),
    )

    # Filter by date range
    start_dt = pl.lit(start_date).str.strptime(pl.Date, format="%Y-%m-%d")
    end_dt = pl.lit(end_date).str.strptime(pl.Date, format="%Y-%m-%d")
    data = data.filter(
        (pl.col("Date") > start_dt) & (pl.col("Date") <= end_dt)
    )

    if data.height == 0:
        raise RuntimeError("No matches found in the specified date range.")

    return build_tennis_data(data, origin_date=origin_date)


def fetch_url_bytes(url: str, timeout: int = 60, attempts: int = 3) -> bytes:
    """Fetch ``url`` with browser-like headers, retrying transient failures.

    The upstream host returns HTTP 403 for the default ``Python-urllib`` user
    agent, so identification headers are always supplied.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=DATA_REQUEST_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last_error}")


def discover_wta_workbook_urls(
    index_url: str = DATA_INDEX_URL,
) -> dict[int, str]:
    """Map season year to workbook URL using the current data index page.

    The site serves workbooks from a hashed directory that changes over time,
    so the links are read from the index page rather than hard-coded. Returns an
    empty mapping when discovery fails; callers then fall back to the legacy
    yearly URL template.
    """
    try:
        html = fetch_url_bytes(index_url).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        print(f"Warning: could not read {index_url}: {exc}")
        return {}

    urls: dict[int, str] = {}
    for match in WORKBOOK_LINK_PATTERN.finditer(html):
        relative_url, year_text = match.group(1), match.group(2)
        urls[int(year_text)] = urllib.parse.urljoin(index_url, relative_url)
    return urls


def _workbook_cache_path(year: int) -> Path:
    return DATA_CACHE_DIR / f"{year}w.xlsx"


def _cache_entry_is_fresh(path: Path, year: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if year >= time.gmtime().tm_year:
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        return age_hours < CURRENT_SEASON_CACHE_TTL_HOURS
    return True


def _workbook_url_candidates(year: int, discovered_url: str | None) -> list[str]:
    """Return candidate download URLs for ``year``, most reliable first."""
    candidates: list[str] = []
    if discovered_url:
        candidates.append(discovered_url)
    candidates.append(DATA_URL_TEMPLATE.format(year=year))
    unique: list[str] = []
    for url in candidates:
        if url not in unique:
            unique.append(url)
    return unique


def load_wta_workbook(year: int, discovered_url: str | None = None) -> pl.DataFrame:
    """Load one seasonal WTA workbook, preferring a fresh local cache.

    The cache avoids re-downloading immutable past seasons and allows a stale
    but valid copy to stand in when the upstream host is unavailable, so daily
    prediction generation degrades rather than fails.
    """
    cache_path = _workbook_cache_path(year)
    if _cache_entry_is_fresh(cache_path, year):
        return pl.read_excel(cache_path)

    last_error: Exception | None = None
    for url in _workbook_url_candidates(year, discovered_url):
        try:
            payload = fetch_url_bytes(url)
            frame = pl.read_excel(io.BytesIO(payload))
        except Exception as exc:  # noqa: BLE001 - try the next candidate URL
            last_error = exc
            continue
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
        except OSError as exc:  # noqa: BLE001 - caching is optional
            print(f"Warning: could not cache WTA {year} workbook: {exc}")
        return frame

    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(
            f"Warning: using cached WTA {year} workbook after download "
            f"failure: {last_error}"
        )
        return pl.read_excel(cache_path)
    raise RuntimeError(last_error or f"no download URL available for WTA {year}")


def append_completed_results(
    historical_data: TennisData,
    completed_results: pl.DataFrame,
    origin_date: str,
) -> tuple[TennisData, int]:
    """Append completed result rows that are not already in the history.

    The historical spreadsheet can lag tournaments that are still in progress.
    WTA API results use the same eight raw columns and are deduplicated by match
    date and winner/loser before rebuilding player IDs and previous timestamps.
    """
    if completed_results.height == 0:
        return historical_data, 0

    raw_columns = [
        "Date",
        "Winner",
        "Loser",
        "Tournament",
        "Location",
        "Tier",
        "Surface",
        "Round",
    ]
    existing = historical_data.polars_data.select(raw_columns)
    updates = completed_results.select(raw_columns).with_columns(
        pl.col("Date").cast(pl.Date),
        _consolidate_name_expr("Winner").alias("Winner"),
        _consolidate_name_expr("Loser").alias("Loser"),
    ).drop_nulls(subset=["Date", "Winner", "Loser"])

    match_key = ["Date", "Winner", "Loser"]
    new_rows = updates.join(existing.select(match_key).unique(), on=match_key, how="anti")
    if new_rows.height == 0:
        return historical_data, 0

    combined = pl.concat([existing, new_rows], how="diagonal_relaxed")
    return build_tennis_data(combined, origin_date=origin_date), new_rows.height


def build_tennis_data(data: pl.DataFrame, origin_date: str) -> TennisData:
    """Build model arrays and metadata from normalized completed match rows."""
    data = data.sort(["Date", "Tournament", "Winner", "Loser"])

    # --- Build player ID mappings ---
    player_names = sorted(
        set(data.select(pl.col("Winner")).to_numpy().flatten().tolist())
        | set(data.select(pl.col("Loser")).to_numpy().flatten().tolist())
    )
    name_to_id = {name: i for i, name in enumerate(player_names)}
    id_to_name = {i: name for i, name in enumerate(player_names)}
    num_players = len(player_names)

    # --- Compute timestamps (days since origin) ---
    origin = pl.lit(origin_date).str.strptime(pl.Date, format="%Y-%m-%d")
    data = data.with_columns(
        (pl.col("Date") - origin).dt.total_days().cast(pl.Int64).alias("timestamp"),
    )

    # --- Compute previous match timestamp per player ---
    # Build a long table: (player, date, timestamp) for both winner and loser
    player_dates = pl.concat(
        [
            data.select(
                pl.col("Winner").alias("player"),
                pl.col("Date").alias("date"),
                pl.col("timestamp"),
            ),
            data.select(
                pl.col("Loser").alias("player"),
                pl.col("Date").alias("date"),
                pl.col("timestamp"),
            ),
        ]
    ).unique().sort(["player", "date"])

    # Shift within each player group to get the previous match timestamp
    player_dates = player_dates.with_columns(
        pl.col("timestamp").shift(1).over("player").alias("prev_timestamp"),
    )

    # Join back: winner's previous and loser's previous
    data = (
        data.join(
            player_dates.select(["player", "date", "prev_timestamp"]),
            left_on=["Winner", "Date"],
            right_on=["player", "date"],
            how="left",
        )
        .rename({"prev_timestamp": "player1_timestamp_previous"})
        .join(
            player_dates.select(["player", "date", "prev_timestamp"]),
            left_on=["Loser", "Date"],
            right_on=["player", "date"],
            how="left",
        )
        .rename({"prev_timestamp": "player2_timestamp_previous"})
    )

    # Fill null previous timestamps with 0 (sentinel — no previous match)
    data = data.with_columns(
        pl.col("player1_timestamp_previous").fill_null(0).cast(pl.Int64),
        pl.col("player2_timestamp_previous").fill_null(0).cast(pl.Int64),
    )

    # --- Assign player IDs ---
    data = data.with_columns(
        pl.col("Winner").replace(name_to_id).cast(pl.Int64).alias("player1_id"),
        pl.col("Loser").replace(name_to_id).cast(pl.Int64).alias("player2_id"),
    )

    # Winner is always player1 in the raw data (Winner column = match winner)
    # So winner = 1.0 always. We'll shuffle player1/player2 assignment so that
    # the model sees both perspectives. For now, keep winner=1.0 (player1=winner).
    data = data.with_columns(
        pl.lit(1.0).alias("winner"),
    )

    # Add match index
    data = data.with_row_index("match_index")

    # --- Convert to JAX NamedTuple ---
    jax_data = to_jax_data(data)
    
    # --- Extract match metadata (strings) ---
    match_metadata = TennisMatchMetadata(
        tournament=data.select(pl.col("Tournament")).to_numpy().flatten().tolist(),
        location=data.select(pl.col("Location")).to_numpy().flatten().tolist(),
        tier=data.select(pl.col("Tier")).to_numpy().flatten().tolist(),
        surface=data.select(pl.col("Surface")).to_numpy().flatten().tolist(),
        round=data.select(pl.col("Round")).to_numpy().flatten().tolist(),
    )

    return TennisData(
        jax_data=jax_data,
        polars_data=data,
        id_to_name=id_to_name,
        name_to_id=name_to_id,
        num_players=num_players,
        num_matches=data.height,
        match_metadata=match_metadata,
    )


def to_jax_data(data: pl.DataFrame) -> WTATennisResults:
    """Convert a polars DataFrame to a ``WTATennisResults`` NamedTuple.

    Args:
        data: Polars DataFrame with columns: match_index, player1_id,
            player2_id, winner, timestamp, player1_timestamp_previous,
            player2_timestamp_previous.

    Returns:
        ``WTATennisResults`` with JAX arrays (leading temporal dimension T).
    """
    return WTATennisResults(
        match_index=jnp.array(data.select(pl.col("match_index")).to_numpy().flatten()),
        player1_id=jnp.array(data.select(pl.col("player1_id")).to_numpy().flatten()),
        player2_id=jnp.array(data.select(pl.col("player2_id")).to_numpy().flatten()),
        winner=jnp.array(data.select(pl.col("winner")).to_numpy().flatten()),
        timestamp=jnp.array(data.select(pl.col("timestamp")).to_numpy().flatten()),
        player1_timestamp_previous=jnp.array(
            data.select(pl.col("player1_timestamp_previous")).to_numpy().flatten()
        ),
        player2_timestamp_previous=jnp.array(
            data.select(pl.col("player2_timestamp_previous")).to_numpy().flatten()
        ),
    )


def most_recent_timestamp_by_player(
    data: pl.DataFrame, num_players: int, default: float = 0.0
) -> jnp.ndarray:
    """Extract the most recent timestamp for each player.

    Mirrors cuthberto-carlos's ``most_recent_timestamp_by_team``.

    Args:
        data: Polars DataFrame with columns ``player1_id``, ``player2_id``,
            and ``timestamp``.
        num_players: Total number of players.
        default: Default timestamp for players without matches.

    Returns:
        JAX array of shape (num_players,) with the most recent timestamp
        for each player.
    """
    timestamps = jnp.array(data.select(pl.col("timestamp")).to_numpy().flatten())
    p1_ids = jnp.array(data.select(pl.col("player1_id")).to_numpy().flatten())
    p2_ids = jnp.array(data.select(pl.col("player2_id")).to_numpy().flatten())

    most_recent = jnp.full(num_players, default, dtype=timestamps.dtype)
    most_recent = most_recent.at[p1_ids].max(timestamps)
    most_recent = most_recent.at[p2_ids].max(timestamps)
    return most_recent
