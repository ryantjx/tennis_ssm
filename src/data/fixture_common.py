"""Shared future-fixture normalization helpers."""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl

from src.data.data import _consolidate_name_string

FIXTURE_COLUMNS = [
    "tour",
    "date",
    "timestamp",
    "player1",
    "player2",
    "player1_full_name",
    "player2_full_name",
    "player1_alt",
    "player2_alt",
    "tournament",
    "location",
    "tier",
    "surface",
    "round",
    "source",
    "source_match_id",
    "source_tournament_id",
    "source_event",
    "date_source",
    "match_state",
]


def empty_fixture_frame() -> pl.DataFrame:
    """Return an empty fixture frame with the normalized schema."""
    return pl.DataFrame(
        schema={
            "tour": pl.String,
            "date": pl.String,
            "timestamp": pl.Int64,
            "player1": pl.String,
            "player2": pl.String,
            "player1_full_name": pl.String,
            "player2_full_name": pl.String,
            "player1_alt": pl.String,
            "player2_alt": pl.String,
            "tournament": pl.String,
            "location": pl.String,
            "tier": pl.String,
            "surface": pl.String,
            "round": pl.String,
            "source": pl.String,
            "source_match_id": pl.String,
            "source_tournament_id": pl.String,
            "source_event": pl.String,
            "date_source": pl.String,
            "match_state": pl.String,
        }
    )


def filter_known_fixtures(fixtures: pl.DataFrame, name_to_id: dict[str, int]) -> pl.DataFrame:
    """Keep fixtures where both players can be mapped to the trained model."""
    if fixtures.height == 0:
        return fixtures.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("player1_id"),
            pl.lit(None, dtype=pl.Int64).alias("player2_id"),
        ).filter(pl.lit(False))

    rows: list[dict[str, Any]] = []
    for row in fixtures.iter_rows(named=True):
        player1_id = resolve_player_id(row, "player1", name_to_id)
        player2_id = resolve_player_id(row, "player2", name_to_id)
        if player1_id is None or player2_id is None:
            continue
        rows.append({**row, "player1_id": player1_id, "player2_id": player2_id})

    if not rows:
        return empty_fixture_frame().with_columns(
            pl.lit(None, dtype=pl.Int64).alias("player1_id"),
            pl.lit(None, dtype=pl.Int64).alias("player2_id"),
        ).filter(pl.lit(False))
    return pl.DataFrame(rows)


def resolve_player_id(row: dict[str, Any], player_col: str, name_to_id: dict[str, int]) -> int | None:
    """Resolve a fixture player using model-style and source-name aliases."""
    suffix = "1" if player_col == "player1" else "2"
    candidates = [
        row.get(player_col),
        row.get(f"{player_col}_alt"),
        _consolidate_name_string(row.get(f"player{suffix}_full_name")),
    ]
    for candidate in candidates:
        if candidate in name_to_id:
            return name_to_id[candidate]
    return None


def player_keys(first_name: str | None, last_name: str | None) -> tuple[str, str]:
    """Return tennis-data compatible keys for a first/last name pair."""
    first = str(first_name or "").strip()
    last = str(last_name or "").strip()
    if not first or not last:
        full = _consolidate_name_string(full_name(first, last)) or ""
        return full, full

    initial_key = _consolidate_name_string(f"{last} {first[0]}.") or f"{last} {first[0]}"
    alt_key = _consolidate_name_string(f"{last} {first[:3]}.") or initial_key
    return initial_key, alt_key


def parse_iso_datetime(value: str | None) -> dt.datetime | None:
    """Parse common source API timestamp strings."""
    if not value or value == "Unknown":
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def full_name(first_name: str | None, last_name: str | None) -> str:
    """Join source first/last names, ignoring missing parts."""
    return " ".join(
        part for part in [str(first_name or "").strip(), str(last_name or "").strip()] if part
    )


def first_present(*values: Any) -> str:
    """Return the first non-empty value as a string, or ``Unknown``."""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return "Unknown"


def first_present_or_none(*values: Any) -> str | None:
    """Return the first non-empty value as a string, or None."""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None
