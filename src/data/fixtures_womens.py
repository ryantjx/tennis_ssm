"""Future fixture ingestion for scheduled WTA matches.

WTA fixtures come from the public JSON API used by wtatennis.com. These rows
usually include scheduled timestamps, so they are prediction-ready once both
players resolve to the trained model's player mapping.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from typing import Any

import polars as pl

from src.data.fixture_common import (
    FIXTURE_COLUMNS,
    empty_fixture_frame,
    first_present,
    full_name,
    parse_iso_datetime,
    player_keys,
)

WTA_API_BASE = "https://api.wtatennis.com/tennis"
WTA_ACCOUNT_HEADER = {"account": "wta"}

ROUND_LABELS = {
    "F": "Final",
    "S": "Semifinals",
    "Q": "Quarterfinals",
    "R16": "Round of 16",
    "R32": "Round of 32",
    "R64": "Round of 64",
    "R128": "Round of 128",
}


def load_wta_fixtures(
    start_date: str | None = None,
    end_date: str | None = None,
    origin_date: str = "2022-12-31",
    match_states: tuple[str, ...] = ("U",),
) -> pl.DataFrame:
    """Load scheduled WTA singles fixtures from the public WTA site API."""
    if start_date is None:
        start = dt.date.today()
    else:
        start = dt.date.fromisoformat(start_date)
    if end_date is None:
        end = start + dt.timedelta(days=7)
    else:
        end = dt.date.fromisoformat(end_date)

    tournaments = fetch_wta_tournaments(start.isoformat(), end.isoformat())
    frames: list[pl.DataFrame] = []
    for tournament in tournaments:
        group = tournament.get("tournamentGroup") or {}
        tournament_group_id = group.get("id")
        tournament_year = tournament.get("year")
        if tournament_group_id is None or tournament_year is None:
            continue

        matches_response = fetch_wta_tournament_matches(
            int(tournament_group_id),
            int(tournament_year),
            start.isoformat(),
            end.isoformat(),
        )
        frame = normalize_wta_match_rows(
            matches_response.get("matches") or [],
            tournament=tournament,
            origin_date=origin_date,
            match_states=match_states,
        )
        if frame.height:
            frames.append(frame)

    if not frames:
        return empty_fixture_frame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["date", "source_match_id"])


def fetch_wta_tournaments(start_date: str, end_date: str) -> list[dict[str, Any]]:
    """Fetch WTA tournaments active in a date range."""
    response = _wta_get_json(
        "/tournaments/",
        {
            "page": 0,
            "pageSize": 50,
            "excludeLevels": "ITF",
            "from": start_date,
            "to": end_date,
        },
    )
    return response.get("content") or []


def fetch_wta_tournament_matches(
    tournament_group_id: int,
    tournament_year: int,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Fetch WTA tournament matches for a date range."""
    return _wta_get_json(
        f"/tournaments/{tournament_group_id}/{tournament_year}/matches",
        {"from": start_date, "to": end_date},
    )


def normalize_wta_match_rows(
    matches: list[dict[str, Any]],
    tournament: dict[str, Any],
    origin_date: str = "2022-12-31",
    match_states: tuple[str, ...] = ("U",),
) -> pl.DataFrame:
    """Normalize raw WTA API match rows into model-ready fixture rows."""
    rows: list[dict[str, Any]] = []
    origin = dt.date.fromisoformat(origin_date)
    tournament_group = tournament.get("tournamentGroup") or {}
    tournament_group_id = tournament_group.get("id")

    for match in matches:
        if match.get("DrawMatchType") != "S":
            continue
        match_state = match.get("MatchState")
        if match_state not in match_states:
            continue

        player1_full = full_name(match.get("PlayerNameFirstA"), match.get("PlayerNameLastA"))
        player2_full = full_name(match.get("PlayerNameFirstB"), match.get("PlayerNameLastB"))
        if not player1_full or not player2_full:
            continue

        match_datetime = parse_iso_datetime(match.get("MatchTimeStamp"))
        if match_datetime is None:
            continue
        match_date = match_datetime.date()

        player1_key, player1_alt = wta_player_keys(
            match.get("PlayerNameFirstA"),
            match.get("PlayerNameLastA"),
        )
        player2_key, player2_alt = wta_player_keys(
            match.get("PlayerNameFirstB"),
            match.get("PlayerNameLastB"),
        )

        rows.append(
            {
                "tour": "WTA",
                "date": match_date.isoformat(),
                "timestamp": (match_date - origin).days,
                "player1": player1_key,
                "player2": player2_key,
                "player1_full_name": player1_full,
                "player2_full_name": player2_full,
                "player1_alt": player1_alt,
                "player2_alt": player2_alt,
                "tournament": first_present(
                    tournament.get("name"),
                    tournament_group.get("metadata", {}).get("tournament_summary_heading"),
                    tournament_group.get("name"),
                    "Unknown",
                ),
                "location": first_present(
                    tournament.get("city"),
                    tournament.get("location"),
                    tournament_group.get("location"),
                    "Unknown",
                ),
                "tier": first_present(tournament_group.get("level"), tournament.get("level"), "Unknown"),
                "surface": first_present(tournament.get("surface"), match.get("Surface"), "Unknown"),
                "round": round_label(match.get("RoundID")),
                "source": "wta_api",
                "source_match_id": match.get("MatchID") or "",
                "source_tournament_id": str(tournament_group_id or ""),
                "source_event": "S",
                "date_source": "match_timestamp",
                "match_state": match_state or "",
            }
        )

    if not rows:
        return empty_fixture_frame()
    return pl.DataFrame(rows).select(FIXTURE_COLUMNS)


def wta_player_keys(first_name: str | None, last_name: str | None) -> tuple[str, str]:
    """Return tennis-data compatible keys for a WTA first/last name pair."""
    return player_keys(first_name, last_name)


def round_label(round_id: str | None) -> str:
    """Convert WTA round IDs into readable round labels."""
    if not round_id:
        return "Unknown"
    round_id_str = str(round_id)
    if round_id_str in ROUND_LABELS:
        return ROUND_LABELS[round_id_str]
    if round_id_str.isdigit():
        return f"Round {round_id_str}"
    return round_id_str


def _wta_get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{WTA_API_BASE}{path}?{query}" if query else f"{WTA_API_BASE}{path}"
    request = urllib.request.Request(url, headers=WTA_ACCOUNT_HEADER)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    """Fetch future WTA fixtures and print the normalized Polars table."""
    parser = argparse.ArgumentParser(description="Fetch scheduled WTA singles fixtures.")
    parser.add_argument("--start-date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--end-date",
        default=(dt.date.today() + dt.timedelta(days=7)).isoformat(),
    )
    parser.add_argument("--origin-date", default="2022-12-31")
    parser.add_argument("--output", default="wta_fixtures.csv")
    args = parser.parse_args()

    fixtures = load_wta_fixtures(
        start_date=args.start_date,
        end_date=args.end_date,
        origin_date=args.origin_date,
    )
    fixtures.write_csv(args.output)
    print(
        f"Loaded {fixtures.height} normalized WTA singles fixtures "
        f"from {args.start_date} to {args.end_date}"
    )
    print(fixtures)


if __name__ == "__main__":
    main()
