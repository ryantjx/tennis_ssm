"""Future fixture ingestion for ATP men's singles matchups.

ATP fixtures come from public Tennis TV / ATP Media JSON endpoints. Match-list
rows are used when they include a reliable ``MatchDate``; draw rows can be
included explicitly as research rows for known future matchups. Draw-only rows
keep ``date`` and ``timestamp`` null and set ``date_source = "draw_unknown"``,
so they are excluded from the default fixture pull.
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
    first_present_or_none,
    full_name,
    parse_iso_datetime,
    player_keys,
)

TENNIS_TV_API_BASE = "https://api.tennistv.com/tennis/v1"
TENNIS_TV_HEADERS = {"account-id": "35"}


def load_atp_fixtures(
    start_date: str | None = None,
    end_date: str | None = None,
    origin_date: str = "2022-12-31",
    include_draw_unknown_dates: bool = False,
) -> pl.DataFrame:
    """Load ATP singles fixtures from the public Tennis TV endpoints.

    By default this returns only rows with a reliable ``date`` and
    ``timestamp``. Pass ``include_draw_unknown_dates=True`` to include
    draw-known research rows where Tennis TV has not exposed an exact scheduled
    date/time.
    """
    if start_date is None:
        start = dt.date.today()
    else:
        start = dt.date.fromisoformat(start_date)
    if end_date is None:
        end = start + dt.timedelta(days=7)
    else:
        end = dt.date.fromisoformat(end_date)

    tournaments = fetch_tennis_tv_tournaments(start.isoformat(), end.isoformat())
    frames: list[pl.DataFrame] = []

    for tournament in tournaments:
        if not _is_atp_tournament(tournament):
            continue
        tournament_id = tournament.get("id")
        tournament_year = tournament.get("year")
        if tournament_id is None or tournament_year is None:
            continue

        try:
            match_response = fetch_tennis_tv_tournament_matches(
                int(tournament_id),
                int(tournament_year),
            )
            match_frame = normalize_atp_match_rows(
                match_response.get("matches") or [],
                tournament=tournament,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                origin_date=origin_date,
            )
            if match_frame.height:
                frames.append(match_frame)
        except Exception as exc:
            print(f"Warning: could not load Tennis TV matches for {tournament_id}/{tournament_year}: {exc}")

        if include_draw_unknown_dates:
            try:
                draw_response = fetch_tennis_tv_tournament_draws(
                    int(tournament_id),
                    int(tournament_year),
                )
                draw_frame = normalize_atp_draw_rows(
                    draw_response,
                    tournament=tournament,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    origin_date=origin_date,
                )
                if draw_frame.height:
                    frames.append(draw_frame)
            except Exception as exc:
                print(f"Warning: could not load Tennis TV draws for {tournament_id}/{tournament_year}: {exc}")

    if not frames:
        return empty_fixture_frame()

    data = pl.concat(frames, how="diagonal_relaxed")
    return (
        data.unique(subset=["source", "source_match_id"], keep="first")
        .sort(["date", "source_match_id"], nulls_last=True)
        .select(FIXTURE_COLUMNS)
    )


def fetch_tennis_tv_tournaments(start_date: str, end_date: str) -> list[dict[str, Any]]:
    """Fetch Tennis TV tournaments active in a date range."""
    response = _tennis_tv_get_json("/tournaments", {"from": start_date, "to": end_date})
    if isinstance(response, list):
        return response
    return response.get("tournaments") or response.get("content") or []


def fetch_tennis_tv_tournament_matches(tournament_id: int, tournament_year: int) -> dict[str, Any]:
    """Fetch Tennis TV match rows for a tournament."""
    response = _tennis_tv_get_json(
        "/matches",
        {"tournamentId": tournament_id, "year": tournament_year},
    )
    if isinstance(response, dict):
        return response
    return {"matches": []}


def fetch_tennis_tv_tournament_draws(tournament_id: int, tournament_year: int) -> dict[str, Any]:
    """Fetch Tennis TV draw rows for a tournament."""
    response = _tennis_tv_get_json(f"/tournaments/{tournament_id}/{tournament_year}/draws", {})
    if isinstance(response, dict):
        return response
    return {}


def normalize_atp_match_rows(
    matches: list[dict[str, Any]],
    tournament: dict[str, Any],
    start_date: str,
    end_date: str,
    origin_date: str = "2022-12-31",
) -> pl.DataFrame:
    """Normalize Tennis TV match rows that have reliable match dates."""
    rows: list[dict[str, Any]] = []
    origin = dt.date.fromisoformat(origin_date)
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    for match in matches:
        match_id = str(match.get("MatchId") or match.get("MatchCode") or "")
        if not _tennis_tv_is_singles_match(match, match_id):
            continue
        if _tennis_tv_match_completed(match):
            continue

        match_datetime = parse_iso_datetime(match.get("MatchDate"))
        if match_datetime is None:
            continue
        match_date = match_datetime.date()
        if match_date < start or match_date > end:
            continue

        row = _build_atp_fixture_row(
            player1=_tennis_tv_team_player(match.get("PlayerTeam1")),
            player2=_tennis_tv_team_player(match.get("PlayerTeam2")),
            tournament=tournament,
            match_date=match_date,
            timestamp=(match_date - origin).days,
            round_name=_tennis_tv_round_name(match.get("Round")),
            source_match_id=match_id,
            source="tennistv_matches",
            source_event="MS",
            date_source="match_date",
            match_state=str(match.get("Status") or match.get("PulseStatus") or "scheduled"),
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return empty_fixture_frame()
    return pl.DataFrame(rows).select(FIXTURE_COLUMNS)


def normalize_atp_draw_rows(
    draws: dict[str, Any],
    tournament: dict[str, Any],
    start_date: str,
    end_date: str,
    origin_date: str = "2022-12-31",
) -> pl.DataFrame:
    """Normalize known future ATP singles matchups from Tennis TV draw rows."""
    rows: list[dict[str, Any]] = []
    origin = dt.date.fromisoformat(origin_date)
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    singles_draw = draws.get("MS") or {}

    for round_info in singles_draw.get("Rounds") or []:
        round_name = first_present(
            round_info.get("RoundName"),
            round_info.get("ShortName"),
            "Unknown",
        )
        for fixture in round_info.get("Fixtures") or []:
            if not (fixture.get("IsTopKnown") or fixture.get("topKnown")):
                continue
            if not (fixture.get("IsBottomKnown") or fixture.get("bottomKnown")):
                continue
            if _tennis_tv_draw_completed(fixture):
                continue

            match = fixture.get("Match") or {}
            result = fixture.get("Result") or {}
            match_datetime = parse_iso_datetime(
                first_present_or_none(match.get("MatchDate"), result.get("MatchTime"))
            )
            match_date = match_datetime.date() if match_datetime else None
            if match_date is not None and (match_date < start or match_date > end):
                continue

            row = _build_atp_fixture_row(
                player1=_tennis_tv_draw_player(
                    fixture.get("DrawLineTop"),
                    match.get("PlayerTeam1"),
                    result.get("TeamTop"),
                ),
                player2=_tennis_tv_draw_player(
                    fixture.get("DrawLineBottom"),
                    match.get("PlayerTeam2"),
                    result.get("TeamBottom"),
                ),
                tournament=tournament,
                match_date=match_date,
                timestamp=(match_date - origin).days if match_date is not None else None,
                round_name=round_name,
                source_match_id=str(
                    fixture.get("MatchCode") or match.get("MatchId") or result.get("MatchCode") or ""
                ),
                source="tennistv_draws",
                source_event="MS",
                date_source="draw_match_date" if match_date is not None else "draw_unknown",
                match_state=str(fixture.get("PulseStatus") or match.get("Status") or "draw_known_unplayed"),
            )
            if row is not None:
                rows.append(row)

    if not rows:
        return empty_fixture_frame()
    return pl.DataFrame(rows).select(FIXTURE_COLUMNS)


def atp_player_keys(first_name: str | None, last_name: str | None) -> tuple[str, str]:
    """Return tennis-data compatible keys for an ATP first/last name pair."""
    return player_keys(first_name, last_name)


def _tennis_tv_get_json(path: str, params: dict[str, Any]) -> dict[str, Any] | list[Any]:
    query = urllib.parse.urlencode(params)
    url = f"{TENNIS_TV_API_BASE}{path}?{query}" if query else f"{TENNIS_TV_API_BASE}{path}"
    request = urllib.request.Request(url, headers=TENNIS_TV_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _is_atp_tournament(tournament: dict[str, Any]) -> bool:
    gender = str(tournament.get("gender") or "").upper()
    return gender in {"ATP", "JOINT"}


def _tennis_tv_is_singles_match(match: dict[str, Any], match_id: str) -> bool:
    if match.get("isSinglesMatch") is True:
        return True
    if match.get("isSinglesMatch") is False:
        return False
    if match_id.startswith("MS") or match_id.startswith("LS"):
        return True
    team1 = match.get("PlayerTeam1") or {}
    team2 = match.get("PlayerTeam2") or {}
    return not team1.get("PartnerId") and not team2.get("PartnerId")


def _tennis_tv_match_completed(match: dict[str, Any]) -> bool:
    status = str(match.get("Status") or "").lower()
    result = str(match.get("ResultString") or "").strip()
    winner = match.get("WinningPlayerId") or match.get("Winner")
    return bool(winner or result or status in {"finished", "complete", "completed"})


def _tennis_tv_draw_completed(fixture: dict[str, Any]) -> bool:
    result = fixture.get("Result") or {}
    match = fixture.get("Match") or {}
    winner = fixture.get("Winner") or result.get("Winner") or match.get("WinningPlayerId")
    result_string = first_present_or_none(
        fixture.get("ResultString"),
        result.get("ResultString"),
        match.get("ResultString"),
    )
    return bool(winner or result_string)


def _tennis_tv_team_player(team: dict[str, Any] | None) -> dict[str, Any] | None:
    if not team:
        return None
    first = first_present_or_none(team.get("PlayerFirstNameFull"), team.get("PlayerFirstName"))
    last = first_present_or_none(team.get("PlayerLastName"))
    if not first or not last:
        return None
    return {
        "id": team.get("PlayerId"),
        "first_name": first.replace(".", ""),
        "last_name": last,
    }


def _tennis_tv_draw_player(
    draw_line: dict[str, Any] | None,
    match_team: dict[str, Any] | None,
    result_team: dict[str, Any] | None,
) -> dict[str, Any] | None:
    player = None
    if draw_line:
        players = draw_line.get("Players") or []
        if len(players) == 1:
            player = players[0]
    if player is None and result_team:
        player = result_team.get("Player")
    if player is not None:
        first = first_present_or_none(player.get("FirstName"))
        last = first_present_or_none(player.get("LastName"))
        if first and last:
            return {"id": player.get("PlayerId"), "first_name": first, "last_name": last}
    return _tennis_tv_team_player(match_team)


def _build_atp_fixture_row(
    player1: dict[str, Any] | None,
    player2: dict[str, Any] | None,
    tournament: dict[str, Any],
    match_date: dt.date | None,
    timestamp: int | None,
    round_name: str,
    source_match_id: str,
    source: str,
    source_event: str,
    date_source: str,
    match_state: str,
) -> dict[str, Any] | None:
    if not player1 or not player2 or not source_match_id:
        return None
    player1_key, player1_alt = atp_player_keys(player1.get("first_name"), player1.get("last_name"))
    player2_key, player2_alt = atp_player_keys(player2.get("first_name"), player2.get("last_name"))
    player1_full = full_name(player1.get("first_name"), player1.get("last_name"))
    player2_full = full_name(player2.get("first_name"), player2.get("last_name"))
    if not player1_full or not player2_full:
        return None

    info = tournament.get("info") or {}
    tournament_id = tournament.get("id")
    tournament_year = tournament.get("year")
    return {
        "tour": "ATP",
        "date": match_date.isoformat() if match_date is not None else None,
        "timestamp": timestamp,
        "player1": player1_key,
        "player2": player2_key,
        "player1_full_name": player1_full,
        "player2_full_name": player2_full,
        "player1_alt": player1_alt,
        "player2_alt": player2_alt,
        "tournament": first_present(info.get("title"), tournament.get("name"), "Unknown"),
        "location": first_present(info.get("city"), tournament.get("location"), "Unknown"),
        "tier": first_present(tournament.get("type"), "Unknown"),
        "surface": first_present(tournament.get("surface"), "Unknown"),
        "round": round_name,
        "source": source,
        "source_match_id": source_match_id,
        "source_tournament_id": (
            f"{tournament_id}/{tournament_year}" if tournament_id is not None and tournament_year is not None else ""
        ),
        "source_event": source_event,
        "date_source": date_source,
        "match_state": match_state,
    }


def _tennis_tv_round_name(round_value: Any) -> str:
    if isinstance(round_value, dict):
        return first_present(round_value.get("LongName"), round_value.get("ShortName"), "Unknown")
    return first_present(round_value, "Unknown")


def main() -> None:
    """Fetch ATP future matchups and print the normalized Polars table."""
    parser = argparse.ArgumentParser(description="Fetch ATP men's singles future matchups.")
    parser.add_argument("--start-date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--end-date",
        default=(dt.date.today() + dt.timedelta(days=7)).isoformat(),
    )
    parser.add_argument("--origin-date", default="2022-12-31")
    parser.add_argument("--output", default="atp_fixtures.csv")
    parser.add_argument(
        "--include-draw-unknown-dates",
        action="store_true",
        help="Include draw-known ATP matchups that lack exact scheduled dates.",
    )
    args = parser.parse_args()

    fixtures = load_atp_fixtures(
        start_date=args.start_date,
        end_date=args.end_date,
        origin_date=args.origin_date,
        include_draw_unknown_dates=args.include_draw_unknown_dates,
    )
    fixtures.write_csv(args.output)
    print(
        f"Loaded {fixtures.height} normalized ATP singles fixtures "
        f"from {args.start_date} to {args.end_date}"
    )
    print(fixtures)


if __name__ == "__main__":
    main()
