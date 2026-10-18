"""Optional future-fixture ingestion from the Live Tennis API.

This is an **additional** fixture source, not a replacement. ``fixtures_womens``
(WTA site API) and ``fixtures_men`` (ATP site API) remain the defaults and are
untouched. Nothing here runs unless a caller explicitly asks for it and supplies
a key, so an installation that ignores this module behaves exactly as before.

Why it exists: the repository's stated limitation is that ATP fixtures are hard
to obtain, and each tour currently needs its own bespoke loader against a tour
website's private JSON API. ``GET /fixtures`` serves ``atp``, ``wta``,
``challenger``, ``itf`` and ``juniors`` through one documented, versioned shape,
so a second tour costs a parameter rather than a new scraper.

Configuration
-------------
Set ``LIVETENNISAPI_KEY`` (or pass ``api_key=``). With no key set, this module is
inert -- it is imported but never reached by the default pipeline.

Scope
-----
Only fields published in the v1.1.0 OpenAPI description are read:
``id``, ``event_date``, ``tour``, ``tournament``, ``round``, ``surface``,
``player1_name``, ``player2_name`` and ``status``. Fields the provider does not
publish (venue, tournament id, tier, scheduled time-of-day) are recorded as
``Unknown`` rather than guessed, so no value in the output is synthesised.

Spec: https://github.com/livetennisapi/openapi -- docs: https://docs.livetennisapi.com
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import polars as pl

from src.data.fixture_common import (
    FIXTURE_COLUMNS,
    empty_fixture_frame,
    first_present,
    player_keys,
)

LIVETENNISAPI_BASE = "https://api.livetennisapi.com/api/public/v1"
LIVETENNISAPI_KEY_ENV = "LIVETENNISAPI_KEY"

#: Vocabulary accepted by the ``tour`` query parameter of ``GET /fixtures``.
#: This is the *filter* vocabulary. It is deliberately kept separate from the
#: ``tour`` field on a fixture record, which is a different, more granular
#: vocabulary -- see ``_is_doubles_tour``.
TOUR_CHOICES = ("atp", "wta", "challenger", "itf", "juniors")

#: ``limit`` is capped at 200 by the provider.
PAGE_LIMIT = 200

#: Refuse to page forever if ``has_more`` never goes false.
MAX_PAGES = 50


def load_livetennisapi_fixtures(
    tour: str = "wta",
    start_date: str | None = None,
    end_date: str | None = None,
    origin_date: str = "2022-12-31",
    api_key: str | None = None,
    max_pages: int = MAX_PAGES,
) -> pl.DataFrame:
    """Load scheduled singles fixtures for one tour from the Live Tennis API.

    Args:
        tour: One of :data:`TOUR_CHOICES`. Required rather than optional so the
            normalized ``tour`` column is exactly what was asked for; the tour
            reported *on a record* is a different vocabulary and is not parsed
            back into this one.
        start_date: Inclusive ISO date lower bound. Defaults to today.
        end_date: Inclusive ISO date upper bound. Defaults to ``start + 7d``.
        origin_date: Reference date for the integer ``timestamp`` column.
        api_key: Overrides the ``LIVETENNISAPI_KEY`` environment variable.
        max_pages: Safety stop for pagination.

    Returns:
        A frame with the normalized :data:`FIXTURE_COLUMNS` schema, matching
        what ``load_wta_fixtures`` and ``load_atp_fixtures`` return.
    """
    tour_key = str(tour or "").strip().lower()
    if tour_key not in TOUR_CHOICES:
        raise ValueError(f"tour must be one of {TOUR_CHOICES}, got {tour!r}")

    start = dt.date.today() if start_date is None else dt.date.fromisoformat(start_date)
    end = start + dt.timedelta(days=7) if end_date is None else dt.date.fromisoformat(end_date)

    key = resolve_api_key(api_key)
    records: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        payload = _livetennisapi_get_json(
            "/fixtures",
            {"tour": tour_key, "limit": PAGE_LIMIT, "offset": offset},
            api_key=key,
        )
        page = payload.get("data") or []
        records.extend(page)

        meta = payload.get("meta") or {}
        # The provider documents `has_more` as the pagination signal; comparing
        # `count` against `limit` is explicitly not the supported test.
        if not meta.get("has_more"):
            break
        if not page:
            break
        offset += len(page)

    frame = normalize_livetennisapi_fixture_rows(
        records,
        tour=tour_key,
        origin_date=origin_date,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )
    if frame.height == 0:
        return frame
    return frame.sort(["date", "source_match_id"])


def normalize_livetennisapi_fixture_rows(
    fixtures: list[dict[str, Any]],
    tour: str = "wta",
    origin_date: str = "2022-12-31",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Normalize raw ``GET /fixtures`` records into model-ready fixture rows.

    Pure function: no network access, so it is directly unit-testable against
    spec-shaped records.
    """
    origin = dt.date.fromisoformat(origin_date)
    start = dt.date.fromisoformat(start_date) if start_date else None
    end = dt.date.fromisoformat(end_date) if end_date else None
    tour_label = str(tour or "").strip().upper() or "Unknown"

    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue

        # The record's own `tour` is an opaque string in a more granular
        # vocabulary than the filter. The one thing it is documented to encode
        # is case: a doubles team reports it uppercase where an individual
        # reports lowercase. This model is singles-only, so drop doubles.
        if _is_doubles_tour(fixture.get("tour")):
            continue

        player1_full = _clean_name(fixture.get("player1_name"))
        player2_full = _clean_name(fixture.get("player2_name"))
        if not player1_full or not player2_full:
            continue
        # A "/" separated name is a doubles pairing, not a player.
        if "/" in player1_full or "/" in player2_full:
            continue

        # `event_date` is a date, not a datetime, and is nullable. A fixture we
        # cannot place in time cannot become an observation timestamp, so it is
        # dropped rather than defaulted to today.
        match_date = _parse_date(fixture.get("event_date"))
        if match_date is None:
            continue
        if start is not None and match_date < start:
            continue
        if end is not None and match_date > end:
            continue

        player1_key, player1_alt = livetennisapi_player_keys(player1_full)
        player2_key, player2_alt = livetennisapi_player_keys(player2_full)
        if not player1_key or not player2_key:
            continue

        rows.append(
            {
                "tour": tour_label,
                "date": match_date.isoformat(),
                "timestamp": (match_date - origin).days,
                "player1": player1_key,
                "player2": player2_key,
                "player1_full_name": player1_full,
                "player2_full_name": player2_full,
                "player1_alt": player1_alt,
                "player2_alt": player2_alt,
                "tournament": first_present(fixture.get("tournament"), "Unknown"),
                # The provider publishes no venue/city or tournament tier on a
                # fixture, and no tournament entity to look one up from.
                "location": "Unknown",
                "tier": "Unknown",
                "surface": first_present(fixture.get("surface"), "Unknown"),
                "round": first_present(fixture.get("round"), "Unknown"),
                "source": "livetennisapi",
                "source_match_id": first_present_id(fixture.get("id")),
                # No tournament identifier is published for a fixture.
                "source_tournament_id": "",
                # Keep the record's own opaque tour string rather than discard
                # it; it is informational only and is never parsed.
                "source_event": first_present(fixture.get("tour"), "Unknown"),
                "date_source": "event_date",
                "match_state": first_present(fixture.get("status"), ""),
            }
        )

    if not rows:
        return empty_fixture_frame()
    return pl.DataFrame(rows).select(FIXTURE_COLUMNS)


def livetennisapi_player_keys(name: str | None) -> tuple[str, str]:
    """Return tennis-data compatible keys for a single full-name string.

    The provider publishes ``player1_name`` as one string and does not document
    the token order, so both readings are produced:

    * ``player1``     -- "First ... Last" read, e.g. "Linda Noskova" -> "Noskova L"
    * ``player1_alt`` -- "Last ... First" read, e.g. "Noskova Linda" -> "Noskova L"

    Both are *exact* keys in the model's existing normalized key space. Nothing
    here does approximate or edit-distance matching: a key either is present in
    ``name_to_id`` or the fixture is dropped. Use
    :func:`resolve_livetennisapi_fixtures` when you want the stronger guarantee
    that a name resolving differently under the two readings is discarded.
    """
    cleaned = _clean_name(name)
    if not cleaned:
        return "", ""

    tokens = cleaned.split()
    if len(tokens) == 1:
        only, _ = player_keys(tokens[0], tokens[0])
        return only, only

    forward, _ = player_keys(tokens[0], tokens[-1])
    reversed_, _ = player_keys(tokens[-1], tokens[0])
    return forward, reversed_


def resolve_livetennisapi_fixtures(
    fixtures: pl.DataFrame,
    name_to_id: dict[str, int],
) -> pl.DataFrame:
    """Ambiguity-safe variant of ``filter_known_fixtures`` for this source.

    ``filter_known_fixtures`` tries each candidate key in order and takes the
    first hit. That is fine for a source whose name order is known. Here the
    order is not documented, so a fixture whose two readings resolve to two
    *different* players is genuinely ambiguous and is dropped rather than
    guessed at.
    """
    if fixtures.height == 0:
        return _empty_resolved_frame()

    rows: list[dict[str, Any]] = []
    for row in fixtures.iter_rows(named=True):
        player1_id = _resolve_unambiguous(row.get("player1"), row.get("player1_alt"), name_to_id)
        player2_id = _resolve_unambiguous(row.get("player2"), row.get("player2_alt"), name_to_id)
        if player1_id is None or player2_id is None:
            continue
        if player1_id == player2_id:
            continue
        rows.append({**row, "player1_id": player1_id, "player2_id": player2_id})

    if not rows:
        return _empty_resolved_frame()
    return pl.DataFrame(rows)


def resolve_api_key(api_key: str | None = None) -> str:
    """Return the configured API key, or raise a message naming the variable."""
    key = api_key or os.environ.get(LIVETENNISAPI_KEY_ENV, "")
    key = str(key).strip()
    if not key:
        raise RuntimeError(
            f"No Live Tennis API key. Set {LIVETENNISAPI_KEY_ENV} or pass api_key=. "
            "This source is optional; the default WTA/ATP fixture loaders need no key."
        )
    return key


def first_present_id(value: Any) -> str:
    """Stringify a fixture id, or empty string when absent."""
    if value is None or str(value).strip() == "":
        return ""
    return str(value).strip()


def _resolve_unambiguous(
    primary: str | None,
    alternate: str | None,
    name_to_id: dict[str, int],
) -> int | None:
    """Resolve a player only when the two name readings do not disagree."""
    hits = {
        name_to_id[key]
        for key in (primary, alternate)
        if key and key in name_to_id
    }
    if len(hits) != 1:
        return None
    return hits.pop()


def _empty_resolved_frame() -> pl.DataFrame:
    return (
        empty_fixture_frame()
        .with_columns(
            pl.lit(None, dtype=pl.Int64).alias("player1_id"),
            pl.lit(None, dtype=pl.Int64).alias("player2_id"),
        )
        .filter(pl.lit(False))
    )


def _is_doubles_tour(tour_value: Any) -> bool:
    """Detect a doubles record from the documented casing of its own tour string.

    The provider documents that a doubles team reports its tour uppercase
    (``ATP``) where an individual reports it lowercase (``atp``). The value is
    otherwise treated as opaque and is never mapped onto the filter vocabulary.
    """
    if tour_value is None:
        return False
    text = str(tour_value).strip()
    if not text:
        return False
    return text.isupper()


def _clean_name(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _parse_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _livetennisapi_get_json(
    path: str,
    params: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{LIVETENNISAPI_BASE}{path}?{query}" if query else f"{LIVETENNISAPI_BASE}{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 401/403 is the provider saying no; retrying cannot fix it.
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"Live Tennis API rejected the key for {path} (HTTP {exc.code}). "
                    f"Check {LIVETENNISAPI_KEY_ENV} and that your plan covers this endpoint."
                ) from exc
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError(f"Live Tennis API request failed: {url}")


def main() -> None:
    """Fetch future fixtures from the Live Tennis API and print them."""
    parser = argparse.ArgumentParser(
        description="Fetch normalized future singles fixtures from the Live Tennis API."
    )
    parser.add_argument("--tour", choices=list(TOUR_CHOICES), default="wta")
    parser.add_argument("--start-date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--end-date",
        default=(dt.date.today() + dt.timedelta(days=7)).isoformat(),
    )
    parser.add_argument("--origin-date", default="2022-12-31")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    fixtures = load_livetennisapi_fixtures(
        tour=args.tour,
        start_date=args.start_date,
        end_date=args.end_date,
        origin_date=args.origin_date,
    )
    output = args.output or f"livetennisapi_{args.tour}_fixtures.csv"
    fixtures.write_csv(output)
    print(
        f"Loaded {fixtures.height} normalized {args.tour.upper()} singles fixtures "
        f"from {args.start_date} to {args.end_date}"
    )
    print(fixtures)


if __name__ == "__main__":
    main()
