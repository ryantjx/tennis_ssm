"""Compatibility wrapper for future fixture ingestion.

Tour-specific implementations live in:

* ``src.data.fixtures_womens`` for WTA fixtures
* ``src.data.fixtures_men`` for ATP fixtures

``src.data.fixtures_livetennisapi`` is an optional additional source covering
either tour through one endpoint. It is off unless ``--source livetennisapi`` is
requested and a key is configured; the defaults above are unchanged.

This wrapper preserves older imports such as ``from src.data.fixtures import
load_wta_fixtures`` while the project transitions to the split modules.
"""

from __future__ import annotations

import argparse
import datetime as dt

import polars as pl

from src.data.fixture_common import empty_fixture_frame, filter_known_fixtures, resolve_player_id
from src.data.fixtures_men import (
    atp_player_keys,
    load_atp_fixtures,
    normalize_atp_draw_rows,
    normalize_atp_match_rows,
)
from src.data.fixtures_livetennisapi import (
    TOUR_CHOICES as LIVETENNISAPI_TOUR_CHOICES,
    livetennisapi_player_keys,
    load_livetennisapi_fixtures,
    normalize_livetennisapi_fixture_rows,
    resolve_livetennisapi_fixtures,
)
from src.data.fixtures_womens import (
    load_wta_fixtures,
    normalize_wta_match_rows,
    round_label,
    wta_player_keys,
)

__all__ = [
    "LIVETENNISAPI_TOUR_CHOICES",
    "atp_player_keys",
    "empty_fixture_frame",
    "filter_known_fixtures",
    "livetennisapi_player_keys",
    "load_atp_fixtures",
    "load_livetennisapi_fixtures",
    "load_wta_fixtures",
    "normalize_atp_draw_rows",
    "normalize_atp_match_rows",
    "normalize_livetennisapi_fixture_rows",
    "normalize_wta_match_rows",
    "resolve_livetennisapi_fixtures",
    "resolve_player_id",
    "round_label",
    "wta_player_keys",
]


def main() -> None:
    """Fetch future fixtures and print the normalized Polars table."""
    parser = argparse.ArgumentParser(description="Fetch normalized future singles fixtures.")
    parser.add_argument("--tour", choices=["wta", "atp", "all"], default="wta")
    parser.add_argument(
        "--source",
        choices=["native", "livetennisapi"],
        default="native",
        help=(
            "Fixture provider. 'native' (default) uses the WTA and ATP site APIs. "
            "'livetennisapi' uses the Live Tennis API and needs LIVETENNISAPI_KEY."
        ),
    )
    parser.add_argument("--start-date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--end-date",
        default=(dt.date.today() + dt.timedelta(days=7)).isoformat(),
    )
    parser.add_argument("--origin-date", default="2022-12-31")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--include-atp-draw-unknown-dates",
        action="store_true",
        help="Include ATP draw-known future matchups that lack exact scheduled dates.",
    )
    args = parser.parse_args()

    frames: list[pl.DataFrame] = []
    if args.source == "livetennisapi":
        for tour in ["wta", "atp"] if args.tour == "all" else [args.tour]:
            frames.append(
                load_livetennisapi_fixtures(
                    tour=tour,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    origin_date=args.origin_date,
                )
            )
    else:
        if args.tour in {"wta", "all"}:
            frames.append(
                load_wta_fixtures(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    origin_date=args.origin_date,
                )
            )
        if args.tour in {"atp", "all"}:
            frames.append(
                load_atp_fixtures(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    origin_date=args.origin_date,
                    include_draw_unknown_dates=args.include_atp_draw_unknown_dates,
                )
            )

    fixtures = pl.concat(frames, how="diagonal_relaxed") if frames else empty_fixture_frame()
    output = args.output or f"{args.tour}_fixtures.csv"
    fixtures.write_csv(output)
    print(
        f"Loaded {fixtures.height} normalized {args.tour.upper()} singles fixtures "
        f"from {args.start_date} to {args.end_date}"
    )
    print(fixtures)


if __name__ == "__main__":
    main()
