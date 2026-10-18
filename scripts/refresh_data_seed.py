"""Refresh the committed historical workbook seed.

tennis-data.co.uk returns HTTP 403 to datacenter IP ranges, so GitHub Actions
runners cannot download the spreadsheets. The committed copies under
``data/historical`` are therefore the base of the historical dataset, and CI
appends recently completed results from the WTA API on top of them.

Run this locally (where the host is reachable) to refresh the seed:

    python scripts/refresh_data_seed.py            # seasons that are missing
    python scripts/refresh_data_seed.py --force    # re-download every season

Commit the result. The daily workflow does not modify the seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/refresh_data_seed.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data import (  # noqa: E402
    DATA_CACHE_DIR,
    DATA_SEED_DIR,
    discover_wta_workbook_urls,
    fetch_url_bytes,
)

FIRST_SEASON = 2021
SEED_FILENAME_TEMPLATE = "{year}w.xlsx"


def seasons_to_refresh(season_end: int, force: bool) -> list[int]:
    seasons = []
    for year in range(FIRST_SEASON, season_end + 1):
        target = DATA_SEED_DIR / SEED_FILENAME_TEMPLATE.format(year=year)
        if force or not target.exists() or target.stat().st_size == 0:
            seasons.append(year)
    return seasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download seasons that are already present",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="last season to fetch (defaults to the current year)",
    )
    args = parser.parse_args(argv)

    import datetime as dt

    end_year = args.end_year or dt.datetime.now(dt.timezone.utc).year
    seasons = seasons_to_refresh(end_year, args.force)
    if not seasons:
        print(f"Seed is already complete through {end_year}: {DATA_SEED_DIR}")
        return 0

    print(f"Refreshing seed for {seasons} from tennis-data.co.uk")
    discovered = discover_wta_workbook_urls()
    DATA_SEED_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[int] = []
    for year in seasons:
        url = discovered.get(year)
        if url is None:
            failures.append(year)
            print(f"  {year}: no workbook link found on the data index page")
            continue
        try:
            payload = fetch_url_bytes(url)
        except Exception as exc:  # noqa: BLE001 - reported per season
            failures.append(year)
            print(f"  {year}: download failed: {exc}")
            continue
        target = DATA_SEED_DIR / SEED_FILENAME_TEMPLATE.format(year=year)
        target.write_bytes(payload)
        # Keep the local cache in step so the next run does not re-download.
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_CACHE_DIR / target.name).write_bytes(payload)
        print(f"  {year}: wrote {target} ({len(payload)} bytes)")

    if failures:
        print(
            f"\nFailed seasons: {failures}. "
            "The seed may be incomplete; check the host or your network.",
            file=sys.stderr,
        )
        return 1
    print(f"\nSeed refreshed in {DATA_SEED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
