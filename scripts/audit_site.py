"""Smoke-check the deployed site and the published prediction data.

The daily generation workflow can fail silently for weeks while the site keeps
serving an old snapshot. This script asserts the end-to-end path still works:
the Pages site responds, its assets resolve, and the JSON the frontend actually
fetches is parseable and current.

Recency failures are reported separately from availability failures so a stale
but reachable site produces a clear diagnosis rather than a generic error.

Usage:
    python scripts/audit_site.py
    python scripts/audit_site.py --site-url ... --data-base-url ... --max-age-days 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_SITE_URL = "https://ryantjx.github.io/tennis_ssm/"
DEFAULT_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/ryantjx/tennis_ssm/data/outputs/latest"
)
DEFAULT_MAX_AGE_DAYS = 3
REQUEST_TIMEOUT_SECONDS = 30
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUIRED_PAYLOAD_KEYS = (
    "generated_at",
    "model_state_as_of_date",
    "metrics",
    "matches",
    "future_matches",
)


def fetch(url: str) -> tuple[int, bytes]:
    """Return ``(status, body)`` for ``url``, treating HTTP errors as statuses."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "*/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        return 0, str(exc.reason).encode()


def check_site(site_url: str, failures: list[str]) -> list[str]:
    """Verify the Pages site responds and its referenced assets resolve."""
    status, body = fetch(site_url)
    if status != 200:
        failures.append(f"Site {site_url} returned HTTP {status}")
        return []

    html = body.decode("utf-8", errors="replace")
    if "WTA Match Prediction" not in html:
        failures.append(f"Site {site_url} did not contain the expected page title")

    assets = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', html)
    if not assets:
        failures.append(f"Site {site_url} referenced no bundled JS or CSS assets")
    # Vite emits root-absolute asset paths that already include the Pages base
    # path, so they must be resolved against the origin rather than the site URL
    # to avoid duplicating the base segment.
    parsed_site = urllib.parse.urlparse(site_url)
    origin = f"{parsed_site.scheme}://{parsed_site.netloc}"
    for asset in assets:
        if asset.startswith("http"):
            asset_url = asset
        else:
            asset_url = urllib.parse.urljoin(origin, asset)
        asset_status, _ = fetch(asset_url)
        if asset_status != 200:
            failures.append(f"Asset {asset_url} returned HTTP {asset_status}")
    print(f"Site OK: {site_url} (HTTP 200, {len(assets)} asset(s) resolved)")
    return assets


def check_payload(
    name: str,
    data_base_url: str,
    failures: list[str],
) -> dict | None:
    """Fetch and validate one published JSON artifact."""
    url = f"{data_base_url}/{name}"
    status, body = fetch(url)
    if status != 200:
        failures.append(f"{url} returned HTTP {status}")
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        failures.append(f"{url} is not valid JSON: {exc}")
        return None
    print(f"{name} OK: {len(body)} bytes from {url}")
    return payload


def check_freshness(
    payload: dict,
    max_age_days: int,
    failures: list[str],
) -> None:
    """Assert the payload describes recent data and has the expected shape."""
    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        failures.append(f"Prediction payload missing keys: {missing}")

    as_of = payload.get("model_state_as_of_date")
    if not as_of:
        failures.append("Prediction payload has no model_state_as_of_date")
        return
    try:
        as_of_date = dt.date.fromisoformat(str(as_of))
    except ValueError:
        failures.append(f"model_state_as_of_date is not an ISO date: {as_of!r}")
        return

    today = dt.datetime.now(dt.timezone.utc).date()
    age_days = (today - as_of_date).days
    print(f"Model state as of {as_of} ({age_days} day(s) old)")
    if age_days > max_age_days:
        failures.append(
            f"Published data is {age_days} days old (limit {max_age_days}); "
            "the generation workflow is probably failing"
        )

    n_matches = len(payload.get("matches") or [])
    n_future = len(payload.get("future_matches") or [])
    print(f"Completed forecasts: {n_matches}; future fixtures: {n_future}")
    if n_matches == 0:
        failures.append("Prediction payload contains no completed forecasts")

    fixture_status = payload.get("fixture_status") or {}
    if fixture_status.get("error"):
        failures.append(f"fixture_status error: {fixture_status['error']}")
    market_status = payload.get("market_status") or {}
    if market_status.get("error"):
        # Polymarket coverage is optional, so this is reported without failing.
        print(f"Note: market_status error: {market_status['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    parser.add_argument("--data-base-url", default=DEFAULT_DATA_BASE_URL)
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    args = parser.parse_args(argv)

    failures: list[str] = []
    check_site(args.site_url, failures)
    check_payload("results.json", args.data_base_url, failures)
    predictions = check_payload("predictions.json", args.data_base_url, failures)
    if predictions is not None:
        check_freshness(predictions, args.max_age_days, failures)

    if failures:
        print("\nAudit failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
            print(f"::error::{failure}")
        return 1

    print("\nAudit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
