"""Report the freshness of the generated prediction payload.

Exits non-zero only when the payload is unusable (missing or unparseable, or
without a ``model_state_as_of_date``). Recency problems are emitted as workflow
warnings instead of errors: blocking publication on staleness would hide the
last known-good forecast from the deployed site.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PAYLOAD_PATH = Path("outputs/latest/predictions.json")


def main() -> int:
    if not PAYLOAD_PATH.exists():
        print(f"::error::No prediction payload at {PAYLOAD_PATH}")
        return 1
    try:
        payload = json.loads(PAYLOAD_PATH.read_text())
    except json.JSONDecodeError as exc:
        print(f"::error::Prediction payload is not valid JSON: {exc}")
        return 1

    freshness = payload.get("data_freshness") or {}
    as_of = freshness.get("as_of")
    age_days = freshness.get("age_days")
    stale = freshness.get("stale")
    warnings = freshness.get("warnings") or []

    print(f"Data as of {as_of} ({age_days} days old, stale={stale})")
    n_matches = len(payload.get("matches") or [])
    n_future = len(payload.get("future_matches") or [])
    print(f"Completed forecasts: {n_matches}; future fixtures: {n_future}")

    # Annotations make warnings visible on the run summary without failing it.
    for warning in warnings:
        print(f"::warning::{warning}")

    if not as_of:
        print("::error::Generated payload has no model_state_as_of_date")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
