"""Tests for the optional Live Tennis API fixture source.

Every test runs against spec-shaped stub records built from the v1.1.0 OpenAPI
description. No network call is made and no live API key was available, so
these assert the contract this repo depends on, not the provider's live data.
"""

import datetime as dt
import unittest
from unittest.mock import patch

from src.data.fixture_common import FIXTURE_COLUMNS
from src.data.fixtures_livetennisapi import (
    TOUR_CHOICES,
    livetennisapi_player_keys,
    load_livetennisapi_fixtures,
    normalize_livetennisapi_fixture_rows,
    resolve_api_key,
    resolve_livetennisapi_fixtures,
)

ORIGIN = "2022-12-31"

# Shaped after components/schemas/Fixture: id, event_date, tour, tournament,
# round, surface, player1_name, player2_name, status -- all nullable but `id`.
SAMPLE_FIXTURES = [
    {
        "id": 900001,
        "event_date": "2026-08-05",
        "tour": "wta",
        "tournament": "Cincinnati Open",
        "round": "Quarterfinals",
        "surface": "Hard",
        "player1_name": "Linda Noskova",
        "player2_name": "Coco Gauff",
        "status": "scheduled",
    },
    {
        # Doubles team: the record's own tour is documented to be UPPERCASE for
        # a team. Singles-only model, so this must be dropped.
        "id": 900002,
        "event_date": "2026-08-05",
        "tour": "WTA",
        "tournament": "Cincinnati Open",
        "round": "Round of 16",
        "surface": "Hard",
        "player1_name": "Katerina Siniakova",
        "player2_name": "Taylor Townsend",
        "status": "scheduled",
    },
    {
        # Slash-separated pairing is a doubles entry regardless of tour casing.
        "id": 900003,
        "event_date": "2026-08-06",
        "tour": "wta",
        "tournament": "Cincinnati Open",
        "round": "Round of 16",
        "surface": "Hard",
        "player1_name": "Siniakova / Townsend",
        "player2_name": "Dabrowski / Routliffe",
        "status": "scheduled",
    },
    {
        # Null event_date: cannot be placed in time, must not be defaulted.
        "id": 900004,
        "event_date": None,
        "tour": "wta",
        "tournament": "Cincinnati Open",
        "round": "Final",
        "surface": "Hard",
        "player1_name": "Iga Swiatek",
        "player2_name": "Aryna Sabalenka",
        "status": "scheduled",
    },
    {
        # Outside the requested window.
        "id": 900005,
        "event_date": "2026-09-30",
        "tour": "wta",
        "tournament": "Some Later Event",
        "round": "Round of 32",
        "surface": "Hard",
        "player1_name": "Jessica Pegula",
        "player2_name": "Emma Navarro",
        "status": "scheduled",
    },
    {
        # Sparse record: every optional field null. Must survive without
        # inventing values.
        "id": 900006,
        "event_date": "2026-08-07",
        "tour": None,
        "tournament": None,
        "round": None,
        "surface": None,
        "player1_name": "Maria Sakkari",
        "player2_name": "Elena Rybakina",
        "status": None,
    },
]


def normalize(fixtures=None, tour="wta", start="2026-08-01", end="2026-08-10"):
    return normalize_livetennisapi_fixture_rows(
        SAMPLE_FIXTURES if fixtures is None else fixtures,
        tour=tour,
        origin_date=ORIGIN,
        start_date=start,
        end_date=end,
    )


class TestNormalization(unittest.TestCase):
    def test_schema_matches_shared_fixture_columns(self):
        frame = normalize()
        self.assertEqual(frame.columns, FIXTURE_COLUMNS)

    def test_keeps_only_dated_singles_in_window(self):
        frame = normalize()
        self.assertEqual(sorted(frame["source_match_id"].to_list()), ["900001", "900006"])

    def test_row_values(self):
        row = normalize().filter(normalize()["source_match_id"] == "900001").to_dicts()[0]
        self.assertEqual(row["tour"], "WTA")
        self.assertEqual(row["date"], "2026-08-05")
        self.assertEqual(
            row["timestamp"],
            (dt.date(2026, 8, 5) - dt.date.fromisoformat(ORIGIN)).days,
        )
        self.assertEqual(row["player1"], "Noskova L")
        self.assertEqual(row["player2"], "Gauff C")
        self.assertEqual(row["player1_full_name"], "Linda Noskova")
        self.assertEqual(row["tournament"], "Cincinnati Open")
        self.assertEqual(row["surface"], "Hard")
        self.assertEqual(row["round"], "Quarterfinals")
        self.assertEqual(row["source"], "livetennisapi")
        self.assertEqual(row["date_source"], "event_date")
        self.assertEqual(row["match_state"], "scheduled")
        self.assertEqual(row["source_event"], "wta")

    def test_absent_fields_are_unknown_never_synthesised(self):
        row = normalize().filter(normalize()["source_match_id"] == "900006").to_dicts()[0]
        # The provider publishes no venue or tier on a fixture, and no
        # tournament entity to derive one from.
        self.assertEqual(row["location"], "Unknown")
        self.assertEqual(row["tier"], "Unknown")
        self.assertEqual(row["tournament"], "Unknown")
        self.assertEqual(row["surface"], "Unknown")
        self.assertEqual(row["round"], "Unknown")
        self.assertEqual(row["source_tournament_id"], "")

    def test_empty_input_returns_typed_empty_frame(self):
        frame = normalize([])
        self.assertEqual(frame.height, 0)
        self.assertEqual(frame.columns, FIXTURE_COLUMNS)

    def test_ignores_non_dict_records(self):
        frame = normalize([None, "junk", 7])
        self.assertEqual(frame.height, 0)

    def test_no_window_keeps_all_dated_singles(self):
        frame = normalize(start=None, end=None)
        self.assertEqual(
            sorted(frame["source_match_id"].to_list()),
            ["900001", "900005", "900006"],
        )

    def test_tour_label_follows_the_requested_filter(self):
        frame = normalize(
            [
                {
                    "id": 1,
                    "event_date": "2026-08-05",
                    # Granular record vocabulary; must not leak into the column.
                    "tour": "challenger_men",
                    "player1_name": "Some Player",
                    "player2_name": "Other Player",
                }
            ],
            tour="challenger",
        )
        self.assertEqual(frame["tour"].to_list(), ["CHALLENGER"])
        self.assertEqual(frame["source_event"].to_list(), ["challenger_men"])


class TestPlayerKeys(unittest.TestCase):
    def test_both_name_orders_are_produced(self):
        primary, alternate = livetennisapi_player_keys("Linda Noskova")
        self.assertEqual(primary, "Noskova L")
        self.assertEqual(alternate, "Linda N")

    def test_reversed_source_order_still_yields_the_model_key(self):
        # If the provider emits "Last First", the model key lands in the alt slot.
        _, alternate = livetennisapi_player_keys("Noskova Linda")
        self.assertEqual(alternate, "Noskova L")

    def test_multi_token_name_uses_first_and_last_token(self):
        primary, _ = livetennisapi_player_keys("Anna Karolina Schmiedlova")
        self.assertEqual(primary, "Schmiedlova A")

    def test_accents_are_stripped_to_match_the_model_key_space(self):
        primary, _ = livetennisapi_player_keys("Linda Nosková")
        self.assertEqual(primary, "Noskova L")

    def test_blank_and_single_token_names(self):
        self.assertEqual(livetennisapi_player_keys(None), ("", ""))
        self.assertEqual(livetennisapi_player_keys("   "), ("", ""))
        primary, alternate = livetennisapi_player_keys("Cher")
        self.assertEqual(primary, alternate)


class TestResolution(unittest.TestCase):
    NAME_TO_ID = {"Noskova L": 11, "Gauff C": 22, "Sakkari M": 33, "Rybakina E": 44}

    def test_resolves_forward_reading(self):
        resolved = resolve_livetennisapi_fixtures(normalize(), self.NAME_TO_ID)
        self.assertEqual(sorted(resolved["source_match_id"].to_list()), ["900001", "900006"])
        row = resolved.filter(resolved["source_match_id"] == "900001").to_dicts()[0]
        self.assertEqual(row["player1_id"], 11)
        self.assertEqual(row["player2_id"], 22)

    def test_unresolvable_players_are_dropped(self):
        resolved = resolve_livetennisapi_fixtures(normalize(), {"Noskova L": 11})
        self.assertEqual(resolved.height, 0)

    def test_ambiguous_name_order_is_dropped_not_guessed(self):
        # "Ann Li" reads as "Li A" one way and "Ann L" the other. If both are
        # real, distinct players in the model, the fixture is genuinely
        # ambiguous and must not be resolved to either.
        frame = normalize(
            [
                {
                    "id": 5,
                    "event_date": "2026-08-05",
                    "tour": "wta",
                    "player1_name": "Ann Li",
                    "player2_name": "Coco Gauff",
                }
            ]
        )
        ambiguous = {"Li A": 77, "Ann L": 88, "Gauff C": 22}
        self.assertEqual(resolve_livetennisapi_fixtures(frame, ambiguous).height, 0)
        # With only one reading known, it resolves cleanly.
        unambiguous = {"Li A": 77, "Gauff C": 22}
        self.assertEqual(resolve_livetennisapi_fixtures(frame, unambiguous).height, 1)

    def test_self_match_is_dropped(self):
        frame = normalize(
            [
                {
                    "id": 6,
                    "event_date": "2026-08-05",
                    "tour": "wta",
                    "player1_name": "Linda Noskova",
                    "player2_name": "Noskova Linda",
                }
            ]
        )
        self.assertEqual(resolve_livetennisapi_fixtures(frame, self.NAME_TO_ID).height, 0)

    def test_empty_frame_resolves_to_empty_with_id_columns(self):
        resolved = resolve_livetennisapi_fixtures(normalize([]), self.NAME_TO_ID)
        self.assertEqual(resolved.height, 0)
        self.assertIn("player1_id", resolved.columns)
        self.assertIn("player2_id", resolved.columns)


class TestLoader(unittest.TestCase):
    def test_paginates_on_has_more_and_sends_documented_params(self):
        pages = [
            {"data": SAMPLE_FIXTURES[:3], "meta": {"has_more": True, "count": 3}},
            {"data": SAMPLE_FIXTURES[3:], "meta": {"has_more": False, "count": 3}},
        ]
        calls = []

        def fake_get(path, params, api_key):
            calls.append((path, params))
            return pages[len(calls) - 1]

        with patch(
            "src.data.fixtures_livetennisapi._livetennisapi_get_json", side_effect=fake_get
        ):
            frame = load_livetennisapi_fixtures(
                tour="wta",
                start_date="2026-08-01",
                end_date="2026-08-10",
                origin_date=ORIGIN,
                api_key="test-key",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "/fixtures")
        self.assertEqual(calls[0][1], {"tour": "wta", "limit": 200, "offset": 0})
        self.assertEqual(calls[1][1], {"tour": "wta", "limit": 200, "offset": 3})
        self.assertEqual(sorted(frame["source_match_id"].to_list()), ["900001", "900006"])

    def test_stops_when_has_more_absent(self):
        with patch(
            "src.data.fixtures_livetennisapi._livetennisapi_get_json",
            return_value={"data": [], "meta": {}},
        ) as mocked:
            frame = load_livetennisapi_fixtures(tour="atp", api_key="test-key")
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(frame.height, 0)
        self.assertEqual(frame.columns, FIXTURE_COLUMNS)

    def test_rejects_tour_outside_the_documented_enum(self):
        with self.assertRaises(ValueError):
            load_livetennisapi_fixtures(tour="mixed", api_key="test-key")
        self.assertEqual(TOUR_CHOICES, ("atp", "wta", "challenger", "itf", "juniors"))

    def test_missing_key_raises_naming_the_variable(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_api_key()
        self.assertIn("LIVETENNISAPI_KEY", str(ctx.exception))

    def test_key_is_read_from_the_environment(self):
        with patch.dict("os.environ", {"LIVETENNISAPI_KEY": "env-key"}, clear=True):
            self.assertEqual(resolve_api_key(), "env-key")


class TestDefaultPipelineUntouched(unittest.TestCase):
    def test_importing_this_module_needs_no_key(self):
        # The default WTA/ATP pipeline imports src.data.fixtures, which now
        # imports this module. That must not require any configuration.
        with patch.dict("os.environ", {}, clear=True):
            import importlib

            module = importlib.import_module("src.data.fixtures")
            self.assertTrue(hasattr(module, "load_wta_fixtures"))
            self.assertTrue(hasattr(module, "load_livetennisapi_fixtures"))


if __name__ == "__main__":
    unittest.main()
