import datetime as dt
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from jax import numpy as jnp
import polars as pl

from main import (
    apply_market_predictions,
    build_match_predictions_json,
    build_predictions_payload,
    build_results_json,
    compute_data_freshness,
    evaluate_predictions,
    extract_polymarket_moneylines,
    generate_fixture_predictions,
    prune_daily_snapshots,
    validate_predictions_payload,
)
from src.data.data import TennisData
from src.data.data_types import TennisMatchMetadata
from src.model.gaussianfactorial_tennis import GaussianFactorialTennis


class PredictionExportTest(unittest.TestCase):
    def test_fixture_predictions_export_real_wta_fixture_fields(self):
        model = GaussianFactorialTennis(
            init_mean=0.0,
            init_var=1.0,
            tau=0.1,
            s=1.0,
            num_players=2,
        )
        state = model.initial_state()
        fixtures = pl.DataFrame(
            [
                {
                    "date": "2026-07-09",
                    "timestamp": 1286,
                    "player1_id": 0,
                    "player2_id": 1,
                    "player1_full_name": "Player One",
                    "player2_full_name": "Player Two",
                    "tournament": "Test Open",
                    "location": "London",
                    "tier": "WTA500",
                    "surface": "Grass",
                    "round": "Semifinals",
                    "source": "wta_api",
                    "source_match_id": "match-1",
                    "source_tournament_id": "tournament-1",
                    "match_state": "P",
                }
            ]
        )

        exported = generate_fixture_predictions(
            loaded_model=model,
            loaded_state=state,
            current_time=1285,
            fixtures=fixtures,
            id_to_name={0: "Player One", 1: "Player Two"},
            player_rankings_by_id={
                0: {"rank": 4, "skill": 1.2, "variance": 0.3},
                1: {"rank": 9, "skill": 0.8, "variance": 0.4},
            },
        )

        self.assertEqual(len(exported), 1)
        row = exported[0]
        self.assertEqual(
            row["id"],
            "future:wta_api:2026-07-09:tournament-1:match-1:0:1",
        )
        self.assertEqual(row["source"], "wta_api")
        self.assertEqual(row["source_tournament_id"], "tournament-1")
        self.assertEqual(row["tournament"], "Test Open")
        self.assertEqual(row["match_status"], "in_progress")
        self.assertEqual(row["player1_rank"], 4)
        self.assertEqual(row["player2_rank"], 9)
        self.assertTrue(row["is_future"])
        self.assertIsNone(row["actual_winner"])
        self.assertIsNone(row["brier_score"])
        self.assertAlmostEqual(row["p_player1_win"] + row["p_player2_win"], 1.0, places=4)

    def test_evaluate_predictions_reports_binary_brier_scores(self):
        predictions = SimpleNamespace(
            p_player1_win=jnp.array([0.9, 0.5, 0.1]),
            p_player2_win=jnp.array([0.1, 0.5, 0.9]),
        )

        metrics = evaluate_predictions(predictions, jnp.array([1.0, 1.0, 1.0]))

        self.assertEqual(metrics["avg_brier_score"], 0.3567)
        self.assertEqual(metrics["brier_uniform_baseline"], 0.25)

    def test_evaluate_predictions_handles_an_empty_window(self):
        predictions = SimpleNamespace(
            p_player1_win=jnp.array([]),
            p_player2_win=jnp.array([]),
        )

        metrics = evaluate_predictions(predictions, jnp.array([]))

        self.assertEqual(metrics["avg_brier_score"], 0.0)
        self.assertEqual(metrics["brier_uniform_baseline"], 0.25)

    def test_evaluate_predictions_uses_player2_win_outcomes(self):
        predictions = SimpleNamespace(
            p_player1_win=jnp.array([0.9, 0.1]),
            p_player2_win=jnp.array([0.1, 0.9]),
        )

        metrics = evaluate_predictions(predictions, jnp.array([0.0, 0.0]))

        self.assertEqual(metrics["n_correct"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["avg_brier_score"], 0.41)

    def test_completed_match_export_uses_each_actual_outcome(self):
        test_data = SimpleNamespace(
            num_matches=2,
            match_metadata=SimpleNamespace(
                tournament=["Test Open", "Test Open"],
                location=["London", "London"],
                tier=["WTA500", "WTA500"],
                surface=["Grass", "Grass"],
                round=["Final", "Final"],
            ),
        )
        test_jax = SimpleNamespace(
            player1_id=jnp.array([0, 0]),
            player2_id=jnp.array([1, 1]),
            winner=jnp.array([1.0, 0.0]),
            timestamp=jnp.array([1100, 1101]),
        )
        predictions = SimpleNamespace(
            p_player1_win=jnp.array([0.8, 0.8]),
            p_player2_win=jnp.array([0.2, 0.2]),
            player1_mean=jnp.array([[[1.0]], [[1.0]]]),
            player2_mean=jnp.array([[[0.0]], [[0.0]]]),
            player1_var=jnp.array([[0.25], [0.25]]),
            player2_var=jnp.array([[0.25], [0.25]]),
        )

        exported = build_match_predictions_json(
            test_data=test_data,
            test_jax=test_jax,
            predictions=predictions,
            id_to_name={0: "Player One", 1: "Player Two"},
        )

        self.assertEqual(exported[0]["brier_score"], 0.04)
        self.assertEqual(exported[0]["actual_winner"], "Player One")
        self.assertEqual(exported[1]["brier_score"], 0.64)
        self.assertEqual(exported[1]["actual_winner"], "Player Two")
        self.assertEqual(exported[1]["log_score"], -1.6094)

    def test_fixture_prediction_ids_include_tournament_context(self):
        model = GaussianFactorialTennis(
            init_mean=0.0,
            init_var=1.0,
            tau=0.1,
            s=1.0,
            num_players=2,
        )
        state = model.initial_state()
        shared = {
            "date": "2026-07-09",
            "timestamp": 1286,
            "player1_id": 0,
            "player2_id": 1,
            "player1_full_name": "Player One",
            "player2_full_name": "Player Two",
            "location": "London",
            "tier": "WTA500",
            "surface": "Grass",
            "round": "Semifinals",
            "source": "wta_api",
            "source_match_id": "LS004",
            "match_state": "U",
        }
        fixtures = pl.DataFrame(
            [
                {**shared, "tournament": "First Open", "source_tournament_id": "100"},
                {**shared, "tournament": "Second Open", "source_tournament_id": "200"},
            ]
        )

        exported = generate_fixture_predictions(
            loaded_model=model,
            loaded_state=state,
            current_time=1285,
            fixtures=fixtures,
            id_to_name={0: "Player One", 1: "Player Two"},
        )

        self.assertEqual(len({row["id"] for row in exported}), 2)
        self.assertIn(":100:LS004:", exported[0]["id"])
        self.assertIn(":200:LS004:", exported[1]["id"])

    def test_prediction_payload_rejects_duplicate_future_ids(self):
        payload = {
            "generated_at": "2026-07-09T12:00:00Z",
            "data_windows": {},
            "data_freshness": {
                "as_of": "2026-07-09",
                "age_days": 0,
                "threshold_days": 3,
                "stale": False,
                "warnings": [],
            },
            "model_params": {},
            "optimization": {},
            "metrics": {
                "test": {
                    "n_matches": 0,
                    "n_correct": 0,
                    "accuracy": 0.0,
                    "avg_brier_score": 0.0,
                    "brier_uniform_baseline": 0.25,
                },
                "eval": {
                    "n_matches": 0,
                    "n_correct": 0,
                    "accuracy": 0.0,
                    "avg_brier_score": 0.0,
                    "brier_uniform_baseline": 0.25,
                },
            },
            "top_players": [],
            "matches": [],
            "future_matches": [{"id": "duplicate"}, {"id": "duplicate"}],
        }

        with self.assertRaisesRegex(ValueError, "future match IDs must be unique"):
            validate_predictions_payload(payload)

    def test_prediction_payload_contract_requires_core_arrays(self):
        payload = build_predictions_payload(
            trained_seed_params={"tau": 0.1, "s": 1.0, "init_var": 1.0},
            optimization={
                "objective": "maximize_2026_test_avg_log_score",
                "best_params": {"tau": 0.1, "s": 1.0, "init_var": 1.0},
                "best_metrics": {"accuracy": 0.5, "avg_log_score": -0.69},
                "candidate_count": 1,
                "trials": [],
            },
            test_metrics={
                "n_matches": 2,
                "n_correct": 1,
                "accuracy": 0.5,
                "avg_log_score": -0.69,
                "avg_brier_score": 0.3,
                "uniform_baseline": -0.6931,
                "brier_uniform_baseline": 0.25,
            },
            eval_metrics={
                "n_matches": 1,
                "n_correct": 1,
                "accuracy": 1.0,
                "avg_log_score": -0.4,
                "avg_brier_score": 0.16,
                "uniform_baseline": -0.6931,
                "brier_uniform_baseline": 0.25,
            },
            top_players=[
                {"rank": rank, "name": f"Player {rank}", "skill": 1 / rank, "variance": 0.1}
                for rank in range(1, 53)
            ],
            matches=[
                {"date": "2026-06-27", "brier_score": 0.16},
            ],
            test_window_matches=[
                {"date": "2025-01-01"},
                {"date": "2025-12-31"},
            ],
            future_matches=[],
            fixture_status={"source": "wta_api"},
        )

        validate_predictions_payload(payload)
        self.assertIn("optimization", payload)
        self.assertIn("matches", payload)
        self.assertIn("future_matches", payload)
        self.assertIn("market_status", payload)
        self.assertEqual(payload["metrics"]["n_future_matches"], 0)
        self.assertEqual(payload["metrics"]["test"]["n_correct"], 1)
        self.assertEqual(payload["metrics"]["eval"]["accuracy"], 1.0)
        self.assertEqual(payload["metrics"]["test"]["avg_brier_score"], 0.3)
        self.assertEqual(payload["matches"][0]["brier_score"], 0.16)
        self.assertEqual(len(payload["top_players"]), 52)
        self.assertEqual(payload["data_windows"]["train_display_start"], "2022-01-01")
        self.assertEqual(payload["data_windows"]["train_display_end"], "2024-12-31")
        self.assertEqual(payload["data_windows"]["test_display_start"], "2025-01-01")
        self.assertEqual(payload["data_windows"]["test_display_end"], "2025-12-31")
        self.assertEqual(payload["data_windows"]["test_match_start"], "2025-01-01")
        self.assertEqual(payload["data_windows"]["test_match_end"], "2025-12-31")
        self.assertEqual(payload["data_windows"]["prediction_display_start"], "2026-01-01")
        self.assertEqual(payload["data_windows"]["prediction_display_end"], "2026-12-31")
        self.assertEqual(payload["data_windows"]["eval_match_start"], "2026-06-27")
        self.assertEqual(payload["data_windows"]["eval_match_end"], "2026-06-27")
        self.assertNotIn("upcoming_match_start", payload["data_windows"])
        self.assertIn("data_freshness", payload)
        self.assertIn("warnings", payload["data_freshness"])
        self.assertIsInstance(payload["data_freshness"]["stale"], bool)

    def test_prediction_payload_rejects_invalid_brier_scores(self):
        payload = {
            "generated_at": "2026-07-09T12:00:00Z",
            "data_windows": {},
            "data_freshness": {
                "as_of": "2026-07-09",
                "age_days": 0,
                "threshold_days": 3,
                "stale": False,
                "warnings": [],
            },
            "model_params": {},
            "optimization": {},
            "metrics": {
                "test": {
                    "n_matches": 1,
                    "n_correct": 1,
                    "accuracy": 1.0,
                    "avg_brier_score": 0.1,
                    "brier_uniform_baseline": 0.25,
                },
                "eval": {
                    "n_matches": 1,
                    "n_correct": 1,
                    "accuracy": 1.0,
                    "avg_brier_score": 0.1,
                    "brier_uniform_baseline": 0.25,
                },
            },
            "top_players": [],
            "matches": [{"brier_score": 1.1}],
            "future_matches": [],
        }

        with self.assertRaisesRegex(ValueError, "brier_score must be finite"):
            validate_predictions_payload(payload)

        payload["matches"][0]["brier_score"] = None
        payload["metrics"]["test"]["avg_brier_score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "avg_brier_score must be finite"):
            validate_predictions_payload(payload)

    def test_polymarket_moneyline_matches_fixture_by_player_pair(self):
        events = [
            {
                "id": "680884",
                "title": "Wimbledon WTA: Karolina Muchova vs Coco Gauff",
                "slug": "wta-muchova-gauff-2026-07-09",
                "updatedAt": "2026-07-09T12:00:00Z",
                "markets": [
                    {
                        "id": "market-1",
                        "question": "Wimbledon WTA: Karolina Muchova vs Coco Gauff",
                        "slug": "moneyline",
                        "sportsMarketType": "moneyline",
                        "outcomes": '["Karolina Muchova", "Coco Gauff"]',
                        "outcomePrices": '["0.49", "0.51"]',
                        "volume": "3000000",
                        "liquidity": "356000",
                    }
                ],
            }
        ]
        markets = extract_polymarket_moneylines(events)
        matches = [
            {
                "player1": "Coco Gauff",
                "player2": "Karolina Muchova",
                "p_player1_win": 0.58,
                "p_player2_win": 0.42,
            }
        ]

        matched = apply_market_predictions(matches, markets)

        self.assertEqual(matched, 1)
        market = matches[0]["market"]
        self.assertEqual(market["event_id"], "680884")
        self.assertEqual(market["player1_price"], 0.51)
        self.assertEqual(market["player2_price"], 0.49)
        self.assertAlmostEqual(market["player1_edge"], 0.07)

    def test_completed_results_json_has_stable_match_fields(self):
        data = TennisData(
            jax_data=None,
            polars_data=pl.DataFrame(
                [
                    {
                        "Date": dt.date(2026, 1, 4),
                        "Winner": "Winner A",
                        "Loser": "Loser B",
                    }
                ]
            ),
            id_to_name={0: "Winner A", 1: "Loser B"},
            name_to_id={"Winner A": 0, "Loser B": 1},
            num_players=2,
            num_matches=1,
            match_metadata=TennisMatchMetadata(
                tournament=["Test Open"],
                location=["Auckland"],
                tier=["WTA250"],
                surface=["Hard"],
                round=["1st Round"],
            ),
        )

        payload = build_results_json(data)

        self.assertEqual(payload["source"], "tennis-data.co.uk WTA historical results")
        self.assertEqual(payload["data_window"], {"start": "2026-01-04", "end": "2026-01-04"})
        self.assertEqual(len(payload["results"]), 1)
        result = payload["results"][0]
        self.assertEqual(result["date"], "2026-01-04")
        self.assertEqual(result["winner"], "Winner A")
        self.assertEqual(result["loser"], "Loser B")
        self.assertEqual(result["surface"], "Hard")
        self.assertNotIn("current_matches", payload)


def _complete_payload(**overrides):
    """Return a minimal payload that passes structural validation."""
    payload = {
        "generated_at": "2026-07-09T12:00:00Z",
        "data_windows": {},
        "data_freshness": {
            "as_of": "2026-07-09",
            "age_days": 0,
            "threshold_days": 3,
            "stale": False,
            "warnings": [],
        },
        "model_params": {},
        "optimization": {},
        "metrics": {
            "test": {
                "n_matches": 0,
                "n_correct": 0,
                "accuracy": 0.0,
                "avg_brier_score": 0.0,
                "brier_uniform_baseline": 0.25,
            },
            "eval": {
                "n_matches": 0,
                "n_correct": 0,
                "accuracy": 0.0,
                "avg_brier_score": 0.0,
                "brier_uniform_baseline": 0.25,
            },
        },
        "top_players": [],
        "matches": [],
        "future_matches": [],
    }
    payload.update(overrides)
    return payload


class DataFreshnessTest(unittest.TestCase):
    def test_current_data_is_not_stale_and_produces_no_warnings(self):
        freshness = compute_data_freshness(
            "2026-07-09",
            result_update_status={"appended": 3},
            fixture_status={"matched_model_players": 5},
            future_matches=[{"id": "future"}],
            today=dt.date(2026, 7, 10),
        )

        self.assertFalse(freshness["stale"])
        self.assertEqual(freshness["age_days"], 1)
        self.assertEqual(freshness["warnings"], [])

    def test_old_data_is_flagged_stale_with_an_explanatory_warning(self):
        freshness = compute_data_freshness(
            "2026-06-01",
            fixture_status={"matched_model_players": 5},
            future_matches=[{"id": "future"}],
            today=dt.date(2026, 7, 10),
        )

        self.assertTrue(freshness["stale"])
        self.assertEqual(freshness["age_days"], 39)
        self.assertTrue(any("39 days old" in warning for warning in freshness["warnings"]))

    def test_staleness_is_a_warning_not_an_exception(self):
        # The 52-day outage taught us that blocking publication hides the last
        # known-good forecast, so staleness must never raise.
        freshness = compute_data_freshness(
            "2026-01-01",
            future_matches=[{"id": "future"}],
            today=dt.date(2026, 7, 10),
        )

        self.assertTrue(freshness["stale"])

    def test_missing_or_invalid_as_of_date_is_reported(self):
        missing = compute_data_freshness(None, today=dt.date(2026, 7, 10))
        self.assertIsNone(missing["age_days"])
        self.assertFalse(missing["stale"])
        self.assertTrue(any("missing" in warning for warning in missing["warnings"]))

        invalid = compute_data_freshness("not-a-date", today=dt.date(2026, 7, 10))
        self.assertTrue(any("not a valid ISO date" in warning for warning in invalid["warnings"]))

    def test_upstream_errors_and_empty_outputs_are_surfaced(self):
        freshness = compute_data_freshness(
            "2026-07-09",
            result_update_status={"error": "HTTP Error 403: Forbidden"},
            fixture_status={"matched_model_players": 0},
            market_status={"error": "timeout"},
            future_matches=[],
            today=dt.date(2026, 7, 9),
        )

        warnings = " | ".join(freshness["warnings"])
        self.assertIn("result_update_status reported an error", warnings)
        self.assertIn("market_status reported an error", warnings)
        self.assertIn("no future fixture predictions", warnings)
        self.assertIn("no WTA fixtures matched", warnings)


class DailySnapshotRetentionTest(unittest.TestCase):
    def setUp(self):
        self.daily_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for entry in self.daily_dir.iterdir():
            if entry.is_dir():
                for child in entry.iterdir():
                    child.unlink()
                entry.rmdir()
            else:
                entry.unlink()
        self.daily_dir.rmdir()

    def _add_snapshot(self, name: str) -> None:
        snapshot = self.daily_dir / name
        snapshot.mkdir()
        (snapshot / "predictions.json").write_text("{}")

    def test_prunes_snapshots_older_than_the_retention_window(self):
        for name in ("2026-06-01", "2026-06-20", "2026-07-08", "2026-07-09"):
            self._add_snapshot(name)

        removed = prune_daily_snapshots(
            keep_days=30, daily_dir=self.daily_dir, today=dt.date(2026, 7, 10)
        )

        self.assertEqual(removed, ["2026-06-01"])
        self.assertEqual(
            sorted(entry.name for entry in self.daily_dir.iterdir()),
            ["2026-06-20", "2026-07-08", "2026-07-09"],
        )

    def test_ignores_directories_that_are_not_iso_dates(self):
        self._add_snapshot("archive")
        self._add_snapshot("2026-07-09")

        removed = prune_daily_snapshots(
            keep_days=1, daily_dir=self.daily_dir, today=dt.date(2026, 7, 10)
        )

        self.assertEqual(removed, [])
        self.assertTrue((self.daily_dir / "archive").is_dir())

    def test_missing_daily_directory_is_not_an_error(self):
        removed = prune_daily_snapshots(
            keep_days=30,
            daily_dir=self.daily_dir / "absent",
            today=dt.date(2026, 7, 10),
        )

        self.assertEqual(removed, [])

    def test_retains_everything_when_window_is_large(self):
        for name in ("2026-01-01", "2026-07-09"):
            self._add_snapshot(name)

        removed = prune_daily_snapshots(
            keep_days=3650, daily_dir=self.daily_dir, today=dt.date(2026, 7, 10)
        )

        self.assertEqual(removed, [])
        self.assertEqual(len(list(self.daily_dir.iterdir())), 2)


class PayloadFreshnessContractTest(unittest.TestCase):
    def test_payload_requires_data_freshness(self):
        payload = _complete_payload()
        del payload["data_freshness"]

        with self.assertRaisesRegex(ValueError, "data_freshness"):
            validate_predictions_payload(payload)

    def test_payload_requires_freshness_fields(self):
        payload = _complete_payload(data_freshness={"as_of": "2026-07-09"})

        with self.assertRaisesRegex(ValueError, "data_freshness missing required fields"):
            validate_predictions_payload(payload)

    def test_stale_payload_still_validates(self):
        # Staleness must warn, never raise, so the site keeps the last forecast.
        payload = _complete_payload(
            data_freshness={
                "as_of": "2026-01-01",
                "age_days": 190,
                "threshold_days": 3,
                "stale": True,
                "warnings": ["newest completed observation is 190 days old"],
            }
        )

        validate_predictions_payload(payload)


if __name__ == "__main__":
    unittest.main()
