import datetime as dt
import unittest

import polars as pl

from main import (
    apply_market_predictions,
    build_predictions_payload,
    build_results_json,
    extract_polymarket_moneylines,
    generate_fixture_predictions,
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
        self.assertEqual(row["id"], "future-match-1")
        self.assertEqual(row["source"], "wta_api")
        self.assertEqual(row["tournament"], "Test Open")
        self.assertEqual(row["match_status"], "in_progress")
        self.assertEqual(row["player1_rank"], 4)
        self.assertEqual(row["player2_rank"], 9)
        self.assertTrue(row["is_future"])
        self.assertIsNone(row["actual_winner"])
        self.assertAlmostEqual(row["p_player1_win"] + row["p_player2_win"], 1.0, places=4)

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
            final_metrics={
                "n_test_matches": 1,
                "accuracy": 0.5,
                "avg_log_score": -0.69,
                "uniform_baseline": -0.6931,
            },
            top_players=[],
            matches=[],
            future_matches=[],
            fixture_status={"source": "wta_api"},
        )

        validate_predictions_payload(payload)
        self.assertIn("optimization", payload)
        self.assertIn("matches", payload)
        self.assertIn("future_matches", payload)
        self.assertIn("market_status", payload)
        self.assertEqual(payload["metrics"]["n_future_matches"], 0)

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

        payload = build_results_json(
            data,
            current_matches=[
                {
                    "id": "future-match-1",
                    "date": "2026-07-09",
                    "player1": "Player One",
                    "player2": "Player Two",
                    "predicted_winner": "Player One",
                    "p_player1_win": 0.6,
                    "p_player2_win": 0.4,
                    "confidence": 0.6,
                    "match_status": "in_progress",
                    "match_state": "P",
                    "tournament": "Test Open",
                    "location": "London",
                    "tier": "WTA500",
                    "surface": "Grass",
                    "round": "Semifinals",
                    "source": "wta_api",
                    "source_match_id": "match-1",
                }
            ],
        )

        self.assertEqual(payload["source"], "tennis-data.co.uk WTA historical results")
        self.assertEqual(len(payload["results"]), 1)
        result = payload["results"][0]
        self.assertEqual(result["date"], "2026-01-04")
        self.assertEqual(result["winner"], "Winner A")
        self.assertEqual(result["loser"], "Loser B")
        self.assertEqual(result["surface"], "Hard")
        self.assertEqual(len(payload["current_matches"]), 1)
        self.assertEqual(payload["current_matches"][0]["match_status"], "in_progress")


if __name__ == "__main__":
    unittest.main()
