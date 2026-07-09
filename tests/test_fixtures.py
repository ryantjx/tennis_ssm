import datetime as dt
import unittest
from unittest.mock import patch

from src.data.fixture_common import filter_known_fixtures
from src.data.fixtures_men import (
    atp_player_keys,
    load_atp_fixtures,
    normalize_atp_draw_rows,
    normalize_atp_match_rows,
)
from src.data.fixtures_womens import (
    normalize_wta_match_rows,
    wta_player_keys,
)


SAMPLE_TOURNAMENT = {
    "name": "The Championships, Wimbledon",
    "year": 2026,
    "surface": "Grass",
    "tournamentGroup": {
        "id": 904,
        "name": "WIMBLEDON",
        "level": "Grand Slam",
    },
}

SAMPLE_MATCHES = [
    {
        "MatchID": "LS72320488",
        "DrawMatchType": "S",
        "MatchState": "F",
        "MatchTimeStamp": "2026-07-08T12:00:00+00:00",
        "RoundID": "Q",
        "PlayerNameFirstA": "Linda",
        "PlayerNameLastA": "Noskova",
        "PlayerNameFirstB": "Elise",
        "PlayerNameLastB": "Mertens",
    },
    {
        "MatchID": "LD72320730",
        "DrawMatchType": "D",
        "MatchState": "U",
        "MatchTimeStamp": "2026-07-09T11:00+01:00",
        "RoundID": "Q",
        "PlayerNameFirstA": "Katerina",
        "PlayerNameLastA": "Siniakova",
        "PlayerNameFirstB": "Hanyu",
        "PlayerNameLastB": "Guo",
    },
    {
        "MatchID": "LS72320486",
        "DrawMatchType": "S",
        "MatchState": "U",
        "MatchTimeStamp": "2026-07-09T13:30+01:00",
        "RoundID": "S",
        "PlayerNameFirstA": "Karolina",
        "PlayerNameLastA": "Muchova",
        "PlayerNameFirstB": "Coco",
        "PlayerNameLastB": "Gauff",
    },
    {
        "MatchID": "LS72320484",
        "DrawMatchType": "S",
        "MatchState": "U",
        "MatchTimeStamp": "2026-07-09T14:40+01:00",
        "RoundID": "S",
        "PlayerNameFirstA": "Marta",
        "PlayerNameLastA": "Kostyuk",
        "PlayerNameFirstB": "Linda",
        "PlayerNameLastB": "Noskova",
    },
]

SAMPLE_ATP_TOURNAMENT = {
    "id": 540,
    "year": 2026,
    "name": "Wimbledon",
    "gender": "JOINT",
    "type": "GS",
    "location": "England",
    "surface": "Grass",
    "info": {"title": "Wimbledon", "city": "London"},
}

SAMPLE_ATP_DRAWS = {
    "MS": {
        "Rounds": [
            {
                "RoundName": "Semifinal",
                "ShortName": "SF",
                "Fixtures": [
                    {
                        "MatchCode": "MSSR72318982",
                        "IsTopKnown": True,
                        "IsBottomKnown": True,
                        "Winner": 0,
                        "ResultString": "",
                        "PulseStatus": "C",
                        "DrawLineTop": {
                            "Players": [
                                {
                                    "PlayerId": "S0AG",
                                    "FirstName": "Jannik",
                                    "LastName": "Sinner",
                                }
                            ]
                        },
                        "DrawLineBottom": {
                            "Players": [
                                {
                                    "PlayerId": "D643",
                                    "FirstName": "Novak",
                                    "LastName": "Djokovic",
                                }
                            ]
                        },
                        "Match": {
                            "MatchId": "MSSR72318982",
                            "PlayerTeam1": {
                                "PlayerId": "S0AG",
                                "PlayerFirstNameFull": "Jannik",
                                "PlayerLastName": "Sinner",
                            },
                            "PlayerTeam2": {
                                "PlayerId": "D643",
                                "PlayerFirstNameFull": "Novak",
                                "PlayerLastName": "Djokovic",
                            },
                        },
                    },
                    {
                        "MatchCode": "MSSR72318994",
                        "IsTopKnown": True,
                        "IsBottomKnown": False,
                        "Winner": 0,
                        "ResultString": "",
                    },
                    {
                        "MatchCode": "MSSR72318000",
                        "IsTopKnown": True,
                        "IsBottomKnown": True,
                        "Winner": 1,
                        "ResultString": "6/4 6/4 6/4",
                    },
                ],
            }
        ]
    },
    "MD": {
        "Rounds": [
            {
                "RoundName": "Final",
                "Fixtures": [
                    {
                        "MatchCode": "MDSR72320608",
                        "IsTopKnown": True,
                        "IsBottomKnown": True,
                        "Winner": 0,
                    }
                ],
            }
        ]
    },
}

SAMPLE_ATP_MATCHES = [
    {
        "MatchId": "MS001",
        "MatchDate": "2026-07-09T13:30:00",
        "Status": "S",
        "Round": {"LongName": "Semifinal", "ShortName": "SF"},
        "PlayerTeam1": {
            "PlayerId": "S0AG",
            "PlayerFirstNameFull": "Jannik",
            "PlayerLastName": "Sinner",
        },
        "PlayerTeam2": {
            "PlayerId": "D643",
            "PlayerFirstNameFull": "Novak",
            "PlayerLastName": "Djokovic",
        },
    },
    {
        "MatchId": "MD001",
        "MatchDate": "2026-07-09T11:00:00",
        "Status": "S",
        "PlayerTeam1": {
            "PlayerFirstNameFull": "Harri",
            "PlayerLastName": "Heliovaara",
            "PartnerId": "P0G6",
        },
        "PlayerTeam2": {
            "PlayerFirstNameFull": "Thanasi",
            "PlayerLastName": "Kokkinakis",
            "PartnerId": "K0AZ",
        },
    },
    {
        "MatchId": "MS002",
        "MatchDate": "2026-07-09T15:00:00",
        "Status": "F",
        "ResultString": "6/4 6/4",
        "PlayerTeam1": {
            "PlayerFirstNameFull": "Arthur",
            "PlayerLastName": "Fery",
        },
        "PlayerTeam2": {
            "PlayerFirstNameFull": "Alexander",
            "PlayerLastName": "Zverev",
        },
    },
]


class FixtureNormalizationTest(unittest.TestCase):
    def test_normalizes_upcoming_wta_singles(self):
        fixtures = normalize_wta_match_rows(
            SAMPLE_MATCHES,
            SAMPLE_TOURNAMENT,
            origin_date="2022-12-31",
        )

        self.assertEqual(fixtures.height, 2)
        rows = fixtures.sort("source_match_id").iter_rows(named=True)
        first = next(rows)
        second = next(rows)

        expected_timestamp = (
            dt.date.fromisoformat("2026-07-09") - dt.date.fromisoformat("2022-12-31")
        ).days

        self.assertEqual(first["source_match_id"], "LS72320484")
        self.assertEqual(first["player1_full_name"], "Marta Kostyuk")
        self.assertEqual(first["player2_full_name"], "Linda Noskova")
        self.assertEqual(first["player1"], "Kostyuk M")
        self.assertEqual(first["player2"], "Noskova L")
        self.assertEqual(first["timestamp"], expected_timestamp)
        self.assertEqual(first["round"], "Semifinals")
        self.assertEqual(first["tier"], "Grand Slam")
        self.assertEqual(first["surface"], "Grass")

        self.assertEqual(second["source_match_id"], "LS72320486")
        self.assertEqual(second["player1_full_name"], "Karolina Muchova")
        self.assertEqual(second["player2_full_name"], "Coco Gauff")
        self.assertEqual(second["player1"], "Muchova K")
        self.assertEqual(second["player2"], "Gauff C")

    def test_filters_unknown_model_players(self):
        fixtures = normalize_wta_match_rows(SAMPLE_MATCHES, SAMPLE_TOURNAMENT)
        known = filter_known_fixtures(
            fixtures,
            {
                "Muchova K": 10,
                "Gauff C": 11,
                "Kostyuk M": 12,
            },
        )

        self.assertEqual(known.height, 1)
        row = known.row(0, named=True)
        self.assertEqual(row["source_match_id"], "LS72320486")
        self.assertEqual(row["player1_id"], 10)
        self.assertEqual(row["player2_id"], 11)

    def test_player_alt_key_matches_tennis_data_longer_initials(self):
        primary, alt = wta_player_keys("Xinyu", "Wang")

        self.assertEqual(primary, "Wang X")
        self.assertEqual(alt, "Wang Xin")

    def test_normalizes_atp_draw_known_unplayed_singles(self):
        fixtures = normalize_atp_draw_rows(
            SAMPLE_ATP_DRAWS,
            SAMPLE_ATP_TOURNAMENT,
            start_date="2026-07-08",
            end_date="2026-07-10",
            origin_date="2022-12-31",
        )

        self.assertEqual(fixtures.height, 1)
        row = fixtures.row(0, named=True)
        self.assertEqual(row["tour"], "ATP")
        self.assertEqual(row["source"], "tennistv_draws")
        self.assertEqual(row["source_match_id"], "MSSR72318982")
        self.assertEqual(row["player1_full_name"], "Jannik Sinner")
        self.assertEqual(row["player2_full_name"], "Novak Djokovic")
        self.assertEqual(row["player1"], "Sinner J")
        self.assertEqual(row["player2"], "Djokovic N")
        self.assertIsNone(row["date"])
        self.assertIsNone(row["timestamp"])
        self.assertEqual(row["date_source"], "draw_unknown")

    def test_normalizes_atp_match_rows_with_dates_only(self):
        fixtures = normalize_atp_match_rows(
            SAMPLE_ATP_MATCHES,
            SAMPLE_ATP_TOURNAMENT,
            start_date="2026-07-08",
            end_date="2026-07-10",
            origin_date="2022-12-31",
        )

        expected_timestamp = (
            dt.date.fromisoformat("2026-07-09") - dt.date.fromisoformat("2022-12-31")
        ).days

        self.assertEqual(fixtures.height, 1)
        row = fixtures.row(0, named=True)
        self.assertEqual(row["source"], "tennistv_matches")
        self.assertEqual(row["source_match_id"], "MS001")
        self.assertEqual(row["date"], "2026-07-09")
        self.assertEqual(row["timestamp"], expected_timestamp)
        self.assertEqual(row["round"], "Semifinal")
        self.assertEqual(row["source_event"], "MS")

    def test_atp_player_alt_key_matches_tennis_data_longer_initials(self):
        primary, alt = atp_player_keys("Alexander", "Zverev")

        self.assertEqual(primary, "Zverev A")
        self.assertEqual(alt, "Zverev Ale")

    def test_load_atp_fixtures_defaults_to_dated_match_rows(self):
        with (
            patch(
                "src.data.fixtures_men.fetch_tennis_tv_tournaments",
                return_value=[SAMPLE_ATP_TOURNAMENT],
            ),
            patch(
                "src.data.fixtures_men.fetch_tennis_tv_tournament_matches",
                return_value={"matches": SAMPLE_ATP_MATCHES},
            ),
            patch(
                "src.data.fixtures_men.fetch_tennis_tv_tournament_draws",
                return_value=SAMPLE_ATP_DRAWS,
            ),
        ):
            fixtures = load_atp_fixtures(
                start_date="2026-07-08",
                end_date="2026-07-10",
                origin_date="2022-12-31",
            )

        self.assertEqual(fixtures.height, 1)
        row = fixtures.row(0, named=True)
        self.assertEqual(row["source"], "tennistv_matches")
        self.assertEqual(row["date"], "2026-07-09")
        self.assertEqual(row["timestamp"], 1286)
        self.assertEqual(set(fixtures["tour"].to_list()), {"ATP"})

    def test_load_atp_fixtures_can_include_draw_unknown_research_rows(self):
        with (
            patch(
                "src.data.fixtures_men.fetch_tennis_tv_tournaments",
                return_value=[SAMPLE_ATP_TOURNAMENT],
            ),
            patch(
                "src.data.fixtures_men.fetch_tennis_tv_tournament_matches",
                return_value={"matches": SAMPLE_ATP_MATCHES},
            ),
            patch(
                "src.data.fixtures_men.fetch_tennis_tv_tournament_draws",
                return_value=SAMPLE_ATP_DRAWS,
            ),
        ):
            fixtures = load_atp_fixtures(
                start_date="2026-07-08",
                end_date="2026-07-10",
                origin_date="2022-12-31",
                include_draw_unknown_dates=True,
            )

        self.assertEqual(fixtures.height, 2)
        self.assertEqual(set(fixtures["source"].to_list()), {"tennistv_matches", "tennistv_draws"})
        self.assertEqual(set(fixtures["tour"].to_list()), {"ATP"})


if __name__ == "__main__":
    unittest.main()
