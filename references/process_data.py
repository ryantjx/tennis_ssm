# from datetime import datetime

# import numpy as np
# import pandas as pd
# import polars as pl
# from jax import numpy as jnp

# from cuthberto_carlos.data import DATA_URL
from cuthberto_carlos.data_types import ResultData

from datetime import datetime
import polars as pl
import jax.numpy as jnp

ORIGIN_DATE = "1872-11-30"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def process_data_pl(
    train_start: str = "2000-01-01",
    train_end: str = "2024-01-01",
    test_end: str | None = None,
    max_goals: int = 8,
):
    """Process football match data into training and optional test sets.

    Args:
        train_start: Start date for training data.
        train_end: End date for training data (also start date for test data).
        test_end: End date for test data. If None, returns only training data.
        max_goals: Maximum goals filter for completed matches.

    Returns:
        If test_end is provided: (train_pl, train_jax, test_pl, test_jax, id_to_name)
        If test_end is None: (train_pl, train_jax, id_to_name)
    """
    # Determine overall date range to load
    end_date = test_end if test_end else train_end

    data = pl.read_csv(DATA_URL, null_values=["NA"])

    # Initial filtering by date and goals
    # First filter by date only (future matches have NA scores)
    data = data.filter(
        (pl.col("date") >= train_start),
        (pl.col("date") <= end_date),
    ).with_columns(
        # Fill NULL scores with -1 (indicates future/unplayed matches)
        pl.col("home_score").fill_null(-1),
        pl.col("away_score").fill_null(-1),
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
        pl.col("tournament").str.contains("Friendly").alias("friendly")
    ).with_columns(
        (pl.col("date") - pl.col("date").min()).dt.total_days().alias("timestamp_days"),
    )

    # Now filter by goals (only for completed matches with scores >= 0)
    data = data.filter(
        (pl.col("home_score") <= max_goals) | (pl.col("home_score") == -1),
        (pl.col("away_score") <= max_goals) | (pl.col("away_score") == -1),
    )

    # Build team mappings from ALL data (train + test) to ensure consistency
    team_names = sorted(set(
        data.select(pl.col("home_team")).to_numpy().flatten()
    ) | set(
        data.select(pl.col("away_team")).to_numpy().flatten()
    ))
    teams_name_to_id = {name: i for i, name in enumerate(team_names)}
    teams_id_to_name = {i: name for i, name in enumerate(team_names)}

    # Build lookup of previous match date per team
    prev_dates = (
        pl.concat([
            data.select(pl.col("home_team").alias("team"), pl.col("date")),
            data.select(pl.col("away_team").alias("team"), pl.col("date")),
        ])
        .unique()
        .sort(["team", "date"])
        .with_columns(pl.col("date").shift(1).over("team").alias("prev_date"))
    )

    # Left join previous dates for home and away.
    # Store the ABSOLUTE timestamp (days since origin) of the previous match,
    # not the delta, so that the model can compute dt = timestamp - timestamp_previous.
    # A value of 0 means "no previous match" (sentinel), since the origin is 1872-11-30
    # and no real match has timestamp 0.
    data = (
        data
        .join(prev_dates, left_on=["home_team", "date"], right_on=["team", "date"], how="left")
        .rename({"prev_date": "home_prev_date"})
        .join(prev_dates, left_on=["away_team", "date"], right_on=["team", "date"], how="left")
        .rename({"prev_date": "away_prev_date"})
        .with_columns(
            (pl.col("home_prev_date") - pl.col("date").min()).dt.total_days().fill_null(0).alias("home_timestamp_previous"),
            (pl.col("away_prev_date") - pl.col("date").min()).dt.total_days().fill_null(0).alias("away_timestamp_previous"),
        )
        .drop(["home_prev_date", "away_prev_date"])
    )

    # Convert teams to IDs using the consistent mapping
    data = data.with_columns(
        pl.col("home_team").replace(teams_name_to_id).cast(pl.Int64).alias("home_team_id"),
        pl.col("away_team").replace(teams_name_to_id).cast(pl.Int64).alias("away_team_id"),
    )

    # Split into train and test sets
    # Convert string dates to polars Date for comparison
    train_end_date = pl.lit(train_end).str.strptime(pl.Date, format="%Y-%m-%d")
    train_data = data.filter(pl.col("date") <= train_end_date)
    if test_end:
        test_data = data.filter(pl.col("date") > train_end_date)
    else:
        test_data = None

    def _to_jax_data(pl_df):
        """Convert polars DataFrame to ResultData."""
        return ResultData(
            match_index=jnp.arange(pl_df.height),
            home_team_id=jnp.array(pl_df.select(pl.col("home_team_id")).to_numpy().flatten()),
            away_team_id=jnp.array(pl_df.select(pl.col("away_team_id")).to_numpy().flatten()),
            home_score=jnp.array(pl_df.select(pl.col("home_score")).to_numpy().flatten()),
            away_score=jnp.array(pl_df.select(pl.col("away_score")).to_numpy().flatten()),
            neutral=jnp.array(pl_df.select(pl.col("neutral")).to_numpy().flatten()),
            friendly=jnp.array(pl_df.select(pl.col("friendly")).to_numpy().flatten()),
            timestamp=jnp.array(pl_df.select(pl.col("timestamp_days")).to_numpy().flatten()),
            home_timestamp_previous=jnp.array(pl_df.select(pl.col("home_timestamp_previous")).to_numpy().flatten()),
            away_timestamp_previous=jnp.array(pl_df.select(pl.col("away_timestamp_previous")).to_numpy().flatten()),
        )

    train_jax = _to_jax_data(train_data)

    if test_end:
        test_jax = _to_jax_data(test_data)
        return train_data, train_jax, test_data, test_jax, teams_id_to_name
    else:
        return train_data, train_jax, teams_id_to_name


def main():
    # Example: Train on 2020-2024, test on 2024-2026
    pl_data, jax_data, pl_data_future, jax_data_future, id_to_name = process_data_pl(
        train_start="2020-01-01",
        train_end="2026-06-10",
        test_end="2026-07-19",
        max_goals=8,
    )
    print(f"Training matches: {len(pl_data)}")
    print(f"Test matches: {len(pl_data_future)}")
    print(f"Number of teams: {len(id_to_name)}")


if __name__ == "__main__":
    main()