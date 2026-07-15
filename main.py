"""End-to-end workflow for WTA state-space tennis predictions.

The canonical generated prediction artifact is ``outputs/predictions.json``.
The frontend receives a copied, ignored runtime copy at
``frontend/public/data/predictions.json`` during local generation and deploys.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import jax
from jax import numpy as jnp, tree_util as jtu
import polars as pl

from src.data.data import (
    TennisData,
    append_completed_results,
    load_wta,
    most_recent_timestamp_by_player,
)
from src.data.data_types import TennisDynamicsOnlyData, TennisMatchMetadata, WTATennisResults
from src.data.fixture_common import filter_known_fixtures
from src.data.fixtures_womens import load_wta_completed_results, load_wta_fixtures
from src.model.gaussianfactorial_tennis import GaussianFactorialTennis, TennisMatchPrediction, train


ORIGIN_DATE = "2021-12-31"
TRAIN_START = "2021-12-31"
TRAIN_END = "2024-12-31"
TEST_START = "2024-12-31"
TEST_END = "2025-12-31"
EVAL_START = "2025-12-31"
EVAL_END = "2026-12-31"
TRAIN_DISPLAY_START = "2022-01-01"
TRAIN_DISPLAY_END = "2024-12-31"
TEST_DISPLAY_START = "2025-01-01"
TEST_DISPLAY_END = "2025-12-31"
EVAL_DISPLAY_START = "2026-01-01"
EVAL_DISPLAY_END = "2026-12-31"

OUTPUT_DIR = Path("outputs")
STATE_FILE = OUTPUT_DIR / "tennis_factorial_state.json"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.json"
LATEST_OUTPUT_DIR = OUTPUT_DIR / "latest"
DAILY_OUTPUT_DIR = OUTPUT_DIR / "daily"
LATEST_PREDICTIONS_FILE = LATEST_OUTPUT_DIR / "predictions.json"
LATEST_RESULTS_FILE = LATEST_OUTPUT_DIR / "results.json"
FRONTEND_DATA_DIR = Path("frontend/public/data")
FRONTEND_PREDICTIONS_FILE = FRONTEND_DATA_DIR / "predictions.json"
RESULTS_FILE = FRONTEND_DATA_DIR / "results.json"
WTA_ACTIVE_MATCH_STATES = ("U", "P", "L", "I", "S")
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLYMARKET_TENNIS_URL = "https://polymarket.com/predictions/tennis"
POLYMARKET_TAG_SLUG = "tennis"
WTA_COMPLETED_LOOKBACK_DAYS = 30


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    daily_output_dir = DAILY_OUTPUT_DIR / run_date
    daily_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. Loading WTA historical data and recent completed results")
    print("=" * 70)
    historical_data = load_wta(
        start_date=TRAIN_START,
        end_date=EVAL_END,
        origin_date=ORIGIN_DATE,
    )
    historical_data, result_update_status = load_latest_completed_wta_results(historical_data)
    test_indices = match_indices_between(historical_data, TEST_START, TEST_END)
    eval_indices = match_indices_between(historical_data, EVAL_START, EVAL_END)
    test_data = slice_tennis_data(historical_data, test_indices)
    eval_data = slice_tennis_data(historical_data, eval_indices)
    print(f"  Historical matches: {historical_data.num_matches}")
    print(f"  Newly appended WTA API results: {result_update_status['appended']}")
    print(f"  2025 validation matches: {test_data.num_matches}")
    print(f"  2026 completed matches: {eval_data.num_matches}")
    print()

    print("=" * 70)
    print("2. Loading saved model parameters")
    print("=" * 70)
    best_params = load_saved_model_params(STATE_FILE)
    print(
        "  Loaded parameters: "
        f"tau={best_params['tau']:.6f}, "
        f"s={best_params['s']:.6f}, "
        f"init_var={best_params['init_var']:.6f}"
    )
    print()

    print("=" * 70)
    print("3. Filtering historical matches with saved parameters")
    print("=" * 70)
    final_model = GaussianFactorialTennis(
        init_mean=0.0,
        init_var=best_params["init_var"],
        tau=best_params["tau"],
        s=best_params["s"],
        num_players=historical_data.num_players,
    )
    filtered_state, historical_predictions = filter_and_predict_matches(
        model=final_model,
        matches=historical_data.jax_data,
    )
    final_state, current_time, skill_means, skill_vars = synchronize_state_to_latest(
        model=final_model,
        state=filtered_state,
        data=historical_data,
    )
    final_model.save_state(final_state, str(STATE_FILE))
    print(f"  Saved state to {STATE_FILE}")
    print_top_players(historical_data, skill_means, skill_vars)
    player_rankings_by_id = build_player_rankings_by_id(historical_data, skill_means, skill_vars)
    test_predictions = slice_predictions(historical_predictions, test_indices)
    predictions = slice_predictions(historical_predictions, eval_indices)
    eval_metrics = evaluate_predictions(test_predictions)
    optimization = saved_parameter_metadata(best_params, eval_metrics)
    print()

    print("=" * 70)
    print("4. Loading WTA fixtures")
    print("=" * 70)
    future_fixtures, fixture_status = load_known_wta_future_fixtures(
        name_to_id=historical_data.name_to_id,
    )
    future_matches_json = generate_future_predictions(
        loaded_model=final_model,
        loaded_state=final_state,
        current_time=current_time,
        id_to_name=historical_data.id_to_name,
        name_to_id=historical_data.name_to_id,
        num_players=historical_data.num_players,
        player_rankings_by_id=player_rankings_by_id,
        fixtures=future_fixtures,
        use_synthetic_fallback=False,
    )
    print(f"  Latest historical timestamp: {current_time}")
    print(f"  Future fixture predictions: {len(future_matches_json)}")
    print()

    print("=" * 70)
    print("5. Exporting predictions and completed results")
    print("=" * 70)
    matches_json = build_match_predictions_json(
        test_data=eval_data,
        test_jax=eval_data.jax_data,
        predictions=predictions,
        id_to_name=historical_data.id_to_name,
        player_rankings_by_id=player_rankings_by_id,
    )
    test_window_json = build_data_window_json(test_data)
    top_players_json = build_top_players_json(historical_data, skill_means, skill_vars)
    polymarket_matches, market_status = load_polymarket_tennis_markets()
    market_status["matched_model_matches"] = apply_market_predictions(
        [*future_matches_json, *matches_json],
        polymarket_matches,
    )
    print(f"  Matched Polymarket prices to {market_status['matched_model_matches']} model matches")
    output_json = build_predictions_payload(
        trained_seed_params=best_params,
        optimization=optimization,
        final_metrics=eval_metrics,
        top_players=top_players_json,
        matches=matches_json,
        test_window_matches=test_window_json,
        future_matches=future_matches_json,
        fixture_status=fixture_status,
        market_status=market_status,
        model_state_as_of=current_time,
        result_update_status=result_update_status,
    )
    validate_predictions_payload(output_json)

    PREDICTIONS_FILE.write_text(json.dumps(output_json, indent=2) + "\n")
    LATEST_PREDICTIONS_FILE.write_text(json.dumps(output_json, indent=2) + "\n")
    (daily_output_dir / "predictions.json").write_text(json.dumps(output_json, indent=2) + "\n")
    shutil.copyfile(PREDICTIONS_FILE, FRONTEND_PREDICTIONS_FILE)

    results_json = build_results_json(
        eval_data,
        source="tennis-data.co.uk history + WTA API completed results",
    )
    RESULTS_FILE.write_text(json.dumps(results_json, indent=2) + "\n")
    LATEST_RESULTS_FILE.write_text(json.dumps(results_json, indent=2) + "\n")
    (daily_output_dir / "results.json").write_text(json.dumps(results_json, indent=2) + "\n")

    print(f"  Saved canonical predictions to {PREDICTIONS_FILE}")
    print(f"  Saved latest prediction data to {LATEST_OUTPUT_DIR}")
    print(f"  Saved dated prediction data to {daily_output_dir}")
    print(f"  Copied frontend runtime predictions to {FRONTEND_PREDICTIONS_FILE}")
    print(f"  Saved committed completed results to {RESULTS_FILE}")
    print("=" * 70)
    print("Done")
    print("=" * 70)


def fit_and_sync_model(
    train_data: TennisData,
    params: dict[str, float],
) -> tuple[GaussianFactorialTennis, Any, int, jnp.ndarray, jnp.ndarray]:
    model = GaussianFactorialTennis(
        init_mean=0.0,
        init_var=params["init_var"],
        tau=params["tau"],
        s=params["s"],
        num_players=train_data.num_players,
    )
    final_state = model.filter(train_data.jax_data)
    skill_means = final_state.mean[:, 0]
    skill_vars = jnp.square(final_state.chol_cov[:, 0, 0])

    most_recent_ts = most_recent_timestamp_by_player(
        train_data.polars_data,
        train_data.num_players,
    )
    current_time = int(jnp.max(most_recent_ts))
    sync_data = TennisDynamicsOnlyData(
        player_id=jnp.arange(train_data.num_players),
        timestamp=jnp.broadcast_to(jnp.array(current_time), (train_data.num_players,)),
        timestamp_previous=most_recent_ts,
    )
    sync_state = model.synchronize(final_state, sync_data)
    return model, sync_state, current_time, skill_means, skill_vars


def load_latest_completed_wta_results(
    historical_data: TennisData,
) -> tuple[TennisData, dict[str, Any]]:
    """Backfill recent final WTA matches before the sequential filter runs."""
    latest_historical_date = historical_data.polars_data.select(
        pl.col("Date").max()
    ).item()
    end_date = dt.datetime.now(dt.timezone.utc).date()
    start_date = latest_historical_date - dt.timedelta(days=WTA_COMPLETED_LOOKBACK_DAYS)
    status: dict[str, Any] = {
        "source": "wta_api",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "loaded": 0,
        "appended": 0,
    }
    try:
        completed = load_wta_completed_results(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        updated, appended = append_completed_results(
            historical_data,
            completed,
            origin_date=ORIGIN_DATE,
        )
        status["loaded"] = completed.height
        status["appended"] = appended
        status["latest_completed_date"] = (
            updated.polars_data.select(pl.col("Date").max()).item().isoformat()
        )
        return updated, status
    except Exception as exc:
        status["error"] = str(exc)
        status["latest_completed_date"] = latest_historical_date.isoformat()
        print(f"  Warning: could not load recent completed WTA results: {exc}")
        return historical_data, status


def load_saved_model_params(state_file: Path) -> dict[str, float]:
    if not state_file.exists():
        raise FileNotFoundError(
            f"Saved model state not found at {state_file}. "
            "Run parameter training once before deploying."
        )
    payload = json.loads(state_file.read_text())
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Saved model state at {state_file} is missing params")
    required = {"tau", "s", "init_var"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"Saved model params missing keys: {sorted(missing)}")
    return {name: float(params[name]) for name in sorted(required)}


def saved_parameter_metadata(
    params: dict[str, float],
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "objective": "use_saved_parameters_from_state_file",
        "selection_note": (
            f"Deployment uses tau, s, and init_var from {STATE_FILE}; "
            "the filter is rerun over all completed WTA matches before "
            "future fixture predictions are generated."
        ),
        "candidate_count": 0,
        "best_params": params,
        "best_metrics": validation_metrics,
        "trials": [],
    }


def match_indices_between(data: TennisData, start_date: str, end_date: str) -> list[int]:
    start_dt = pl.lit(start_date).str.strptime(pl.Date, format="%Y-%m-%d")
    end_dt = pl.lit(end_date).str.strptime(pl.Date, format="%Y-%m-%d")
    return (
        data.polars_data
        .filter((pl.col("Date") > start_dt) & (pl.col("Date") <= end_dt))
        .select(pl.col("match_index"))
        .to_series()
        .to_list()
    )


def slice_tennis_data(data: TennisData, indices: list[int]) -> TennisData:
    index_array = jnp.array(indices)
    index_set = set(indices)
    frame = data.polars_data.filter(pl.col("match_index").is_in(index_set))
    metadata_indices = frame.select(pl.col("match_index")).to_series().to_list()
    match_metadata = TennisMatchMetadata(
        tournament=[data.match_metadata.tournament[i] for i in metadata_indices],
        location=[data.match_metadata.location[i] for i in metadata_indices],
        tier=[data.match_metadata.tier[i] for i in metadata_indices],
        surface=[data.match_metadata.surface[i] for i in metadata_indices],
        round=[data.match_metadata.round[i] for i in metadata_indices],
    )
    return TennisData(
        jax_data=jtu.tree_map(lambda x: x[index_array], data.jax_data),
        polars_data=frame,
        id_to_name=data.id_to_name,
        name_to_id=data.name_to_id,
        num_players=data.num_players,
        num_matches=len(indices),
        match_metadata=match_metadata,
    )


def slice_predictions(predictions: Any, indices: list[int]) -> Any:
    index_array = jnp.array(indices)
    return jtu.tree_map(lambda x: x[index_array], predictions)


def filter_and_predict_matches(
    model: GaussianFactorialTennis,
    matches: WTATennisResults,
) -> tuple[Any, Any]:
    from jax.lax import scan as lax_scan

    def predict_from_state(factorial_state: Any, match_data: WTATennisResults) -> Any:
        dynamics_data = TennisDynamicsOnlyData(
            player_id=jnp.array([match_data.player1_id, match_data.player2_id]),
            timestamp=jnp.array([match_data.timestamp, match_data.timestamp]),
            timestamp_previous=jnp.array(
                [
                    match_data.player1_timestamp_previous,
                    match_data.player2_timestamp_previous,
                ]
            ),
        )
        factorial_state_two = jax.vmap(
            model.factorializer.extract, in_axes=(None, 0)
        )(factorial_state, dynamics_data.player_id)
        state_prep = jax.vmap(model.single_player_filter.filter_prepare)(dynamics_data)
        propagated = jax.vmap(model.single_player_filter.filter_combine)(
            factorial_state_two, state_prep
        )

        mu_1 = propagated.mean[0, 0]
        mu_2 = propagated.mean[1, 0]
        var_1 = jnp.square(propagated.chol_cov[0, 0, 0])
        var_2 = jnp.square(propagated.chol_cov[1, 0, 0])
        skill_diff_mean = mu_1 - mu_2
        skill_diff_var = var_1 + var_2
        skill_diff_std = jnp.sqrt(skill_diff_var)
        p_player1_win = jax.nn.sigmoid(
            skill_diff_mean / jnp.sqrt(model.s**2 + skill_diff_var)
        )
        return TennisMatchPrediction(
            p_player1_win=p_player1_win,
            p_player2_win=1.0 - p_player1_win,
            skill_diff_mean=skill_diff_mean,
            skill_diff_std=skill_diff_std,
            player1_mean=propagated.mean[0:1],
            player2_mean=propagated.mean[1:2],
            player1_var=var_1.reshape(1),
            player2_var=var_2.reshape(1),
        )

    def filter_step(factorial_state: Any, match_data: WTATennisResults) -> Any:
        factorial_inds = model.factorializer.get_factorial_indices(match_data)
        factorial_inds = jnp.asarray(factorial_inds)
        local_state = model.factorializer.extract_and_join(factorial_state, match_data)
        prep_state = model.filter_obj.filter_prepare(match_data)
        filtered_joint_state = model.filter_obj.filter_combine(local_state, prep_state)
        local_factorial_filtered_state = model.factorializer.marginalize(
            filtered_joint_state, len(factorial_inds)
        )
        updated_state = model.factorializer.insert(
            local_factorial_filtered_state, factorial_state, factorial_inds
        )
        return updated_state._replace(model_inputs=None)

    def body(factorial_state: Any, match_data: WTATennisResults) -> tuple[Any, Any]:
        prediction = predict_from_state(factorial_state, match_data)
        return filter_step(factorial_state, match_data), prediction

    initial_state = model.initial_state()
    return lax_scan(body, initial_state, matches)


def synchronize_state_to_latest(
    model: GaussianFactorialTennis,
    state: Any,
    data: TennisData,
) -> tuple[Any, int, jnp.ndarray, jnp.ndarray]:
    most_recent_ts = most_recent_timestamp_by_player(
        data.polars_data,
        data.num_players,
    )
    current_time = int(jnp.max(most_recent_ts))
    sync_data = TennisDynamicsOnlyData(
        player_id=jnp.arange(data.num_players),
        timestamp=jnp.broadcast_to(jnp.array(current_time), (data.num_players,)),
        timestamp_previous=most_recent_ts,
    )
    sync_state = model.synchronize(state, sync_data)
    skill_means = sync_state.mean[:, 0]
    skill_vars = jnp.square(sync_state.chol_cov[:, 0, 0])
    return sync_state, current_time, skill_means, skill_vars


def align_test_data_to_training_players(
    train_data: TennisData,
    test_data: TennisData,
) -> dict[str, Any]:
    name_to_id = dict(train_data.name_to_id)
    id_to_name = dict(train_data.id_to_name)
    num_players = train_data.num_players

    def resolve(name: str) -> int:
        nonlocal num_players
        if name not in name_to_id:
            name_to_id[name] = num_players
            id_to_name[num_players] = name
            num_players += 1
        return name_to_id[name]

    test_p1_ids = [resolve(name) for name in test_data.polars_data["Winner"].to_list()]
    test_p2_ids = [resolve(name) for name in test_data.polars_data["Loser"].to_list()]
    test_df = test_data.polars_data
    n_test = test_df.height
    test_jax = WTATennisResults(
        match_index=jnp.arange(n_test),
        player1_id=jnp.array(test_p1_ids),
        player2_id=jnp.array(test_p2_ids),
        winner=jnp.ones(n_test),
        timestamp=jnp.array(test_df.select(pl.col("timestamp")).to_numpy().flatten()),
        player1_timestamp_previous=jnp.array(
            test_df.select(pl.col("player1_timestamp_previous")).to_numpy().flatten()
        ),
        player2_timestamp_previous=jnp.array(
            test_df.select(pl.col("player2_timestamp_previous")).to_numpy().flatten()
        ),
    )
    return {
        "test_jax": test_jax,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "num_players": num_players,
    }


def candidate_params(seed_params: dict[str, Any]) -> list[dict[str, float]]:
    base = {
        "tau": float(seed_params["tau"]),
        "s": float(seed_params["s"]),
        "init_var": float(seed_params["init_var"]),
    }
    scale_sets = [
        (1.0, 1.0, 1.0),
        (0.7, 1.0, 1.0),
        (1.3, 1.0, 1.0),
        (1.0, 0.85, 1.0),
        (1.0, 1.15, 1.0),
        (1.0, 1.0, 0.85),
        (1.0, 1.0, 1.15),
        (0.7, 1.15, 1.0),
        (1.3, 0.85, 1.0),
        (0.7, 1.0, 1.15),
        (1.3, 1.0, 0.85),
    ]
    candidates = [
        {
            "tau": max(base["tau"] * tau_scale, 1e-6),
            "s": max(base["s"] * s_scale, 1e-6),
            "init_var": max(base["init_var"] * init_scale, 1e-6),
        }
        for tau_scale, s_scale, init_scale in scale_sets
    ]

    seen: set[tuple[float, float, float]] = set()
    unique: list[dict[str, float]] = []
    for params in candidates:
        key = tuple(round(params[name], 8) for name in ["tau", "s", "init_var"])
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def optimize_params_for_test_log_score(
    train_data: TennisData,
    test_jax: WTATennisResults,
    seed_params: dict[str, Any],
    total_players: int,
) -> dict[str, Any]:
    trials = []
    best_trial: dict[str, Any] | None = None
    for idx, params in enumerate(candidate_params(seed_params), start=1):
        model, state, current_time, _, _ = fit_and_sync_model(train_data, params)
        state = pad_factorial_state(state, model.init_var, total_players)
        predictions = predict_test_matches(
            model=model,
            state=state,
            current_time=current_time,
            test_jax=test_jax,
        )
        metrics = evaluate_predictions(predictions)
        trial = {
            "trial": idx,
            "params": params,
            "metrics": metrics,
        }
        trials.append(trial)
        print(
            f"  trial={idx:02d} "
            f"avg_log_score={metrics['avg_log_score']:.4f} "
            f"accuracy={metrics['accuracy']:.1%} "
            f"tau={params['tau']:.6f} s={params['s']:.6f} "
            f"init_var={params['init_var']:.6f}"
        )
        if best_trial is None or metrics["avg_log_score"] > best_trial["metrics"]["avg_log_score"]:
            best_trial = trial

    assert best_trial is not None
    return {
        "objective": "maximize_2025_test_avg_log_score",
        "selection_note": "Parameters are selected on the 2025 test set and used for 2026 onward predictions.",
        "candidate_count": len(trials),
        "best_params": best_trial["params"],
        "best_metrics": best_trial["metrics"],
        "trials": trials,
    }


def pad_factorial_state(factorial_state: Any, init_var: Any, num_players: int) -> Any:
    current_players = int(factorial_state.mean.shape[0])
    if num_players <= current_players:
        return factorial_state

    from cuthbertlib.kalman import filtering

    extra = num_players - current_players
    dtype = factorial_state.mean.dtype
    prior_sd = jnp.sqrt(jnp.asarray(init_var, dtype=dtype))
    padded_elem = filtering.FilterScanElement(
        A=jnp.concatenate(
            [factorial_state.elem.A, jnp.zeros((extra, 1, 1), dtype=dtype)],
            axis=0,
        ),
        b=jnp.concatenate(
            [factorial_state.elem.b, jnp.zeros((extra, 1), dtype=dtype)],
            axis=0,
        ),
        U=jnp.concatenate(
            [
                factorial_state.elem.U,
                jnp.broadcast_to(prior_sd.reshape(1, 1, 1), (extra, 1, 1)),
            ],
            axis=0,
        ),
        eta=jnp.concatenate(
            [factorial_state.elem.eta, jnp.zeros((extra, 1), dtype=dtype)],
            axis=0,
        ),
        Z=jnp.concatenate(
            [factorial_state.elem.Z, jnp.zeros((extra, 1, 1), dtype=dtype)],
            axis=0,
        ),
        ell=factorial_state.elem.ell,
    )
    return type(factorial_state)(
        elem=padded_elem,
        model_inputs=None,
        mean_prev=jnp.concatenate(
            [factorial_state.mean_prev, jnp.zeros((extra, 1), dtype=dtype)],
            axis=0,
        ),
    )


def predict_test_matches(
    model: GaussianFactorialTennis,
    state: Any,
    current_time: int,
    test_jax: WTATennisResults,
) -> Any:
    predict_fn = jax.vmap(model.propagate_and_predict, in_axes=(None, None, 0))
    return predict_fn(state, jnp.array(current_time), test_jax)


def evaluate_predictions(predictions: Any) -> dict[str, float | int]:
    p1_probs = predictions.p_player1_win
    correct = jnp.sum(p1_probs > 0.5)
    n_test = int(p1_probs.shape[0])
    accuracy = float(correct) / n_test if n_test else 0.0
    log_scores = jnp.log(jnp.maximum(p1_probs, 1e-8))
    return {
        "n_test_matches": n_test,
        "accuracy": round(accuracy, 4),
        "avg_log_score": round(float(jnp.mean(log_scores)), 4) if n_test else 0.0,
        "uniform_baseline": round(float(jnp.log(0.5)), 4),
    }


def load_known_wta_future_fixtures(name_to_id: dict[str, int]) -> tuple[pl.DataFrame, dict[str, Any]]:
    # Load all available future fixtures (extended window for upcoming tournaments)
    # The model filters all available data and predicts any upcoming fixtures
    fixture_start = dt.date.today()
    fixture_end = fixture_start + dt.timedelta(days=180)  # 6 months ahead
    try:
        loaded_fixtures = load_wta_fixtures(
            start_date=fixture_start.isoformat(),
            end_date=fixture_end.isoformat(),
            origin_date=ORIGIN_DATE,
            match_states=WTA_ACTIVE_MATCH_STATES,
        )
        known_fixtures = filter_known_fixtures(loaded_fixtures, name_to_id)
        skipped = loaded_fixtures.height - known_fixtures.height
        status = {
            "source": "wta_api",
            "start_date": fixture_start.isoformat(),
            "end_date": fixture_end.isoformat(),
            "loaded": loaded_fixtures.height,
            "matched_model_players": known_fixtures.height,
            "skipped_unknown_players": skipped,
            "synthetic_fallback": False,
            "match_states": list(WTA_ACTIVE_MATCH_STATES),
        }
        print(
            f"  Loaded {loaded_fixtures.height} WTA fixtures "
            f"({known_fixtures.height} matched, {skipped} skipped)"
        )
        return known_fixtures, status
    except Exception as exc:
        status = {
            "source": "wta_api",
            "start_date": fixture_start.isoformat(),
            "end_date": fixture_end.isoformat(),
            "loaded": 0,
            "matched_model_players": 0,
            "skipped_unknown_players": 0,
            "synthetic_fallback": False,
            "match_states": list(WTA_ACTIVE_MATCH_STATES),
            "error": str(exc),
        }
        print(f"  Warning: could not load WTA future fixtures: {exc}")
        return filter_known_fixtures(pl.DataFrame(), name_to_id), status


def build_player_rankings_by_id(
    train_data: TennisData,
    skill_means: jnp.ndarray,
    skill_vars: jnp.ndarray,
) -> dict[int, dict[str, Any]]:
    rankings: dict[int, dict[str, Any]] = {}
    for rank, idx in enumerate(jnp.argsort(-skill_means), start=1):
        idx_int = int(idx)
        rankings[idx_int] = {
            "rank": rank,
            "name": train_data.id_to_name.get(idx_int, f"Unknown({idx_int})"),
            "skill": round(float(skill_means[idx]), 4),
            "variance": round(float(skill_vars[idx]), 4),
        }
    return rankings


def player_rank_snapshot(
    player_rankings_by_id: dict[int, dict[str, Any]] | None,
    player_id: int,
) -> dict[str, Any]:
    if not player_rankings_by_id:
        return {}
    return player_rankings_by_id.get(player_id, {})


def load_polymarket_tennis_markets() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "limit": 250,
            "tag_slug": POLYMARKET_TAG_SLUG,
            "active": "true",
            "closed": "false",
        }
    )
    url = f"{POLYMARKET_EVENTS_URL}?{query}"
    status: dict[str, Any] = {
        "source": "polymarket",
        "source_url": POLYMARKET_TENNIS_URL,
        "api_url": url,
        "loaded_events": 0,
        "loaded_moneylines": 0,
        "matched_model_matches": 0,
    }
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            events = json.load(response)
    except Exception as exc:
        status["error"] = str(exc)
        print(f"  Warning: could not load Polymarket tennis markets: {exc}")
        return {}, status

    markets = extract_polymarket_moneylines(events)
    status["loaded_events"] = len(events)
    status["loaded_moneylines"] = len(markets)
    print(f"  Loaded {len(markets)} Polymarket tennis moneylines from {len(events)} events")
    return markets, status


def extract_polymarket_moneylines(events: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    moneylines: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        title = str(event.get("title") or "")
        if " vs " not in title or "(Doubles)" in title or "/" in title:
            continue
        for market in event.get("markets") or []:
            if market.get("sportsMarketType") != "moneyline":
                continue
            outcomes = parse_json_list(market.get("outcomes"))
            prices = [parse_float(price) for price in parse_json_list(market.get("outcomePrices"))]
            if len(outcomes) != 2 or len(prices) != 2 or any(price is None for price in prices):
                continue
            player1, player2 = str(outcomes[0]), str(outcomes[1])
            if "/" in player1 or "/" in player2:
                continue
            key = player_pair_key(player1, player2)
            if not key:
                continue
            moneylines[key] = {
                "source": "polymarket",
                "event_id": str(event.get("id") or ""),
                "event_title": title,
                "event_slug": event.get("slug") or "",
                "event_url": f"https://polymarket.com/event/{event.get('slug')}" if event.get("slug") else POLYMARKET_TENNIS_URL,
                "market_id": str(market.get("id") or ""),
                "market_slug": market.get("slug") or "",
                "market_question": market.get("question") or title,
                "outcome1": player1,
                "outcome2": player2,
                "price1": round(float(prices[0]), 4),
                "price2": round(float(prices[1]), 4),
                "updated_at": market.get("updatedAt") or event.get("updatedAt"),
                "volume": parse_float(market.get("volume") or event.get("volume")),
                "liquidity": parse_float(market.get("liquidity") or event.get("liquidity")),
            }
            break
    return moneylines


def apply_market_predictions(
    matches: list[dict[str, Any]],
    polymarket_moneylines: dict[tuple[str, str], dict[str, Any]],
) -> int:
    matched = 0
    for match in matches:
        key = player_pair_key(match.get("player1"), match.get("player2"))
        if not key or key not in polymarket_moneylines:
            continue
        market = polymarket_moneylines[key]
        player1_key = canonical_player_key(match.get("player1"))
        if player1_key == canonical_player_key(market["outcome1"]):
            player1_market_price = market["price1"]
            player2_market_price = market["price2"]
            market_player1 = market["outcome1"]
            market_player2 = market["outcome2"]
        else:
            player1_market_price = market["price2"]
            player2_market_price = market["price1"]
            market_player1 = market["outcome2"]
            market_player2 = market["outcome1"]
        match["market"] = {
            **market,
            "player1_market_name": market_player1,
            "player2_market_name": market_player2,
            "player1_price": player1_market_price,
            "player2_price": player2_market_price,
            "player1_edge": round(float(match["p_player1_win"]) - float(player1_market_price), 4),
            "player2_edge": round(float(match["p_player2_win"]) - float(player2_market_price), 4),
            "matched_by": "canonical_player_pair",
        }
        matched += 1
    return matched


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def player_pair_key(player1: Any, player2: Any) -> tuple[str, str] | None:
    p1 = canonical_player_key(player1)
    p2 = canonical_player_key(player2)
    if not p1 or not p2:
        return None
    return tuple(sorted((p1, p2)))


def canonical_player_key(name: Any) -> str:
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in ascii_name)
    tokens = [token for token in cleaned.split() if token]
    return " ".join(tokens)


def build_match_predictions_json(
    test_data: TennisData,
    test_jax: WTATennisResults,
    predictions: Any,
    id_to_name: dict[int, str],
    player_rankings_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    matches_json = []
    origin = dt.date.fromisoformat(ORIGIN_DATE)
    for i in range(test_data.num_matches):
        p1_id = int(test_jax.player1_id[i])
        p2_id = int(test_jax.player2_id[i])
        p1_rank = player_rank_snapshot(player_rankings_by_id, p1_id)
        p2_rank = player_rank_snapshot(player_rankings_by_id, p2_id)
        p1_name = id_to_name.get(p1_id, f"Unknown({p1_id})")
        p2_name = id_to_name.get(p2_id, f"Unknown({p2_id})")
        p1_prob = float(predictions.p_player1_win[i])
        p2_prob = float(predictions.p_player2_win[i])
        actual_winner = p1_name
        predicted_winner = p1_name if p1_prob > 0.5 else p2_name
        confidence = max(p1_prob, p2_prob)
        actual_winner_prob = p1_prob
        ts = int(test_jax.timestamp[i])
        matches_json.append(
            {
                "id": f"test-{ts}-{i}-{p1_id}-{p2_id}",
                "date": (origin + dt.timedelta(days=ts)).isoformat(),
                "timestamp": ts,
                "player1": p1_name,
                "player2": p2_name,
                "p_player1_win": round(p1_prob, 4),
                "p_player2_win": round(p2_prob, 4),
                "predicted_winner": predicted_winner,
                "actual_winner": actual_winner,
                "correct": predicted_winner == actual_winner,
                "confidence": round(confidence, 4),
                "log_score": round(float(jnp.log(jnp.maximum(actual_winner_prob, 1e-8))), 4),
                "player1_skill": round(float(predictions.player1_mean[i, 0, 0]), 4),
                "player2_skill": round(float(predictions.player2_mean[i, 0, 0]), 4),
                "player1_skill_sd": round(
                    float(jnp.sqrt(jnp.maximum(predictions.player1_var[i, 0], 1e-8))),
                    4,
                ),
                "player2_skill_sd": round(
                    float(jnp.sqrt(jnp.maximum(predictions.player2_var[i, 0], 1e-8))),
                    4,
                ),
                "player1_rank": p1_rank.get("rank"),
                "player2_rank": p2_rank.get("rank"),
                "player1_latest_skill": p1_rank.get("skill"),
                "player2_latest_skill": p2_rank.get("skill"),
                "player1_latest_variance": p1_rank.get("variance"),
                "player2_latest_variance": p2_rank.get("variance"),
                "tournament": test_data.match_metadata.tournament[i],
                "location": test_data.match_metadata.location[i],
                "tier": test_data.match_metadata.tier[i],
                "surface": test_data.match_metadata.surface[i],
                "round": test_data.match_metadata.round[i],
                "is_future": False,
            }
        )
    return matches_json


def build_top_players_json(
    train_data: TennisData,
    skill_means: jnp.ndarray,
    skill_vars: jnp.ndarray,
) -> list[dict[str, Any]]:
    top_players_json = []
    all_indices = jnp.argsort(-skill_means)
    for rank, idx in enumerate(all_indices, start=1):
        idx_int = int(idx)
        top_players_json.append(
            {
                "rank": rank,
                "name": train_data.id_to_name.get(idx_int, f"Unknown({idx_int})"),
                "skill": round(float(skill_means[idx]), 4),
                "variance": round(float(skill_vars[idx]), 4),
            }
        )
    return top_players_json


def build_data_window_json(data: TennisData) -> list[dict[str, str]]:
    return [{"date": row["Date"].isoformat()} for row in data.polars_data.iter_rows(named=True)]


def evaluated_match_window(matches: list[dict[str, Any]], prefix: str) -> dict[str, str]:
    dates = sorted(str(match["date"]) for match in matches if match.get("date"))
    if not dates:
        return {}
    # For upcoming matches, only return start date (no end date constraint)
    # The model filters all available data and predicts any upcoming fixtures
    if prefix == "upcoming_match":
        return {f"{prefix}_start": dates[0]}
    return {
        f"{prefix}_start": dates[0],
        f"{prefix}_end": dates[-1],
    }


def build_predictions_payload(
    trained_seed_params: dict[str, Any],
    optimization: dict[str, Any],
    final_metrics: dict[str, Any],
    top_players: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    test_window_matches: list[dict[str, Any]],
    future_matches: list[dict[str, Any]],
    fixture_status: dict[str, Any],
    market_status: dict[str, Any] | None = None,
    model_state_as_of: float | None = None,
    result_update_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        **final_metrics,
        "n_future_matches": len(future_matches),
    }
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_state_as_of": model_state_as_of,
        "model_state_as_of_date": (
            dt.date.fromisoformat(ORIGIN_DATE) + dt.timedelta(days=int(model_state_as_of))
        ).isoformat() if model_state_as_of is not None else None,
        "data_windows": {
            "origin_date": ORIGIN_DATE,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "train_display_start": TRAIN_DISPLAY_START,
            "train_display_end": TRAIN_DISPLAY_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "test_display_start": TEST_DISPLAY_START,
            "test_display_end": TEST_DISPLAY_END,
            "prediction_start": EVAL_START,
            "prediction_end": EVAL_END,
            "prediction_display_start": EVAL_DISPLAY_START,
            "prediction_display_end": EVAL_DISPLAY_END,
            "test_match_start": TEST_DISPLAY_START,
            "test_match_end": TEST_DISPLAY_END,
            **evaluated_match_window(future_matches, "upcoming_match"),
        },
        "model_params": optimization["best_params"],
        "seed_model_params": {
            "tau": float(trained_seed_params["tau"]),
            "s": float(trained_seed_params["s"]),
            "init_var": float(trained_seed_params["init_var"]),
        },
        "optimization": optimization,
        "fixture_status": fixture_status,
        "result_update_status": result_update_status or {},
        "market_status": market_status or {"source": "polymarket", "loaded": 0, "matched_model_matches": 0},
        "metrics": metrics,
        "top_players": top_players,
        "matches": matches,
        "future_matches": future_matches,
    }


def validate_predictions_payload(payload: dict[str, Any]) -> None:
    required = {
        "generated_at",
        "data_windows",
        "model_params",
        "optimization",
        "metrics",
        "top_players",
        "matches",
        "future_matches",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Prediction payload missing required keys: {sorted(missing)}")
    if not isinstance(payload["matches"], list):
        raise TypeError("Prediction payload matches must be a list")
    if not isinstance(payload["future_matches"], list):
        raise TypeError("Prediction payload future_matches must be a list")
    future_ids = [match.get("id") for match in payload["future_matches"]]
    if any(not match_id for match_id in future_ids):
        raise ValueError("Prediction payload future matches must have IDs")
    duplicate_future_ids = sorted(
        match_id for match_id in set(future_ids) if future_ids.count(match_id) > 1
    )
    if duplicate_future_ids:
        raise ValueError(
            f"Prediction payload future match IDs must be unique: {duplicate_future_ids}"
        )


def build_results_json(
    test_data: TennisData,
    source: str = "tennis-data.co.uk WTA historical results",
) -> dict[str, Any]:
    results = []
    for i, row in enumerate(test_data.polars_data.iter_rows(named=True)):
        results.append(
            {
                "id": f"result-{row['Date'].isoformat()}-{i}",
                "date": row["Date"].isoformat(),
                "winner": row["Winner"],
                "loser": row["Loser"],
                "player1": row["Winner"],
                "player2": row["Loser"],
                "actual_winner": row["Winner"],
                "tournament": test_data.match_metadata.tournament[i],
                "location": test_data.match_metadata.location[i],
                "tier": test_data.match_metadata.tier[i],
                "surface": test_data.match_metadata.surface[i],
                "round": test_data.match_metadata.round[i],
            }
        )
    result_dates = [result["date"] for result in results]
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": source,
        "data_window": {
            "start": min(result_dates) if result_dates else TEST_DISPLAY_START,
            "end": max(result_dates) if result_dates else TEST_DISPLAY_END,
        },
        "results": results,
    }


def print_top_players(
    train_data: TennisData,
    skill_means: jnp.ndarray,
    skill_vars: jnp.ndarray,
) -> None:
    print("  Top 10 players by filtered skill:")
    for idx in jnp.argsort(-skill_means)[:10]:
        idx_int = int(idx)
        name = train_data.id_to_name[idx_int]
        print(
            f"    {name:30s} "
            f"skill={float(skill_means[idx]):+.3f} "
            f"var={float(skill_vars[idx]):.3f}"
        )


def generate_future_predictions(
    loaded_model: GaussianFactorialTennis,
    loaded_state: Any,
    current_time: int,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    num_players: int,
    n_matches: int = 50,
    fixtures: pl.DataFrame | None = None,
    use_synthetic_fallback: bool = False,
    player_rankings_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate future predictions from real scheduled fixtures.

    Synthetic top-player matchups are opt-in only and remain disabled in the
    main workflow.
    """
    del name_to_id, num_players
    if fixtures is not None and fixtures.height > 0:
        return generate_fixture_predictions(
            loaded_model=loaded_model,
            loaded_state=loaded_state,
            current_time=current_time,
            fixtures=fixtures,
            id_to_name=id_to_name,
            player_rankings_by_id=player_rankings_by_id,
        )
    if not use_synthetic_fallback:
        return []

    top_ids = sorted(
        [(i, float(loaded_state.mean[i, 0])) for i in range(loaded_model.num_players)],
        key=lambda item: -item[1],
    )[:20]
    synthetic_rows = []
    origin = dt.date.fromisoformat(ORIGIN_DATE)
    future_date = origin + dt.timedelta(days=int(current_time) + 1)
    for i in range(len(top_ids)):
        for j in range(i + 1, min(i + 4, len(top_ids))):
            p1_id, p2_id = int(top_ids[i][0]), int(top_ids[j][0])
            synthetic_rows.append(
                {
                    "date": future_date.isoformat(),
                    "timestamp": int(current_time + 1),
                    "player1_id": p1_id,
                    "player2_id": p2_id,
                    "player1_full_name": id_to_name.get(p1_id, f"Unknown({p1_id})"),
                    "player2_full_name": id_to_name.get(p2_id, f"Unknown({p2_id})"),
                    "tournament": "Synthetic matchup",
                    "location": "TBD",
                    "tier": "Synthetic",
                    "surface": "TBD",
                    "round": "Unknown",
                    "source": "synthetic",
                    "source_match_id": f"synthetic-{p1_id}-{p2_id}",
                    "match_state": "",
                }
            )
            if len(synthetic_rows) >= n_matches:
                break
        if len(synthetic_rows) >= n_matches:
            break
    return generate_fixture_predictions(
        loaded_model,
        loaded_state,
        current_time,
        pl.DataFrame(synthetic_rows),
        id_to_name,
        player_rankings_by_id=player_rankings_by_id,
    )


def generate_fixture_predictions(
    loaded_model: GaussianFactorialTennis,
    loaded_state: Any,
    current_time: int,
    fixtures: pl.DataFrame,
    id_to_name: dict[int, str],
    player_rankings_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    future_matches_json = []
    for idx, fixture in enumerate(fixtures.iter_rows(named=True)):
        p1_id = int(fixture["player1_id"])
        p2_id = int(fixture["player2_id"])
        p1_rank = player_rank_snapshot(player_rankings_by_id, p1_id)
        p2_rank = player_rank_snapshot(player_rankings_by_id, p2_id)
        p1_name = fixture.get("player1_full_name") or id_to_name.get(p1_id, f"Unknown({p1_id})")
        p2_name = fixture.get("player2_full_name") or id_to_name.get(p2_id, f"Unknown({p2_id})")
        fixture_timestamp = int(fixture["timestamp"])
        match_data = WTATennisResults(
            match_index=jnp.array(idx),
            player1_id=jnp.array(p1_id),
            player2_id=jnp.array(p2_id),
            winner=jnp.array(1.0),
            timestamp=jnp.array(fixture_timestamp),
            player1_timestamp_previous=jnp.array(current_time),
            player2_timestamp_previous=jnp.array(current_time),
        )
        pred = loaded_model.propagate_and_predict(loaded_state, float(current_time), match_data)
        p1_prob = float(pred.p_player1_win)
        p2_prob = float(pred.p_player2_win)
        confidence = max(p1_prob, p2_prob)
        match_state = fixture.get("match_state") or ""
        match_status = fixture_match_status(match_state)
        future_matches_json.append(
            {
                "id": fixture_prediction_id(fixture, idx, p1_id, p2_id),
                "date": fixture["date"],
                "timestamp": fixture_timestamp,
                "player1": p1_name,
                "player2": p2_name,
                "p_player1_win": round(p1_prob, 4),
                "p_player2_win": round(p2_prob, 4),
                "predicted_winner": p1_name if p1_prob > 0.5 else p2_name,
                "actual_winner": None,
                "correct": None,
                "confidence": round(confidence, 4),
                "log_score": None,
                "player1_skill": round(float(pred.player1_mean[0, 0]), 4),
                "player2_skill": round(float(pred.player2_mean[0, 0]), 4),
                "player1_skill_sd": round(float(jnp.sqrt(jnp.maximum(pred.player1_var[0], 1e-8))), 4),
                "player2_skill_sd": round(float(jnp.sqrt(jnp.maximum(pred.player2_var[0], 1e-8))), 4),
                "player1_rank": p1_rank.get("rank"),
                "player2_rank": p2_rank.get("rank"),
                "player1_latest_skill": p1_rank.get("skill"),
                "player2_latest_skill": p2_rank.get("skill"),
                "player1_latest_variance": p1_rank.get("variance"),
                "player2_latest_variance": p2_rank.get("variance"),
                "tournament": fixture.get("tournament") or "Unknown",
                "location": fixture.get("location") or "Unknown",
                "tier": fixture.get("tier") or "Unknown",
                "surface": fixture.get("surface") or "Unknown",
                "round": fixture.get("round") or "Unknown",
                "source": fixture.get("source") or "wta_api",
                "source_match_id": fixture.get("source_match_id") or "",
                "source_tournament_id": fixture.get("source_tournament_id") or "",
                "match_state": match_state,
                "match_status": match_status,
                "is_future": True,
            }
        )
    return future_matches_json


def fixture_prediction_id(
    fixture: dict[str, Any],
    index: int,
    player1_id: int,
    player2_id: int,
) -> str:
    """Build a stable fixture ID even when a source reuses draw-local match IDs."""

    def component(value: Any, fallback: str) -> str:
        normalized = str(value or fallback).strip()
        return normalized.replace(":", "_")

    source = component(fixture.get("source"), "wta_api")
    date = component(fixture.get("date"), str(fixture.get("timestamp") or "unknown-date"))
    tournament = component(
        fixture.get("source_tournament_id"),
        str(fixture.get("tournament") or "unknown-tournament"),
    )
    source_match = component(fixture.get("source_match_id"), str(index))
    return f"future:{source}:{date}:{tournament}:{source_match}:{player1_id}:{player2_id}"


def fixture_match_status(match_state: str) -> str:
    normalized = str(match_state or "").upper()
    if normalized in {"P", "L", "I"}:
        return "in_progress"
    if normalized == "S":
        return "suspended"
    return "upcoming"


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
