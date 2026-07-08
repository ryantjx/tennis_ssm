"""
End-to-end workflow for the Gaussian Factorial Tennis model.

Following the cuthberto-carlos pattern:
1. Load WTA training data (2023-2025)
2. Run factorial moments filter on training data
3. Synchronize all players to the most recent timestamp
4. Save the factorial state and parameters to disk
5. Load the saved state
6. Load test data (2026) and propagate-and-predict each match
7. Evaluate prediction accuracy
"""

import os

import jax
import jax.tree_util as jtu
from jax import numpy as jnp
import polars as pl

from src.data.data import load_wta, most_recent_timestamp_by_player
from src.data.data_types import WTATennisResults, TennisDynamicsOnlyData
from src.model.gaussianfactorial_tennis import GaussianFactorialTennis

STATE_FILE = "outputs/tennis_factorial_state.json"


def main():
    # ------------------------------------------------------------------
    # 1. Load WTA training data (2023-2025)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Loading WTA training data (2023-2025)...")
    print("=" * 70)

    train_data = load_wta(
        start_date="2022-12-31",
        end_date="2026-01-01",
        origin_date="2022-12-31",
    )

    print(f"  Training matches: {train_data.num_matches}")
    print(f"  Training players: {train_data.num_players}")
    print()

    # ------------------------------------------------------------------
    # 2. Train model parameters (gradient optimization)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("2. Training model parameters (Adam, 100 steps)...")
    print("=" * 70)

    from src.model.gaussianfactorial_tennis import train

    trained_params, train_history = train(
        matches=train_data.jax_data,
        num_players=train_data.num_players,
        steps=100,
        learning_rate=0.1,
        log_every=10,
        initial_tau=0.1,
        initial_s=1.0,
        initial_init_var=1.0,
    )

    print(f"  Final parameters: tau={float(trained_params['tau']):.4f}, "
          f"s={float(trained_params['s']):.4f}, "
          f"init_var={float(trained_params['init_var']):.4f}")
    print()

    # ------------------------------------------------------------------
    # 3. Build model with trained parameters and filter
    # ------------------------------------------------------------------
    print("=" * 70)
    print("3. Running factorial moments filter with trained parameters...")
    print("=" * 70)

    model = GaussianFactorialTennis(
        init_mean=0.0,
        init_var=trained_params["init_var"],
        tau=trained_params["tau"],
        s=trained_params["s"],
        num_players=train_data.num_players,
    )
    print(f"  Parameters: tau={float(trained_params['tau']):.4f}, "
          f"s={float(trained_params['s']):.4f}, "
          f"init_var={float(trained_params['init_var']):.4f}")

    final_state = model.filter(train_data.jax_data)

    skill_means = final_state.mean[:, 0]
    skill_vars = jnp.square(final_state.chol_cov[:, 0, 0])

    print(f"  Filtering complete. Processed {train_data.num_matches} matches.")
    print()

    # Show top-rated players
    print("-" * 70)
    print("  Top 10 players by filtered skill:")
    print("-" * 70)
    top_indices = jnp.argsort(-skill_means)[:10]
    for idx in top_indices:
        name = train_data.id_to_name[int(idx)]
        mean = float(skill_means[idx])
        var = float(skill_vars[idx])
        print(f"    {name:30s}  skill={mean:+.3f}  var={var:.3f}")
    print()

    # ------------------------------------------------------------------
    # 4. Synchronize all players to the most recent timestamp
    # ------------------------------------------------------------------
    print("=" * 70)
    print("4. Synchronizing players to most recent timestamp...")
    print("=" * 70)

    most_recent_ts = most_recent_timestamp_by_player(
        train_data.polars_data, train_data.num_players
    )
    current_time = int(jnp.max(most_recent_ts))
    print(f"  Most recent timestamp: {current_time} days since origin")

    sync_data = TennisDynamicsOnlyData(
        player_id=jnp.arange(train_data.num_players),
        timestamp=jnp.broadcast_to(
            jnp.array(current_time), (train_data.num_players,)
        ),
        timestamp_previous=most_recent_ts,
    )

    sync_state = model.synchronize(final_state, sync_data)
    print(f"  Synchronized {train_data.num_players} players.")
    print()

    # ------------------------------------------------------------------
    # 5. Save the factorial state and parameters
    # ------------------------------------------------------------------
    print("=" * 70)
    print("5. Saving factorial state and parameters...")
    print("=" * 70)

    model.save_state(sync_state, STATE_FILE)
    print(f"  Saved to {STATE_FILE}")
    print()

    # ------------------------------------------------------------------
    # 6. Load the saved state
    # ------------------------------------------------------------------
    print("=" * 70)
    print("6. Loading saved state...")
    print("=" * 70)

    loaded_model, loaded_state = GaussianFactorialTennis.load_state(STATE_FILE)
    print(f"  Loaded model with {loaded_model.num_players} players")
    print(f"  State mean shape: {loaded_state.mean.shape}")
    print()

    # ------------------------------------------------------------------
    # 7. Load test data (2026) and propagate-and-predict
    # ------------------------------------------------------------------
    print("=" * 70)
    print("7. Loading 2026 test data and predicting...")
    print("=" * 70)

    test_data = load_wta(
        start_date="2025-12-31",
        end_date="2027-01-01",
        origin_date="2022-12-31",
    )
    print(f"  Test matches: {test_data.num_matches}")

    # Align test player IDs to the loaded model's player mapping
    # New players in 2026 that weren't in training get their skill from the prior
    name_to_id = dict(train_data.name_to_id)
    id_to_name = dict(train_data.id_to_name)
    num_players = train_data.num_players

    test_p1_ids = []
    test_p2_ids = []
    for name in test_data.polars_data["Winner"].to_list():
        if name not in name_to_id:
            name_to_id[name] = num_players
            id_to_name[num_players] = name
            num_players += 1
        test_p1_ids.append(name_to_id[name])
    for name in test_data.polars_data["Loser"].to_list():
        if name not in name_to_id:
            name_to_id[name] = num_players
            id_to_name[num_players] = name
            num_players += 1
        test_p2_ids.append(name_to_id[name])

    print(f"  Total players (train + new in test): {num_players}")
    print()

    # Build test JAX data with aligned IDs
    test_df = test_data.polars_data
    n_test = test_df.height

    test_jax = WTATennisResults(
        match_index=jnp.arange(n_test),
        player1_id=jnp.array(test_p1_ids),
        player2_id=jnp.array(test_p2_ids),
        winner=jnp.ones(n_test),
        timestamp=jnp.array(
            test_df.select(pl.col("timestamp")).to_numpy().flatten()
        ),
        player1_timestamp_previous=jnp.array(
            test_df.select(pl.col("player1_timestamp_previous"))
            .to_numpy()
            .flatten()
        ),
        player2_timestamp_previous=jnp.array(
            test_df.select(pl.col("player2_timestamp_previous"))
            .to_numpy()
            .flatten()
        ),
    )

    # ------------------------------------------------------------------
    # 8. Evaluate: propagate-and-predict each test match
    # ------------------------------------------------------------------
    print("=" * 70)
    print("8. Evaluating predictions on 2026 matches...")
    print("=" * 70)

    # Vectorized propagate_and_predict over all test matches
    # For players not in the training set (new in 2026), use prior skill (0, var=1)
    # We handle this by clamping player IDs to the loaded state's size
    max_known = loaded_model.num_players
    # Clamp IDs — unknown players will get the prior (skill=0) from the state
    # if their index is beyond the state, we need to handle it
    # For simplicity, pad the loaded state with prior values for new players
    if num_players > max_known:
        # Pad the state with prior values for new players
        padded_mean = jnp.concatenate([
            loaded_state.mean,
            jnp.zeros((num_players - max_known, 1)),
        ])
        padded_chol = jnp.concatenate([
            loaded_state.chol_cov,
            jnp.ones((num_players - max_known, 1, 1)),
        ])
        padded_eta = jnp.concatenate([
            loaded_state.elem.eta,
            jnp.zeros((num_players - max_known, 1)),
        ])
        padded_Z = jnp.concatenate([
            loaded_state.elem.Z,
            jnp.zeros((num_players - max_known, 1, 1)),
        ])
        padded_A = jnp.concatenate([
            loaded_state.elem.A,
            jnp.zeros((num_players - max_known, 1, 1)),
        ])
        from cuthbertlib.kalman import filtering
        padded_elem = filtering.FilterScanElement(
            A=padded_A,
            b=padded_mean,
            U=padded_chol,
            eta=padded_eta,
            Z=padded_Z,
            ell=loaded_state.elem.ell,
        )
        loaded_state = type(loaded_state)(
            elem=padded_elem,
            model_inputs=None,
            mean_prev=jnp.concatenate([
                loaded_state.mean_prev,
                jnp.zeros((num_players - max_known, 1)),
            ]),
        )

    # Vectorized prediction over all test matches
    predict_fn = jax.vmap(
        loaded_model.propagate_and_predict,
        in_axes=(None, None, 0),
    )

    print(f"  Propagating and predicting {n_test} matches...")
    predictions = predict_fn(loaded_state, jnp.array(current_time), test_jax)

    # Evaluate: in raw data, player1 = Winner, so winner = 1.0 always
    # Model "correctly" predicts if P(p1 win) > 0.5
    p1_probs = predictions.p_player1_win
    correct = jnp.sum(p1_probs > 0.5)
    accuracy = float(correct) / n_test

    # Log score
    log_scores = jnp.log(jnp.maximum(p1_probs, 1e-8))
    avg_log_score = float(jnp.mean(log_scores))

    print(f"  Test matches evaluated: {n_test}")
    print(f"  Accuracy:               {accuracy:.1%}")
    print(f"  Avg log-score:          {avg_log_score:.4f}")
    print(f"  Uniform baseline:       {float(jnp.log(0.5)):.4f}")
    print()

    # ------------------------------------------------------------------
    # 9. Export predictions as JSON for the frontend
    # ------------------------------------------------------------------
    print("=" * 70)
    print("9. Exporting predictions for frontend...")
    print("=" * 70)

    import json
    import datetime

    # Build the predictions JSON from test data
    # In the raw data, player1 = Winner, player2 = Loser
    # We want to show both the model prediction and the actual result
    matches_json = []
    for i in range(n_test):
        p1_id = int(test_jax.player1_id[i])
        p2_id = int(test_jax.player2_id[i])
        p1_name = id_to_name.get(p1_id, f"Unknown({p1_id})")
        p2_name = id_to_name.get(p2_id, f"Unknown({p2_id})")
        p1_prob = float(predictions.p_player1_win[i])
        p2_prob = float(predictions.p_player2_win[i])

        # Actual result: player1 = Winner in raw data
        actual_winner = p1_name

        # Date from timestamp
        ts = int(test_jax.timestamp[i])
        match_date = (datetime.date(2022, 12, 31) + datetime.timedelta(days=ts)).isoformat()

        matches_json.append({
            "date": match_date,
            "player1": p1_name,
            "player2": p2_name,
            "p_player1_win": round(p1_prob, 4),
            "p_player2_win": round(p2_prob, 4),
            "actual_winner": actual_winner,
            "correct": p1_prob > 0.5,
        })

    # Build top players list
    top_players_json = []
    all_indices = jnp.argsort(-skill_means)
    for idx in all_indices:
        name = train_data.id_to_name.get(int(idx), f"Unknown({int(idx)})")
        mean = float(skill_means[idx])
        var = float(skill_vars[idx])
        top_players_json.append({
            "name": name,
            "skill": round(mean, 4),
            "variance": round(var, 4),
        })

    output_json = {
        "generated_at": datetime.datetime.now().isoformat(),
        "model_params": {
            "tau": float(trained_params["tau"]),
            "s": float(trained_params["s"]),
            "init_var": float(trained_params["init_var"]),
        },
        "metrics": {
            "n_test_matches": n_test,
            "accuracy": round(accuracy, 4),
            "avg_log_score": round(avg_log_score, 4),
            "uniform_baseline": round(float(jnp.log(0.5)), 4),
        },
        "top_players": top_players_json[:30],
        "matches": matches_json,
    }

    # Save to docs/ for GitHub Pages
    os.makedirs("docs", exist_ok=True)
    with open("docs/predictions.json", "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved {len(matches_json)} predictions to docs/predictions.json")
    print()

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()