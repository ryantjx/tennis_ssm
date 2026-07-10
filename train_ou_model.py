"""Train OU model parameters from scratch."""
import sys
sys.path.insert(0, ".")

from src.data.data import load_wta
from src.data.data_types import WTATennisResults
from src.model.gaussianfactorial_tennis import train
import jax.numpy as jnp

# Load training data (2022-2024)
print("Loading training data...")
data = load_wta(
    start_date="2021-12-31",
    end_date="2024-12-31",
    origin_date="2021-12-31",
)

print(f"Training matches: {data.num_matches}")
print(f"Number of players: {data.num_players}")

# Train with OU dynamics
print("\n" + "="*70)
print("Training OU model parameters")
print("="*70)

best_params, history = train(
    matches=data.jax_data,
    num_players=data.num_players,
    steps=300,
    learning_rate=0.05,
    log_every=20,
    init_mean=0.0,
    initial_tau=0.01,  # OU mean-reversion rate (smaller = slower reversion)
    initial_s=1.5,
    initial_init_var=1.0,
)

print("\n" + "="*70)
print("Final trained parameters:")
print(f"  tau (mean-reversion rate): {float(best_params['tau']):.6f}")
print(f"  s (observation scale): {float(best_params['s']):.6f}")
print(f"  init_var: {float(best_params['init_var']):.6f}")
print("="*70)

# Save to state file
import json
from pathlib import Path

state_file = Path("outputs/tennis_factorial_state.json")
state_file.parent.mkdir(parents=True, exist_ok=True)

with open(state_file, "w") as f:
    json.dump({
        "params": {
            "tau": float(best_params["tau"]),
            "s": float(best_params["s"]),
            "init_var": float(best_params["init_var"]),
            "init_mean": 0.0,
            "num_players": data.num_players,
        },
        "training_history": history[-10:],  # Last 10 steps
    }, f, indent=2)

print(f"\nSaved trained parameters to {state_file}")
