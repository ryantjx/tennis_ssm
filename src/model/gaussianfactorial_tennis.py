"""
Gaussian Factorial model for Tennis — 1D scalar skill with sigmoid observation.

This module implements the model described in the README:

    p(x_0)         ~ N(mu_0, Sigma_0)
    p(x_t | x_{t-1}) ~ N(x_{t-1}, tau_d^2 * Delta_t)       (Wiener process)
    P(y_k = win_i)  = sigmoid((x_i - x_j) / s_d)            (logistic observation)

Each player has a single latent skill that evolves via a Wiener process (random
walk with variance proportional to elapsed time). Match outcomes are observed
through a sigmoid (logistic) link on the skill difference.

Inference uses ``cuthbert``'s moments filter with a factorial state: each player
is an independent factor. The moments filter approximates the non-linear
observation as conditionally Gaussian:

    p(y | x) ~ N(y | h(x), R(x))

where h(x) = sigmoid((x_i - x_j) / s) and R(x) = h(x) * (1 - h(x)).

All building, filtering, smoothing, and predicting is handled through a single
class that wraps the ``cuthbert`` library via composition (no inheritance).

Reference: https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/gaussian/
"""

from functools import partial
from typing import NamedTuple

import jax
from jax import Array
from jax import numpy as jnp, vmap, tree_util as jtu
from jax.typing import ArrayLike

from cuthbert.gaussian import moments
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbert.factorial.gaussian import build_factorializer
from cuthbertlib.linearize.moments import MeanAndCholCovFunc

from src.data.data_types import WTATennisResults, TennisDynamicsOnlyData


# ---------------------------------------------------------------------------
# Prediction result container
# ---------------------------------------------------------------------------


class TennisMatchPrediction(NamedTuple):
    """Container for a single match prediction.

    The sigmoid model only predicts win/loss probabilities (not scorelines).

    Attributes:
        p_player1_win: Probability that player 1 wins the match.
        p_player2_win: Probability that player 2 wins the match.
        skill_diff_mean: Mean of the skill difference (player1 - player2).
        skill_diff_std: Standard deviation of the skill difference.
        player1_mean: Mean skill of player 1 at match time.
        player2_mean: Mean skill of player 2 at match time.
        player1_var: Variance of player 1's skill at match time.
        player2_var: Variance of player 2's skill at match time.
    """

    p_player1_win: Array
    p_player2_win: Array
    skill_diff_mean: Array
    skill_diff_std: Array
    player1_mean: Array
    player2_mean: Array
    player1_var: Array
    player2_var: Array


# ---------------------------------------------------------------------------
# Factorial index helper
# ---------------------------------------------------------------------------


def get_factorial_inds(model_inputs: WTATennisResults) -> Array:
    """Get indices of the two player-factors involved in a match.

    Args:
        model_inputs: The match data.

    Returns:
        Integer array of shape (2,) with the player factor indices.
    """
    return jnp.array([model_inputs.player1_id, model_inputs.player2_id])


# ---------------------------------------------------------------------------
# Init / dynamics / observation parameter functions
# ---------------------------------------------------------------------------


def get_init_params(
    model_inputs: WTATennisResults | TennisDynamicsOnlyData | None,
    init_mean: ArrayLike,
    init_var: ArrayLike,
    num_players: int | None = None,
) -> tuple[Array, Array]:
    """Return the initial mean and Cholesky covariance for each player factor.

    Each player has a 1D scalar skill, so the state dimension is 1.

    The returned shapes must be:
        - mean: (F, 1) where F = num_players (or 1 if None)
        - chol_cov: (F, 1, 1)

    These shapes are required by cuthbert's factorializer, which treats
    1D arrays as scalars and 2D+ arrays as having a leading factorial dimension.

    Args:
        model_inputs: The model inputs (unused except for type dispatch).
        init_mean: Scalar long-run mean of the skill.
        init_var: Scalar initial variance of the skill.
        num_players: Number of players in the factorial state.

    Returns:
        Tuple ``(init_mean, init_chol_cov)`` with shapes ``(F, 1)`` and
        ``(F, 1, 1)`` where F = num_players (or 1 if None).
    """
    init_mean = jnp.asarray(init_mean)
    init_chol = jnp.sqrt(jnp.asarray(init_var))
    if num_players is None:
        # Return (1, 1) and (1, 1, 1) — factorialize_init_state will broadcast
        return init_mean.reshape(1, 1), init_chol.reshape(1, 1, 1)
    return (
        jnp.broadcast_to(init_mean, (num_players, 1)),
        jnp.broadcast_to(init_chol, (num_players, 1, 1)),
    )


def _process_timestamp(t: ArrayLike, num_joined_factors: int) -> Array:
    """Broadcast a timestamp to each joined factor (1D state, no repeat needed).

    Handles the case where ``t`` is already an array of the right length
    (e.g. per-player timestamps) vs a scalar that needs broadcasting.
    Also handles the case where ``t`` has more elements than needed
    (e.g. when two players map to the same factor in the dummy input).
    """
    t = jnp.asarray(t)
    if t.ndim == 0:
        return jnp.broadcast_to(t, (num_joined_factors,))
    # Already an array — take/slice to the right length
    return t[:num_joined_factors]


def get_dynamics_params(
    state: LinearizedKalmanFilterState,
    model_inputs: WTATennisResults | TennisDynamicsOnlyData,
    tau: ArrayLike,
) -> tuple[MeanAndCholCovFunc, Array]:
    """Wiener-process dynamics for 1D player skills.

    The state evolves as a random walk: x_t = x_{t-1} + N(0, tau^2 * dt),
    where dt is the elapsed time (days) since the player's previous match.

    Args:
        state: Current filter state.
        model_inputs: Match or dynamics-only data.
        tau: Dynamics rate parameter (standard deviation per unit time).

    Returns:
        Tuple of (dynamics mean-and-chol-cov function, linearization point).
    """
    num_joined_factors = state.mean.shape[0]  # 1D state, no division needed
    tau = jnp.asarray(tau)

    if isinstance(model_inputs, WTATennisResults):
        assert (
            model_inputs.player1_timestamp_previous is not None
            and model_inputs.player2_timestamp_previous is not None
        ), "player1/2_timestamp_previous are required for dynamics"
        timestamp_previous = jnp.array(
            [
                model_inputs.player1_timestamp_previous,
                model_inputs.player2_timestamp_previous,
            ]
        )
    else:
        timestamp_previous = model_inputs.timestamp_previous

    timestamp = _process_timestamp(model_inputs.timestamp, num_joined_factors)
    timestamp_previous = _process_timestamp(timestamp_previous, num_joined_factors)

    elapsed_days = jnp.maximum(timestamp - timestamp_previous, 0)
    # Wiener process: variance = tau^2 * dt per factor
    # The joint chol_cov is a diagonal matrix of shape (d, d) where d = num_joined_factors
    stds = tau * jnp.sqrt(elapsed_days)  # shape (num_joined_factors,)
    chol_cov = jnp.diag(stds)  # shape (num_joined_factors, num_joined_factors)

    def dynamics_mean_and_chol_cov(x_prev: ArrayLike) -> tuple[Array, Array]:
        x_prev = jnp.asarray(x_prev)
        # Random walk: mean = x_prev (no drift)
        return x_prev, chol_cov

    linearization_point = jnp.zeros(num_joined_factors)

    return dynamics_mean_and_chol_cov, linearization_point


def _sigmoid(x: ArrayLike) -> Array:
    """Numerically stable sigmoid (logistic) function."""
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x)),
    )


def get_observation_params(
    state: LinearizedKalmanFilterState,
    model_inputs: WTATennisResults,
    s: ArrayLike,
) -> tuple[MeanAndCholCovFunc, Array, Array]:
    """Sigmoid observation model for win/loss.

    Approximates the Bernoulli observation as conditionally Gaussian:

        h(x) = sigmoid((x_i - x_j) / s)     (conditional mean = P(p1 wins))
        R(x) = h(x) * (1 - h(x))            (Bernoulli variance)

    The observed value y is 1.0 (player 1 won) or 0.0 (player 2 won).

    Args:
        state: Current filter state (linearized around ``state.mean``).
        model_inputs: The match data.
        s: Scale parameter controlling how much skill difference influences
            win probability. Larger s = more random outcomes.

    Returns:
        Tuple of (observation mean-and-chol-cov function, linearization point,
        observed outcome).
    """
    s = jnp.asarray(s)

    def observation_mean_and_chol_cov(x: ArrayLike) -> tuple[Array, Array]:
        x = jnp.asarray(x)
        # Handle both 1-factor (dummy) and 2-factor (real match) cases
        x_i = x[0]
        x_j = x[-1] if x.shape[0] > 1 else x[0]
        p1_win = _sigmoid((x_i - x_j) / s)
        mean = p1_win.reshape(1)
        var = p1_win * (1.0 - p1_win)
        # Ensure positive variance (avoid degeneracy at 0 or 1)
        var = jnp.maximum(var, 1e-8)
        chol = jnp.sqrt(var).reshape(1, 1)
        return mean, chol

    y = jnp.array([model_inputs.winner], dtype=jnp.float32)
    return observation_mean_and_chol_cov, state.mean, y


def get_observation_params_noop(
    state: LinearizedKalmanFilterState,
    model_inputs: WTATennisResults | TennisDynamicsOnlyData,
) -> tuple[MeanAndCholCovFunc, Array, Array]:
    """Zero-information observation for dynamics-only filtering steps."""
    del model_inputs

    def observation_mean_and_chol_cov(x: ArrayLike) -> tuple[Array, Array]:
        dtype = jnp.asarray(x).dtype
        return jnp.zeros(1, dtype=dtype), jnp.ones((1, 1), dtype=dtype)

    return observation_mean_and_chol_cov, state.mean, jnp.array([jnp.nan])


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class GaussianFactorialTennis:
    """Gaussian factorial state-space model for tennis.

    Wraps ``cuthbert``'s moments filter, factorializer, and smoother into a
    single class that handles building, filtering, smoothing, and predicting.

    Each player is a *factor* with a 1D latent skill evolving via a Wiener
    process (random walk with variance ``tau^2 * dt``). Match outcomes are
    observed through a sigmoid (logistic) link on the skill difference.

    Args:
        init_mean: Scalar initial mean of player skills.
        init_var: Scalar initial variance of player skills.
        tau: Dynamics rate (standard deviation per day). Controls how quickly
            skills drift over time.
        s: Scale parameter for the sigmoid observation. Controls how much
            skill difference influences win probability.
        num_players: Total number of players (factors) in the model.

    Example:
        >>> model = GaussianFactorialTennis(
        ...     init_mean=0.0, init_var=1.0, tau=0.1, s=1.0, num_players=500,
        ... )
        >>> filter_states = model.filter(matches_batch)
        >>> pred = model.predict_match(player1_idx, player2_idx, filter_states)
    """

    def __init__(
        self,
        init_mean: ArrayLike,
        init_var: ArrayLike,
        tau: ArrayLike,
        s: ArrayLike,
        num_players: int,
    ):
        self.init_mean = jnp.asarray(init_mean)
        self.init_var = jnp.asarray(init_var)
        self.tau = jnp.asarray(tau)
        self.s = jnp.asarray(s)
        self.num_players = num_players

        # --- Build the moments filter for match updates ---
        self.filter_obj = moments.build_filter(
            get_init_params=partial(
                get_init_params,
                init_mean=self.init_mean,
                init_var=self.init_var,
                num_players=num_players,
            ),
            get_dynamics_params=partial(
                get_dynamics_params,
                tau=self.tau,
            ),
            get_observation_params=partial(
                get_observation_params,
                s=self.s,
            ),
        )

        # --- Factorializer: extract/join/insert player factors ---
        self.factorializer = build_factorializer(
            get_factorial_indices=get_factorial_inds,
        )

        # --- Single-player dynamics-only filter (for synchronisation) ---
        self.single_player_filter = moments.build_filter(
            get_init_params=partial(
                get_init_params,
                init_mean=self.init_mean,
                init_var=self.init_var,
                num_players=num_players,
            ),
            get_dynamics_params=partial(
                get_dynamics_params,
                tau=self.tau,
            ),
            get_observation_params=get_observation_params_noop,
        )

        # --- Smoother (shares the same dynamics) ---
        self.smoother_obj = moments.build_smoother(
            get_dynamics_params=partial(
                get_dynamics_params,
                tau=self.tau,
            ),
        )

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def initial_state(self) -> LinearizedKalmanFilterState:
        """Return the initial factorial state (all players at the prior).

        Uses ``filter_obj.init_prepare`` to create the state, then
        ``factorializer.factorialize_init_state`` to broadcast it to all
        players.

        Returns a ``LinearizedKalmanFilterState`` with:
        - mean: shape (num_players, 1)
        - chol_cov: shape (num_players, 1, 1)
        """
        init_state = self.filter_obj.init_prepare(None)
        init_state = self.factorializer.factorialize_init_state(init_state, None)
        return init_state

    # ------------------------------------------------------------------
    # Filtering (factorial) — JIT-compiled for speed
    # ------------------------------------------------------------------

    def filter(
        self,
        matches: WTATennisResults,
        initial_state: LinearizedKalmanFilterState | None = None,
    ) -> LinearizedKalmanFilterState:
        """Run the factorial moments filter over a batch of matches.

        Uses ``cuthbert.factorial.filter`` which automatically handles
        extract/join/marginalize/insert for each match.

        A dummy initial input is prepended at index 0 (required by cuthbert's
        factorial filter, matching the cuthberto-carlos pattern).

        Args:
            matches: A ``WTATennisResults`` NamedTuple with leading temporal
                dimension of length T.
            initial_state: Starting factorial state. If None, the filter
                initializes from the prior.

        Returns:
            Final factorial state after processing all matches.
        """
        from cuthbert.factorial import filter as factorial_filter

        # Prepend a dummy model input at index 0 (cuthbert convention)
        model_inputs = jtu.tree_map(
            lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0),
            matches,
        )

        if initial_state is None:
            # Let the factorial filter create the initial state from the prior
            result = factorial_filter(
                self.filter_obj,
                self.factorializer,
                model_inputs,
                output_factorial=False,
            )
            # Returns (init_factorial_state, local_states, final_factorial_state)
            _, _, final_factorial_state = result
        else:
            # Use provided initial state — run the scan manually
            # (same pattern as cuthberto-carlos update_moments_filtering.py)
            from jax import tree_util as tree
            from jax.lax import scan as lax_scan

            prep_model_inputs = tree.tree_map(lambda x: x[1:], model_inputs)

            def body_local(prev_factorial_state, prep_inp):
                factorial_inds = self.factorializer.get_factorial_indices(prep_inp)
                factorial_inds = jnp.asarray(factorial_inds)
                local_state = self.factorializer.extract_and_join(
                    prev_factorial_state, prep_inp
                )
                prep_state = self.filter_obj.filter_prepare(prep_inp)
                filtered_joint_state = self.filter_obj.filter_combine(
                    local_state, prep_state
                )
                local_factorial_filtered_state = self.factorializer.marginalize(
                    filtered_joint_state, len(factorial_inds)
                )
                factorial_state = self.factorializer.insert(
                    local_factorial_filtered_state, prev_factorial_state, factorial_inds
                )
                return factorial_state, None

            final_factorial_state, _ = lax_scan(
                body_local, initial_state, prep_model_inputs
            )

        return final_factorial_state

    # ------------------------------------------------------------------
    # Save / Load state and parameters
    # ------------------------------------------------------------------

    def save_state(self, factorial_state, filepath: str):
        """Save the factorial state and model parameters to a JSON file.

        Args:
            factorial_state: The factorial filter state to save.
            filepath: Path to the JSON file.
        """
        import json
        import os

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Serialize only the essential arrays (elem fields + mean_prev)
        # model_inputs is not needed for prediction
        from cuthbertlib.kalman import filtering

        elem = factorial_state.elem
        state_data = {
            "A": elem.A.tolist(),
            "b": elem.b.tolist(),
            "U": elem.U.tolist(),
            "eta": elem.eta.tolist(),
            "Z": elem.Z.tolist(),
            "ell": elem.ell.tolist(),
            "mean_prev": factorial_state.mean_prev.tolist(),
        }

        params_dict = {
            "init_mean": float(self.init_mean),
            "init_var": float(self.init_var),
            "tau": float(self.tau),
            "s": float(self.s),
            "num_players": self.num_players,
        }

        with open(filepath, "w") as f:
            json.dump({"params": params_dict, "state": state_data}, f)

    @classmethod
    def load_state(cls, filepath: str):
        """Load model parameters and factorial state from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Tuple of (model, factorial_state).
        """
        import json
        from cuthbertlib.kalman import filtering

        with open(filepath, "r") as f:
            data = json.load(f)

        params = data["params"]
        model = cls(
            init_mean=params["init_mean"],
            init_var=params["init_var"],
            tau=params["tau"],
            s=params["s"],
            num_players=params["num_players"],
        )

        # Reconstruct the factorial state
        state_data = data["state"]
        elem = filtering.FilterScanElement(
            A=jnp.array(state_data["A"]),
            b=jnp.array(state_data["b"]),
            U=jnp.array(state_data["U"]),
            eta=jnp.array(state_data["eta"]),
            Z=jnp.array(state_data["Z"]),
            ell=jnp.array(state_data["ell"]),
        )
        factorial_state = LinearizedKalmanFilterState(
            elem=elem,
            model_inputs=None,
            mean_prev=jnp.array(state_data["mean_prev"]),
        )
        return model, factorial_state

    # ------------------------------------------------------------------
    # Synchronisation (propagate players to a common timestamp)
    # ------------------------------------------------------------------

    def synchronize(
        self,
        factorial_state: LinearizedKalmanFilterState,
        dynamics_inputs: TennisDynamicsOnlyData,
    ) -> LinearizedKalmanFilterState:
        """Synchronize the factorial state to the most recent timestamp for
        each player.

        This extracts each player's factor, propagates it forward through the
        Wiener dynamics with no observation, and re-inserts it.

        Args:
            factorial_state: Current factorial state (may have different
                timestamps for different players).
            dynamics_inputs: Per-player dynamics data with a leading factorial
                axis of length ``num_players``.

        Returns:
            Synchronized factorial state.
        """
        num_players = factorial_state.mean.shape[0]

        out_factorial_final = vmap(self.factorializer.extract, in_axes=(None, 0))(
            factorial_state, jnp.arange(num_players)
        )
        state_prep = vmap(self.single_player_filter.filter_prepare)(dynamics_inputs)
        sync_factorial_final = vmap(self.single_player_filter.filter_combine)(
            out_factorial_final, state_prep
        )

        # ell shape (num_players,) -> (,)  [same hack as cuthberto_carlos]
        sync_factorial_final = sync_factorial_final._replace(
            elem=sync_factorial_final.elem._replace(
                ell=sync_factorial_final.elem.ell[0]
            )
        )
        return sync_factorial_final

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def propagate_and_predict(
        self,
        factorial_state: LinearizedKalmanFilterState,
        factorial_timestamp: float,
        match_data: WTATennisResults,
    ) -> TennisMatchPrediction:
        """Propagate the state to the time of the match and predict the outcome.

        This follows the cuthberto-carlos ``propagate_and_predict`` pattern:
        1. Extract the two players' factors from the factorial state
        2. Propagate each forward through the Wiener dynamics to the match time
        3. Predict the win probability from the propagated skill distributions

        Args:
            factorial_state: Current factorial filter state.
            factorial_timestamp: The timestamp the state is currently at.
            match_data: A single match's data (scalar arrays).

        Returns:
            ``TennisMatchPrediction`` with win probabilities.
        """
        # Build dynamics-only data for the two players
        dynamics_data = TennisDynamicsOnlyData(
            player_id=jnp.array([match_data.player1_id, match_data.player2_id]),
            timestamp=jnp.array([match_data.timestamp, match_data.timestamp]),
            timestamp_previous=jnp.array(
                [factorial_timestamp, factorial_timestamp]
            ),
        )

        # Extract the two players' factors
        factorial_state_two = vmap(
            self.factorializer.extract, in_axes=(None, 0)
        )(factorial_state, dynamics_data.player_id)

        # Propagate each through dynamics to the match time
        state_prep = vmap(self.single_player_filter.filter_prepare)(dynamics_data)
        propagated = vmap(self.single_player_filter.filter_combine)(
            factorial_state_two, state_prep
        )

        # Predict from propagated skills
        mu_1 = propagated.mean[0, 0]
        mu_2 = propagated.mean[1, 0]
        var_1 = jnp.square(propagated.chol_cov[0, 0, 0])
        var_2 = jnp.square(propagated.chol_cov[1, 0, 0])

        skill_diff_mean = mu_1 - mu_2
        skill_diff_var = var_1 + var_2
        skill_diff_std = jnp.sqrt(skill_diff_var)

        s = self.s
        p_player1_win = _sigmoid(skill_diff_mean / jnp.sqrt(s**2 + skill_diff_var))
        p_player2_win = 1.0 - p_player1_win

        return TennisMatchPrediction(
            p_player1_win=p_player1_win,
            p_player2_win=p_player2_win,
            skill_diff_mean=skill_diff_mean,
            skill_diff_std=skill_diff_std,
            player1_mean=propagated.mean[0:1],
            player2_mean=propagated.mean[1:2],
            player1_var=var_1.reshape(1),
            player2_var=var_2.reshape(1),
        )

    def predict_match(
        self,
        player1_idx: int,
        player2_idx: int,
        factorial_state: LinearizedKalmanFilterState,
    ) -> TennisMatchPrediction:
        """Predict the win probabilities for a match between two players.

        Extracts the two players' skill distributions from the factorial state
        and computes the probability that player 1 wins using the analytical
        probit approximation:

            If d = x_i - x_j ~ N(mu_d, sigma_d^2), then
            P(p1 wins) = sigmoid(mu_d / sqrt(s^2 + sigma_d^2))

        This is exact for the probit link and an excellent approximation for
        the logistic link.

        Args:
            player1_idx: Factor index of player 1.
            player2_idx: Factor index of player 2.
            factorial_state: Current factorial filter/smoothed state.

        Returns:
            ``TennisMatchPrediction`` with win probabilities.
        """
        # Extract individual player states (1D skills)
        p1_state = self.factorializer.extract(factorial_state, player1_idx)
        p2_state = self.factorializer.extract(factorial_state, player2_idx)

        # Skill means and variances (scalar, shape (1,))
        mu_1 = p1_state.mean[0]
        mu_2 = p2_state.mean[0]
        var_1 = (p1_state.chol_cov @ p1_state.chol_cov.T)[0, 0]
        var_2 = (p2_state.chol_cov @ p2_state.chol_cov.T)[0, 0]

        # Skill difference distribution
        skill_diff_mean = mu_1 - mu_2
        skill_diff_var = var_1 + var_2  # independent factors
        skill_diff_std = jnp.sqrt(skill_diff_var)

        # Probit approximation for sigmoid marginal:
        # P(win) = E[sigmoid(d / s)] ≈ sigmoid(mu_d / sqrt(s^2 + sigma_d^2))
        s = self.s
        p_player1_win = _sigmoid(skill_diff_mean / jnp.sqrt(s**2 + skill_diff_var))
        p_player2_win = 1.0 - p_player1_win

        return TennisMatchPrediction(
            p_player1_win=p_player1_win,
            p_player2_win=p_player2_win,
            skill_diff_mean=skill_diff_mean,
            skill_diff_std=skill_diff_std,
            player1_mean=p1_state.mean,
            player2_mean=p2_state.mean,
            player1_var=var_1.reshape(1),
            player2_var=var_2.reshape(1),
        )


# ---------------------------------------------------------------------------
# Parameter optimization (gradient-based, following cuthberto-carlos)
# ---------------------------------------------------------------------------


def _positive(raw: ArrayLike, floor: float = 1e-6) -> Array:
    """Map an unconstrained scalar to a positive scalar via softplus."""
    return jax.nn.softplus(raw) + floor


def _inverse_positive(value: float, floor: float = 1e-6) -> Array:
    """Map a positive scalar back to the unconstrained softplus scale."""
    return jnp.log(jnp.expm1(jnp.asarray(value - floor)))


def _constrain_params(raw_params: dict[str, Array]) -> dict[str, Array]:
    """Transform unconstrained optimizer parameters to model parameters.

    Trainable parameters:
        - tau: dynamics rate (positive)
        - s: observation scale (positive)
        - init_var: initial variance (positive)

    init_mean is fixed at 0.0 (skills are relative).
    """
    return {
        "tau": _positive(raw_params["tau"]),
        "s": _positive(raw_params["s"]),
        "init_var": _positive(raw_params["init_var"]),
    }


def _format_params(params: dict[str, Array]) -> str:
    """Format constrained parameters for logging."""
    return (
        f"tau={float(params['tau']):.6g}, "
        f"s={float(params['s']):.6g}, "
        f"init_var={float(params['init_var']):.6g}"
    )


def train(
    matches: WTATennisResults,
    num_players: int,
    steps: int = 200,
    learning_rate: float = 0.1,
    log_every: int = 10,
    init_mean: float = 0.0,
    initial_tau: float = 0.1,
    initial_s: float = 1.0,
    initial_init_var: float = 1.0,
) -> tuple[dict[str, Array], list[dict]]:
    """Train model parameters by maximizing the log normalizing constant.

    Uses gradient-based optimization (Adam) on the filter's log-likelihood,
    following the cuthberto-carlos ``train_moments.py`` pattern.

    The log normalizing constant of the filter is the log-likelihood of the
    observed match outcomes under the model. Maximizing it optimizes the
    parameters (tau, s, init_var).

    Args:
        matches: ``WTATennisResults`` with training match data.
        num_players: Total number of players.
        steps: Number of gradient steps.
        learning_rate: Adam learning rate.
        log_every: Print progress every N steps.
        init_mean: Fixed initial mean (default 0.0).
        initial_tau: Initial dynamics rate.
        initial_s: Initial observation scale.
        initial_init_var: Initial variance.

    Returns:
        Tuple of (constrained_params, training_history).
    """
    import optax
    from cuthbert.factorial import filter as factorial_filter

    init_mean_arr = jnp.asarray(init_mean)

    # Prepend dummy initial input (cuthbert convention)
    model_inputs = jtu.tree_map(
        lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0),
        matches,
    )

    # Build factorializer (doesn't depend on params)
    factorializer = build_factorializer(
        get_factorial_indices=get_factorial_inds,
    )

    # Raw (unconstrained) parameters for the optimizer
    raw_params = {
        "tau": _inverse_positive(initial_tau),
        "s": _inverse_positive(initial_s),
        "init_var": _inverse_positive(initial_init_var),
    }

    def log_normalizing_constant(raw: dict[str, Array]) -> Array:
        """Run the filter and return the final log normalizing constant."""
        params = _constrain_params(raw)
        filter_obj = moments.build_filter(
            get_init_params=partial(
                get_init_params,
                init_mean=init_mean_arr,
                init_var=params["init_var"],
                num_players=num_players,
            ),
            get_dynamics_params=partial(
                get_dynamics_params,
                tau=params["tau"],
            ),
            get_observation_params=partial(
                get_observation_params,
                s=params["s"],
            ),
        )
        final_state = factorial_filter(
            filter_obj,
            factorializer,
            model_inputs,
            output_factorial=False,
        )[-1]
        return final_state.log_normalizing_constant

    def loss(raw: dict[str, Array]) -> Array:
        """Negative log normalizing constant for minimization."""
        return -log_normalizing_constant(raw)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(raw_params)
    value_and_grad = jax.jit(jax.value_and_grad(loss))

    history = []
    for step in range(steps + 1):
        loss_value, grads = value_and_grad(raw_params)
        logz = -loss_value
        history.append(
            {
                "step": step,
                "loss": float(loss_value),
                "log_normalizing_constant": float(logz),
            }
        )

        if step % max(log_every, 1) == 0 or step == steps:
            params = _constrain_params(raw_params)
            print(
                f"  step={step:05d}, "
                f"log_lik={float(logz):.6f}, "
                f"loss={float(loss_value):.6f}, "
                f"{_format_params(params)}"
            )

        if step == steps:
            break

        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)

    final_params = _constrain_params(raw_params)
    return final_params, history