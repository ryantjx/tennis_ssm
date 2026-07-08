"""NamedTuples to be used as model_inputs representing the data required for the SSM."""

from typing import NamedTuple

from jax import Array


class WTATennisResults(NamedTuple):
    """NamedTuple containing JAX arrays for a WTA tennis match (or batch of matches).

    Each field should be a JAX array. For a single match these are scalar arrays;
    for a batch they have a leading temporal dimension of length T.

    The model is a 1D scalar-skill sigmoid model (see README): each player has
    a single latent skill, and the observation is a Bernoulli win/loss outcome
    with probability ``sigmoid((skill_i - skill_j) / s)``.

    Attributes:
        match_index: Unique integer index for each match, starting from 0.
        player1_id: Integer ID for player 1.
        player2_id: Integer ID for player 2.
        winner: 1.0 if player 1 won, 0.0 if player 2 won.
        timestamp: Integer days since the origin date for the match.
        player1_timestamp_previous: Timestamp for player 1's previous match.
            0 if no previous match (sentinel — origin is far in the past).
        player2_timestamp_previous: Timestamp for player 2's previous match.
    """

    match_index: Array
    player1_id: Array
    player2_id: Array
    winner: Array
    timestamp: Array
    player1_timestamp_previous: Array
    player2_timestamp_previous: Array


class TennisDynamicsOnlyData(NamedTuple):
    """NamedTuple for propagating a single player's state through the dynamics
    with no observation (e.g. between matches or for synchronisation).

    Attributes:
        player_id: Integer ID for the player.
        timestamp: Integer days since origin for the time to propagate to.
        timestamp_previous: Integer days since origin for the previous state.
    """

    player_id: Array
    timestamp: Array
    timestamp_previous: Array