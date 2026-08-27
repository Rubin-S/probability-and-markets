from __future__ import annotations

from fractions import Fraction

import numpy as np

from probability_and_markets.monte_carlo import fair_walk_stop_at_plus_one
from probability_and_markets.probability import (
    fair_game_stopped_at_plus_one_expectation,
    gamblers_ruin_probability,
)


def test_fair_ruin_is_linear() -> None:
    assert gamblers_ruin_probability(bankroll=0, total=10) == 1
    assert gamblers_ruin_probability(bankroll=10, total=10) == 0
    assert gamblers_ruin_probability(bankroll=10, total=20) == Fraction(1, 2)
    assert gamblers_ruin_probability(bankroll=1, total=5) == Fraction(4, 5)


def test_unfair_ruin_closed_form() -> None:
    # p = 2/3, r = 1/2, i=1, n=2.
    # P(hit n) = (1 - r^i) / (1 - r^n) = (1 - 1/2) / (1 - 1/4) = (1/2)/(3/4) = 2/3
    # P(ruin) = 1/3
    p = gamblers_ruin_probability(bankroll=1, total=2, win_probability=Fraction(2, 3))
    assert p == Fraction(1, 3)


def test_naive_stopped_expectation_is_plus_one() -> None:
    assert fair_game_stopped_at_plus_one_expectation() == 1


def test_finite_horizon_mean_stays_near_zero() -> None:
    """OST holds for bounded time: E[S_{n ∧ τ}] = 0 even if most paths have hit +1."""
    rng = np.random.default_rng(4)
    walked = fair_walk_stop_at_plus_one(n_paths=12_000, max_steps=800, rng=rng)
    assert walked.hit_rate > 0.9
    assert abs(walked.mean_position) < 0.08
    # The hit paths sit at +1; the remainder must be deeply negative on average.
    losers = walked.positions[~walked.hit_plus_one]
    assert losers.mean() < 0
