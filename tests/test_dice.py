from __future__ import annotations

from fractions import Fraction

import numpy as np

from probability_and_markets.monte_carlo import two_dice_max_mean, two_dice_sum_frequency
from probability_and_markets.probability import (
    two_dice_at_least_one_six,
    two_dice_max_expectation,
    two_dice_sum_probability,
)


def test_two_dice_sum_is_exact_fraction() -> None:
    ways = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    for total, n in ways.items():
        assert two_dice_sum_probability(total) == Fraction(n, 36)
    assert two_dice_sum_probability(7) == Fraction(1, 6)
    assert two_dice_sum_probability(1) == 0
    assert two_dice_sum_probability(13) == 0
    assert sum(two_dice_sum_probability(t) for t in range(2, 13)) == 1


def test_two_dice_max_expectation() -> None:
    # Direct sum: P(max=k)=(2k-1)/36.
    direct = sum(Fraction(k * (2 * k - 1), 36) for k in range(1, 7))
    assert two_dice_max_expectation() == Fraction(161, 36)
    assert two_dice_max_expectation() == direct


def test_at_least_one_six() -> None:
    assert two_dice_at_least_one_six() == Fraction(11, 36)
    # Complement: both in {1..5}.
    assert two_dice_at_least_one_six() == 1 - Fraction(25, 36)


def test_monte_carlo_sum_7_tracks_closed_form() -> None:
    rng = np.random.default_rng(0)
    mc = two_dice_sum_frequency(7, n_trials=80_000, rng=rng)
    assert abs(mc - float(Fraction(1, 6))) < 0.01


def test_monte_carlo_max_tracks_closed_form() -> None:
    rng = np.random.default_rng(1)
    mc = two_dice_max_mean(n_trials=80_000, rng=rng)
    assert abs(mc - float(Fraction(161, 36))) < 0.03
