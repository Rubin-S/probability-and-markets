from __future__ import annotations

from fractions import Fraction

import numpy as np

from probability_and_markets.monte_carlo import hypergeometric_frequency
from probability_and_markets.probability import blackjack_natural_probability, hypergeometric_pmf


def test_blackjack_natural() -> None:
    assert blackjack_natural_probability() == Fraction(32, 663)
    # Equivalent counting: 2 * 4 * 16 / (52 * 51).
    assert blackjack_natural_probability() == Fraction(128, 2652)


def test_five_card_exactly_one_ace() -> None:
    p = hypergeometric_pmf(n_success_states=4, population=52, draws=5, observed=1)
    # C(4,1)*C(48,4) / C(52,5)
    from math import comb

    assert p == Fraction(comb(4, 1) * comb(48, 4), comb(52, 5))


def test_hypergeometric_is_a_distribution() -> None:
    total = sum(
        hypergeometric_pmf(n_success_states=4, population=52, draws=5, observed=k)
        for k in range(0, 6)
    )
    assert total == 1


def test_impossible_hands_are_zero() -> None:
    assert hypergeometric_pmf(n_success_states=4, population=52, draws=5, observed=5) == 0
    assert hypergeometric_pmf(n_success_states=4, population=52, draws=5, observed=-1) == 0


def test_monte_carlo_aces() -> None:
    rng = np.random.default_rng(2)
    exact = hypergeometric_pmf(n_success_states=4, population=52, draws=5, observed=1)
    mc = hypergeometric_frequency(
        n_success_states=4,
        population=52,
        draws=5,
        observed=1,
        n_trials=40_000,
        rng=rng,
    )
    assert abs(mc - float(exact)) < 0.015
