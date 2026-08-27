from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from probability_and_markets.monte_carlo import disease_test_positive_posterior_frequency
from probability_and_markets.probability import disease_posterior


def test_classic_rare_disease() -> None:
    # 1% prevalence, 99% sensitivity, 99% specificity → posterior 1/2.
    p = disease_posterior(
        prior=Fraction(1, 100),
        sensitivity=Fraction(99, 100),
        specificity=Fraction(99, 100),
    )
    assert p == Fraction(99, 198)
    assert p == Fraction(1, 2)


def test_perfect_test_recovers_certainty() -> None:
    p = disease_posterior(prior=Fraction(1, 5), sensitivity=Fraction(1), specificity=Fraction(1))
    assert p == Fraction(1)


def test_prior_zero_stays_zero() -> None:
    p = disease_posterior(
        prior=Fraction(0), sensitivity=Fraction(99, 100), specificity=Fraction(9, 10)
    )
    assert p == 0


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        disease_posterior(prior=Fraction(2), sensitivity=Fraction(1, 2), specificity=Fraction(1, 2))


def test_monte_carlo_posterior() -> None:
    rng = np.random.default_rng(3)
    exact = disease_posterior(
        prior=Fraction(1, 100),
        sensitivity=Fraction(99, 100),
        specificity=Fraction(99, 100),
    )
    mc = disease_test_positive_posterior_frequency(
        prior=0.01,
        sensitivity=0.99,
        specificity=0.99,
        n_people=200_000,
        rng=rng,
    )
    assert abs(mc - float(exact)) < 0.04
