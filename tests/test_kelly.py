from __future__ import annotations

from fractions import Fraction

import pytest

from probability_and_markets.bankroll import BayesianCoin, even_money_kelly, kelly_fraction


def test_even_money_kelly() -> None:
    assert even_money_kelly(Fraction(1, 2)) == 0
    assert even_money_kelly(Fraction(11, 20)) == Fraction(1, 10)
    assert even_money_kelly(Fraction(2, 3)) == Fraction(1, 3)
    assert even_money_kelly(Fraction(1, 3)) == Fraction(-1, 3)


def test_general_odds_kelly() -> None:
    # p=0.6, b=1 already covered. b=2 (win 2:1): f* = 0.6 - 0.4/2 = 0.4
    assert kelly_fraction(win_probability=Fraction(3, 5), odds=Fraction(2)) == Fraction(2, 5)


def test_uniform_prior_is_one_half() -> None:
    coin = BayesianCoin()
    assert coin.mean == Fraction(1, 2)
    assert coin.variance == Fraction(1, 12)  # Beta(1,1) is Uniform[0,1]


def test_seven_heads_three_tails() -> None:
    posterior = BayesianCoin().update(heads=7, tails=3)
    assert posterior.alpha == 8
    assert posterior.beta == 4
    assert posterior.mean == Fraction(2, 3)
    assert posterior.kelly_even_money() == Fraction(1, 3)


def test_update_rejects_negative() -> None:
    with pytest.raises(ValueError):
        BayesianCoin().update(heads=-1, tails=0)
