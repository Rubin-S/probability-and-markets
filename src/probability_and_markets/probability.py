"""Closed-form probability and expected-value calculations.

Every function here returns an exact ``Fraction`` (or a pair of them). Monte Carlo
counterparts live in :mod:`probability_and_markets.monte_carlo` and are checked
against these values in tests.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb


def two_dice_sum_probability(total: int) -> Fraction:
    """P(D1 + D2 = total) for two independent fair six-sided dice."""
    if total < 2 or total > 12:
        return Fraction(0)
    ways = sum(1 for face in range(1, 7) if 1 <= total - face <= 6)
    return Fraction(ways, 36)


def two_dice_max_expectation() -> Fraction:
    """E[max(D1, D2)] for two independent fair six-sided dice.

    P(max = k) = (k^2 - (k-1)^2) / 36 = (2k - 1) / 36, so
    E[max] = sum_{k=1}^{6} k(2k-1)/36 = 161/36.
    """
    return Fraction(161, 36)


def two_dice_at_least_one_six() -> Fraction:
    """P(at least one six) = 1 - (5/6)^2 = 11/36."""
    return Fraction(11, 36)


def hypergeometric_pmf(
    *, n_success_states: int, population: int, draws: int, observed: int
) -> Fraction:
    """P(exactly ``observed`` successes) when drawing without replacement.

    Standard hypergeometric: population ``N``, ``K`` success states, ``n`` draws, ``k`` observed.
    """
    n, k = draws, observed
    n_pop, k_states = population, n_success_states
    if k < 0 or k > n or k > k_states or n - k > n_pop - k_states:
        return Fraction(0)
    return Fraction(comb(k_states, k) * comb(n_pop - k_states, n - k), comb(n_pop, n))


def blackjack_natural_probability() -> Fraction:
    """P(Ace and ten-value in two cards from a 52-card deck), order-insensitive.

    4 aces, 16 tens (10, J, Q, K). Two orderings: 2 * (4/52) * (16/51) = 32/663.
    """
    return Fraction(2 * 4 * 16, 52 * 51)


def disease_posterior(
    *,
    prior: Fraction,
    sensitivity: Fraction,
    specificity: Fraction,
) -> Fraction:
    """P(disease | positive test) from Bayes' rule.

    P(+|D) = sensitivity, P(-|no D) = specificity,
    P(+) = sensitivity * prior + (1 - specificity) * (1 - prior).
    """
    if not 0 <= prior <= 1:
        raise ValueError("prior must be in [0, 1]")
    if not 0 <= sensitivity <= 1 or not 0 <= specificity <= 1:
        raise ValueError("sensitivity and specificity must be in [0, 1]")
    p_positive = sensitivity * prior + (1 - specificity) * (1 - prior)
    if p_positive == 0:
        raise ZeroDivisionError("positive-test probability is zero")
    return (sensitivity * prior) / p_positive


def gamblers_ruin_probability(
    *,
    bankroll: int,
    total: int,
    win_probability: Fraction = Fraction(1, 2),
) -> Fraction:
    """P(hit 0 before ``total``) starting from ``bankroll``, unit bets.

    Fair case (p = 1/2): (total - bankroll) / total.
    Unfair: let r = (1-p)/p. Then P(ruin) = (r^{bankroll} - r^{total}) / (1 - r^{total})
    for p ≠ 1/2, which is equivalently
    (1 - r^{bankroll}) / (1 - r^{total}) for the complementary absorption probability.
    """
    i, n = bankroll, total
    if n <= 0:
        raise ValueError("total must be positive")
    if not 0 <= i <= n:
        raise ValueError("bankroll must lie in [0, total]")
    if i == 0:
        return Fraction(1)
    if i == n:
        return Fraction(0)
    p = win_probability
    if p == Fraction(1, 2):
        return Fraction(n - i, n)
    if p <= 0 or p >= 1:
        raise ValueError("win_probability must be in (0, 1)")
    r = (1 - p) / p
    # P(hit n before 0) = (1 - r^i) / (1 - r^n); ruin is the complement.
    hit_upper = (1 - r**i) / (1 - r**n)
    return 1 - hit_upper


def fair_game_stopped_at_plus_one_expectation() -> Fraction:
    """If optional stopping applied naively: stop a fair ±1 walk at +1, 'E[S]' = 1.

    The walk is a martingale, so E[S_{n ∧ τ}] = 0 for every finite n. The limit
    S_τ = 1 almost surely, but E[τ] = ∞, so the optional stopping theorem's
    hypotheses fail and you cannot pocket +1 in expectation. See Monte Carlo.
    """
    return Fraction(1)
