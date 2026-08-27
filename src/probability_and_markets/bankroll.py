"""Kelly sizing and Bayesian updating of a biased coin.

Small, exact formulas. No claimed betting system; Kelly is a growth criterion
under a known (or estimated) edge, and it is aggressive in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def kelly_fraction(*, win_probability: Fraction, odds: Fraction) -> Fraction:
    """Optimal fraction of wealth to bet when a win pays ``odds``:1 net.

    f* = p - q / b, with q = 1-p and b = odds (net units won per unit risked).
    Negative means no bet. ``odds = 1`` is even money, so f* = 2p - 1.
    """
    p = win_probability
    if not 0 <= p <= 1:
        raise ValueError("win_probability must be in [0, 1]")
    if odds <= 0:
        raise ValueError("odds must be positive")
    q = 1 - p
    return p - q / odds


def even_money_kelly(win_probability: Fraction) -> Fraction:
    return kelly_fraction(win_probability=win_probability, odds=Fraction(1))


@dataclass
class BayesianCoin:
    """Beta–Bernoulli posterior for P(heads).

    Prior Beta(α, β); after ``heads`` successes and ``tails`` failures the
    posterior is Beta(α + heads, β + tails). Mean = α' / (α' + β').
    """

    alpha: Fraction = Fraction(1)
    beta: Fraction = Fraction(1)

    def update(self, *, heads: int, tails: int) -> BayesianCoin:
        if heads < 0 or tails < 0:
            raise ValueError("counts must be non-negative")
        return BayesianCoin(alpha=self.alpha + heads, beta=self.beta + tails)

    @property
    def mean(self) -> Fraction:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> Fraction:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1))

    def kelly_even_money(self) -> Fraction:
        """Plug the posterior *mean* into even-money Kelly (myopic, not fully Bayes)."""
        return even_money_kelly(self.mean)
