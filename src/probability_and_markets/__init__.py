"""Interview-prep experiments: probability, expected value, toy market making.

Not affiliated with Jane Street. No live markets, no claimed PnL.
"""

from .bankroll import BayesianCoin, even_money_kelly, kelly_fraction
from .market_making import MarketMakingConfig, simulate
from .probability import (
    blackjack_natural_probability,
    disease_posterior,
    gamblers_ruin_probability,
    hypergeometric_pmf,
    two_dice_max_expectation,
    two_dice_sum_probability,
)

__version__ = "0.1.0"

__all__ = [
    "BayesianCoin",
    "MarketMakingConfig",
    "blackjack_natural_probability",
    "disease_posterior",
    "even_money_kelly",
    "gamblers_ruin_probability",
    "hypergeometric_pmf",
    "kelly_fraction",
    "simulate",
    "two_dice_max_expectation",
    "two_dice_sum_probability",
]
