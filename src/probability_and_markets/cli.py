"""Command-line entry points used by the ``pam-*`` scripts and by experiments/."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np

from probability_and_markets.bankroll import BayesianCoin, even_money_kelly
from probability_and_markets.market_making import MarketMakingConfig, simulate
from probability_and_markets.monte_carlo import (
    disease_test_positive_posterior_frequency,
    fair_walk_stop_at_plus_one,
    two_dice_max_mean,
    two_dice_sum_frequency,
)
from probability_and_markets.probability import (
    blackjack_natural_probability,
    disease_posterior,
    gamblers_ruin_probability,
    two_dice_at_least_one_six,
    two_dice_max_expectation,
    two_dice_sum_probability,
)


def _fmt(frac: Fraction) -> str:
    return f"{frac}  ({float(frac):.6f})"


def run_probability(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Closed form vs Monte Carlo.")
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    print("Two dice, P(sum = 7)")
    exact = two_dice_sum_probability(7)
    mc = two_dice_sum_frequency(7, n_trials=args.trials, rng=rng)
    print(f"  exact {_fmt(exact)}")
    print(f"  monte carlo {mc:.6f}  (n={args.trials})")

    print("Two dice, E[max]")
    print(f"  exact {_fmt(two_dice_max_expectation())}")
    print(f"  monte carlo {two_dice_max_mean(n_trials=args.trials, rng=rng):.6f}")

    print("Two dice, P(at least one six)")
    print(f"  exact {_fmt(two_dice_at_least_one_six())}")

    print("Blackjack natural (Ace + 10-value)")
    print(f"  exact {_fmt(blackjack_natural_probability())}")

    prior, sens, spec = Fraction(1, 100), Fraction(99, 100), Fraction(99, 100)
    post = disease_posterior(prior=prior, sensitivity=sens, specificity=spec)
    print("Bayes: 1% base rate, 99% sensitivity, 99% specificity")
    print(f"  P(disease | +) exact {_fmt(post)}")
    mc_post = disease_test_positive_posterior_frequency(
        prior=float(prior),
        sensitivity=float(sens),
        specificity=float(spec),
        n_people=args.trials,
        rng=rng,
    )
    print(f"  monte carlo {mc_post:.6f}")

    print("Gambler's ruin, fair, start 10, absorb 0 or 20")
    print(f"  P(ruin) exact {_fmt(gamblers_ruin_probability(bankroll=10, total=20))}")

    print("Optional stopping: fair ±1 walk, stop at +1, finite horizon")
    walked = fair_walk_stop_at_plus_one(n_paths=20_000, max_steps=2_000, rng=rng)
    print(f"  mean S_(n∧τ) = {walked.mean_position:.4f}  (martingale ⇒ ~0)")
    print(f"  hit +1 by n={2000}: {walked.hit_rate:.3f}")
    print("  Naive 'stop when ahead' would claim E = +1; OST does not apply (E[τ]=∞).")


def run_market_making(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Toy market-making simulation + plots.")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from probability_and_markets.plots import write_market_making_plots

    args.outdir.mkdir(parents=True, exist_ok=True)
    baseline = simulate(MarketMakingConfig(seed=args.seed))
    no_informed = simulate(MarketMakingConfig(seed=args.seed, informed_prob=0.0))
    tight = simulate(MarketMakingConfig(seed=args.seed, half_spread=0.02, informed_prob=0.5))
    no_skew = simulate(MarketMakingConfig(seed=args.seed, inventory_skew=0.0))

    print("Toy MM (see experiments/market_making.md)")
    print(
        f"  baseline  mean terminal PnL {baseline.mean_terminal_pnl:+.3f}  "
        f"std {baseline.std_terminal_pnl:.3f}"
    )
    print(
        f"  noise only (α=0)  mean {no_informed.mean_terminal_pnl:+.3f}  "
        f"std {no_informed.std_terminal_pnl:.3f}"
    )
    print(
        f"  tight spread, α=0.5  mean {tight.mean_terminal_pnl:+.3f}  "
        f"std {tight.std_terminal_pnl:.3f}"
    )
    print(
        f"  no inventory skew  mean {no_skew.mean_terminal_pnl:+.3f}  "
        f"|q|_T mean {np.abs(no_skew.inventory[:, -1]).mean():.2f}"
    )
    print(
        f"  with skew           mean {baseline.mean_terminal_pnl:+.3f}  "
        f"|q|_T mean {np.abs(baseline.inventory[:, -1]).mean():.2f}"
    )

    paths = write_market_making_plots(
        outdir=args.outdir,
        baseline=baseline,
        no_informed=no_informed,
        tight=tight,
        no_skew=no_skew,
    )
    for p in paths:
        print(f"  wrote {p}")


def run_bankroll(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kelly + Bayesian coin.")
    parser.parse_args(argv)

    p = Fraction(11, 20)
    print(f"Even-money Kelly at p = {p}  →  f* = {even_money_kelly(p)}")
    coin = BayesianCoin()  # uniform prior
    print(f"Uniform prior Beta(1,1), mean = {coin.mean}")
    after = coin.update(heads=7, tails=3)
    print(f"After 7H 3T: posterior mean = {after.mean}  ({float(after.mean):.3f})")
    print(f"Myopic even-money Kelly using that mean: {after.kelly_even_money()}")
    print("Half-Kelly is after.kelly_even_money()/2 — common haircut for estimation error.")


if __name__ == "__main__":
    run_probability()
