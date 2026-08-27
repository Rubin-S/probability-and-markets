# probability-and-markets

Small Python experiments for **probability, expected value, and a toy
market-maker**. Written as interview prep for [Rubin S](https://github.com/Rubin-S)
(NIT Puducherry, ECE): first-principles code with closed forms, Monte Carlo
checks, and tests that lock the math.

Not a trading firm, not a strategy, not affiliated with Jane Street, not live
PnL, not market data.

## How to run

Python 3.11+. From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Optional extras (print tables / write plots):

```bash
python experiments/run_probability.py
python experiments/run_bankroll.py
python experiments/run_market_making.py   # writes experiments/figures/*.png
ruff check src tests experiments
```

## What the numbers mean

### 1. Probability / EV (`src/probability_and_markets/probability.py`)

| Quantity | Closed form | Value |
|---|---|---|
| Two fair dice, P(sum = 7) | 6/36 | 1/6 |
| Two fair dice, E[max] | Σ k(2k−1)/36 | 161/36 ≈ 4.472 |
| Two fair dice, P(at least one 6) | 1 − (5/6)² | 11/36 |
| Ace + ten-value in two cards (52-card deck) | 2·(4/52)·(16/51) | 32/663 ≈ 0.0483 |
| P(disease \| +) with prior 1%, 99% sens/spec | Bayes | 99/198 ≈ 0.5 |
| Fair gambler's ruin, start 10, absorb 0 or 20 | (20−10)/20 | 1/2 |

**Optional stopping.** A fair ±1 random walk stopped at +1 has S_τ = 1 almost
surely, so a naive reading says “bet until you are ahead, EV = +1.” For every
*finite* horizon n, E[S_{n ∧ τ}] = 0 (bounded stopping time, martingale). The
theorem’s hypotheses fail as n → ∞ because E[τ] = ∞: rare, deep-negative paths
cancel the many +1 finishes. The Monte Carlo in the CLI prints that mean near 0
and a high hit rate together — that is the point.

### 2. Toy market making (`market_making.py`)

Each step: true value jumps, the MM quotes around *last* value (plus inventory
skew), then one trader arrives. Informed traders trade only if news has crossed
the quote; noise traders are 50/50. Mark-to-market PnL = cash + inventory · V.

What to notice, briefly (full writeup: [`experiments/market_making.md`](experiments/market_making.md)):

- A spread is not an edge. If informed flow is thick and quotes are stale,
  expected PnL is negative.
- Inventory is the channel for adverse selection. Skew is a risk control, not
  a forecast.
- One PnL path is an anecdote; look at the distribution.

### 3. Kelly / biased coin (`bankroll.py`)

Even-money Kelly is f\* = 2p − 1. After 7 heads and 3 tails, a uniform Beta(1,1)
prior becomes Beta(8,4); posterior mean 2/3; myopic Kelly 1/3. That plug-in
ignores estimation error — practitioners often haircut to half-Kelly. This
module does not tell you to bet.

## Layout

```
src/probability_and_markets/   # library
experiments/                   # CLIs + MM writeup
tests/                         # exact Fractions + seeded Monte Carlo
```

## Scope (honest)

- Discrete toy models. No LOB, no latency, no real tapes, no execution.
- Dependencies: `numpy`, `matplotlib`; `pytest` and `ruff` for development.
- MIT license (see `LICENSE`).
