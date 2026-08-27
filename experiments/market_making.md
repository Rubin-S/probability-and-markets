# What the market-making plots show

This is a **toy**. Quotes sit around *last period's* true value; this period's
Gaussian news is already in `V_t` before anyone trades. Informed traders only
trade when that news has crossed the quote. Noise traders trade randomly.

Run from the repo root:

```bash
python experiments/run_market_making.py
```

Figures land in `experiments/figures/` (gitignored). Three pictures:

## 1. `pnl_paths.png` — mean PnL and a 10–90% band

- **Noise only (α = 0).** The MM is paid the half-spread on almost every fill.
  Mean PnL drifts up. The band widens because inventory still couples you to
  `V_t` whenever volatility is on.
- **Baseline (α = 0.35, half-spread 0.10).** Informed flow picks off stale
  quotes after news. Spread harvest and adverse selection fight; the mean is
  much closer to zero (or negative) than the noise-only line.
- **Tight spread, more informed.** You are the snack. Mean PnL goes negative.
  A market maker who only looks at fill count, not *who* filled them, will
  narrate a busy, losing book as “making markets well.”

What a trader should notice: **the sign of expected PnL is not “I posted a
spread.”** It is whether the spread covers the information in the flow. Markouts
(PnL after the next move) are the honest scoreboard; fill-rate is not.

## 2. `inventory.png` — mean |inventory|

Without skew, inventory is close to a random walk until it hits the cap.
With skew, quotes lean: long inventory → lower bid and ask → you become a
seller, so the position mean-reverts.

What a trader should notice: **inventory is how adverse selection gets into
the P&L.** A picked-off ask leaves you short into a higher `V`. Skew is not
an alpha signal; it is a risk control so one-way flow cannot run you to the
rail. The cap in this toy is a crude substitute for a risk limit.

## 3. `terminal_pnl_hist.png` — distribution, not a path

Noise-only terminal PnL sits mostly to the right of zero. Tight + informed
sits mostly to the left. Both distributions are wide relative to the mean.

What a trader should notice: **one path is an anecdote.** A single upward
PnL line is consistent with negative expectation. Interview translations:
edge vs variance, adverse selection vs bid–ask, inventory as residual risk,
optional stopping on a lucky path (“I’ll flatten when I’m up”) does not
create expectation that was not there.

Default knobs (see `MarketMakingConfig`): `value_vol=0.15`, `half_spread=0.10`,
`informed_prob=0.35`, `inventory_skew=0.02`. Change one at a time. There is
no calibrated “real market” hiding in the numbers.
