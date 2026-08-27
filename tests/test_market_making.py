from __future__ import annotations

import numpy as np

from probability_and_markets.market_making import MarketMakingConfig, simulate


def test_noise_only_constant_value_earns_the_spread() -> None:
    """σ=0, α=0: every fill is at mid ± half_spread, so each fill is worth half_spread."""
    cfg = MarketMakingConfig(
        n_steps=500,
        n_paths=40,
        value_vol=0.0,
        informed_prob=0.0,
        inventory_skew=0.0,
        half_spread=0.10,
        max_inventory=10_000,
        seed=5,
    )
    res = simulate(cfg)
    # Every step a noise trader arrives and is filled (inventory cap is loose).
    fills = res.n_noise_fills
    assert np.all(fills == cfg.n_steps)
    # With constant V, Δmtm per fill = half_spread.
    expected = cfg.n_steps * cfg.half_spread
    assert abs(res.mean_terminal_pnl - expected) < 1e-9


def test_informed_only_small_spread_loses() -> None:
    """Informed traders only take when news has crossed the quote → negative EV."""
    cfg = MarketMakingConfig(
        n_steps=800,
        n_paths=80,
        value_vol=0.25,
        informed_prob=1.0,
        inventory_skew=0.0,
        half_spread=0.02,
        seed=6,
    )
    res = simulate(cfg)
    assert res.mean_terminal_pnl < 0
    assert res.n_informed_fills.mean() > 100


def test_inventory_skew_reduces_position_size() -> None:
    shared = dict(n_steps=1_000, n_paths=60, value_vol=0.15, informed_prob=0.2, seed=7)
    flat = simulate(MarketMakingConfig(inventory_skew=0.0, **shared))
    skewed = simulate(MarketMakingConfig(inventory_skew=0.05, **shared))
    assert np.abs(skewed.inventory[:, -1]).mean() < np.abs(flat.inventory[:, -1]).mean()


def test_shapes() -> None:
    res = simulate(MarketMakingConfig(n_steps=50, n_paths=7, seed=8))
    assert res.pnl.shape == (7, 50)
    assert res.inventory.shape == (7, 50)
    assert res.terminal_pnl.shape == (7,)
