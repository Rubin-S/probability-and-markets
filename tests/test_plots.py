from __future__ import annotations

from pathlib import Path

from probability_and_markets.market_making import MarketMakingConfig, simulate
from probability_and_markets.plots import write_market_making_plots


def test_plots_write_three_pngs(tmp_path: Path) -> None:
    cfg = dict(n_steps=80, n_paths=12, seed=9)
    paths = write_market_making_plots(
        outdir=tmp_path,
        baseline=simulate(MarketMakingConfig(**cfg)),
        no_informed=simulate(MarketMakingConfig(informed_prob=0.0, **cfg)),
        tight=simulate(MarketMakingConfig(half_spread=0.02, informed_prob=0.5, **cfg)),
        no_skew=simulate(MarketMakingConfig(inventory_skew=0.0, **cfg)),
    )
    assert [p.name for p in paths] == ["pnl_paths.png", "inventory.png", "terminal_pnl_hist.png"]
    assert all(p.stat().st_size > 0 for p in paths)
