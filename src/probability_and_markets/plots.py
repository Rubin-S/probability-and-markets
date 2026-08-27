"""Matplotlib helpers. Figures are generated on demand; they are not market data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from probability_and_markets.market_making import SimulationResult


def write_market_making_plots(
    *,
    outdir: Path,
    baseline: SimulationResult,
    no_informed: SimulationResult,
    tight: SimulationResult,
    no_skew: SimulationResult,
) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(1, baseline.pnl.shape[1] + 1)
    for label, res, color in (
        ("noise only (α=0)", no_informed, "#2a9d8f"),
        ("baseline", baseline, "#264653"),
        ("tight spread, more informed", tight, "#e76f51"),
    ):
        mean = res.pnl.mean(axis=0)
        lo, hi = np.quantile(res.pnl, [0.1, 0.9], axis=0)
        ax.plot(steps, mean, color=color, label=label, lw=2)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.12)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("mark-to-market PnL")
    ax.set_title("Mean PnL path (band = 10th–90th percentile across paths)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p1 = outdir / "pnl_paths.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    written.append(p1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        steps,
        np.abs(no_skew.inventory).mean(axis=0),
        label="no inventory skew",
        color="#e76f51",
    )
    ax.plot(
        steps,
        np.abs(baseline.inventory).mean(axis=0),
        label="with inventory skew",
        color="#264653",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("mean |inventory|")
    ax.set_title("Inventory: skew leans quotes to mean-revert position")
    ax.legend(frameon=False)
    fig.tight_layout()
    p2 = outdir / "inventory.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    written.append(p2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(no_informed.terminal_pnl, bins=30, alpha=0.65, label="noise only", color="#2a9d8f")
    ax.hist(tight.terminal_pnl, bins=30, alpha=0.65, label="tight + informed", color="#e76f51")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("terminal PnL")
    ax.set_ylabel("paths")
    ax.set_title("Same toy, different edge: spread harvest vs adverse selection")
    ax.legend(frameon=False)
    fig.tight_layout()
    p3 = outdir / "terminal_pnl_hist.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    written.append(p3)

    return written
