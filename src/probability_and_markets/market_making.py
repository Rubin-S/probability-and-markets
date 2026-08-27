"""Toy market-making simulator: quotes, inventory, adverse selection, PnL.

This is a teaching model, not a market. There is no order book, no real data,
and no claimed edge. The point is to see *why* a spread can still lose money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketMakingConfig:
    n_steps: int = 2_000
    n_paths: int = 200
    start_value: float = 100.0
    value_vol: float = 0.15
    """Std of the Gaussian innovation to true value each step (news)."""
    half_spread: float = 0.10
    informed_prob: float = 0.35
    """Probability the arriving trader is informed (knows V_t). Remainder are noise."""
    inventory_skew: float = 0.02
    """Quote shift per unit of inventory: long inventory → lower quotes."""
    max_inventory: int = 25
    seed: int = 0


@dataclass
class SimulationResult:
    pnl: np.ndarray  # (n_paths, n_steps)
    inventory: np.ndarray
    value: np.ndarray
    cash: np.ndarray
    n_informed_fills: np.ndarray
    n_noise_fills: np.ndarray

    @property
    def terminal_pnl(self) -> np.ndarray:
        return self.pnl[:, -1]

    @property
    def mean_terminal_pnl(self) -> float:
        return float(self.terminal_pnl.mean())

    @property
    def std_terminal_pnl(self) -> float:
        return float(self.terminal_pnl.std(ddof=1))


def _quotes(
    mid: float, inventory: np.ndarray, cfg: MarketMakingConfig
) -> tuple[np.ndarray, np.ndarray]:
    skew = cfg.inventory_skew * inventory
    bid = mid - cfg.half_spread - skew
    ask = mid + cfg.half_spread - skew
    return bid, ask


def simulate(cfg: MarketMakingConfig | None = None) -> SimulationResult:
    """Run ``n_paths`` independent MM sessions.

    Timing each step
    ----------------
    1. True value jumps: V_t = V_{t-1} + σ ξ_t. The MM does **not** see this
       before quoting; quotes are centered on V_{t-1} (lagged fair value), then
       skewed for inventory.
    2. One trader arrives. Informed traders buy iff V_t > ask and sell iff
       V_t < bid (they pick off stale quotes). Noise traders buy or sell 50/50
       as long as the fill would not breach ``max_inventory``.
    3. Mark-to-market PnL = cash + inventory * V_t.

    So the MM earns the half-spread on noise, and *pays* adverse selection on
    informed flow whenever news has moved through the quotes.
    """
    cfg = cfg or MarketMakingConfig()
    rng = np.random.default_rng(cfg.seed)
    p, t = cfg.n_paths, cfg.n_steps

    value = np.empty((p, t), dtype=np.float64)
    pnl = np.empty((p, t), dtype=np.float64)
    inventory = np.zeros((p, t), dtype=np.int32)
    cash = np.zeros((p, t), dtype=np.float64)
    n_informed = np.zeros(p, dtype=np.int32)
    n_noise = np.zeros(p, dtype=np.int32)

    v = np.full(p, cfg.start_value, dtype=np.float64)
    q = np.zeros(p, dtype=np.int32)
    c = np.zeros(p, dtype=np.float64)
    v_prev = v.copy()

    innovations = rng.normal(0.0, cfg.value_vol, size=(p, t))
    informed_mask = rng.random((p, t)) < cfg.informed_prob
    noise_side = rng.random((p, t)) < 0.5  # True → noise buys (hits ask)

    for i in range(t):
        v = v_prev + innovations[:, i]
        bid, ask = _quotes(v_prev, q.astype(np.float64), cfg)

        informed = informed_mask[:, i]
        buy = np.zeros(p, dtype=bool)
        sell = np.zeros(p, dtype=bool)

        buy |= informed & (v > ask)
        sell |= informed & (v < bid)
        # If V is inside the quote, informed does not trade.
        noise = ~informed
        buy |= noise & noise_side[:, i]
        sell |= noise & ~noise_side[:, i]

        would_short = q - 1 < -cfg.max_inventory
        would_long = q + 1 > cfg.max_inventory
        buy &= ~would_short
        sell &= ~would_long

        did_informed = informed & (buy | sell)
        did_noise = noise & (buy | sell)
        n_informed += did_informed.astype(np.int32)
        n_noise += did_noise.astype(np.int32)

        c = c + ask * buy.astype(np.float64) - bid * sell.astype(np.float64)
        q = q - buy.astype(np.int32) + sell.astype(np.int32)

        value[:, i] = v
        inventory[:, i] = q
        cash[:, i] = c
        pnl[:, i] = c + q.astype(np.float64) * v
        v_prev = v

    return SimulationResult(
        pnl=pnl,
        inventory=inventory,
        value=value,
        cash=cash,
        n_informed_fills=n_informed,
        n_noise_fills=n_noise,
    )
