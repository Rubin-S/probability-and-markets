"""Monte Carlo counterparts of the closed-form problems. Seeds are required."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def two_dice_sum_frequency(total: int, *, n_trials: int, rng: np.random.Generator) -> float:
    rolls = rng.integers(1, 7, size=(n_trials, 2))
    return float(np.mean(rolls.sum(axis=1) == total))


def two_dice_max_mean(*, n_trials: int, rng: np.random.Generator) -> float:
    rolls = rng.integers(1, 7, size=(n_trials, 2))
    return float(np.mean(rolls.max(axis=1)))


def hypergeometric_frequency(
    *,
    n_success_states: int,
    population: int,
    draws: int,
    observed: int,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """Simulate without-replacement draws; return frequency of ``observed`` successes."""
    ngood = n_success_states
    nbad = population - n_success_states
    samples = rng.hypergeometric(ngood, nbad, draws, size=n_trials)
    return float(np.mean(samples == observed))


def disease_test_positive_posterior_frequency(
    *,
    prior: float,
    sensitivity: float,
    specificity: float,
    n_people: int,
    rng: np.random.Generator,
) -> float:
    """Simulate a population; return the empirical P(disease | positive)."""
    diseased = rng.random(n_people) < prior
    # Positive if diseased & sensitive, or healthy & false positive.
    false_positive = 1.0 - specificity
    test_positive = np.where(
        diseased,
        rng.random(n_people) < sensitivity,
        rng.random(n_people) < false_positive,
    )
    n_pos = int(test_positive.sum())
    if n_pos == 0:
        raise RuntimeError("no positive tests in sample; increase n_people")
    return float(diseased[test_positive].mean())


@dataclass(frozen=True)
class StoppedWalk:
    """Finite-horizon sample of a fair ±1 walk stopped at +1 (or at ``max_steps``)."""

    positions: np.ndarray
    times: np.ndarray
    hit_plus_one: np.ndarray

    @property
    def mean_position(self) -> float:
        return float(self.positions.mean())

    @property
    def hit_rate(self) -> float:
        return float(self.hit_plus_one.mean())


def fair_walk_stop_at_plus_one(
    *,
    n_paths: int,
    max_steps: int,
    rng: np.random.Generator,
) -> StoppedWalk:
    """Simulate S_{n ∧ τ} with τ = first time S = +1, starting at 0.

    For every finite ``max_steps``, E[S_{n ∧ τ}] = 0 (bounded stopping time).
    Almost every path eventually hits +1, but the rare deep-negative paths
    keep the mean at 0. That is the optional-stopping caveat.
    """
    steps = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_paths, max_steps))
    walks = np.cumsum(steps, axis=1)
    hit = walks == 1
    ever = hit.any(axis=1)
    first = np.where(ever, hit.argmax(axis=1), max_steps - 1)
    positions = walks[np.arange(n_paths), first]
    # Paths that never hit +1 sit at S_{max_steps}, already selected.
    times = first + 1
    times = np.where(ever, times, max_steps)
    return StoppedWalk(positions=positions, times=times, hit_plus_one=ever)
