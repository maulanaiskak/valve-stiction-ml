"""Synthetic training-data augmentation.

Motivation (see docs/ML_PLAN.md's synthetic-augmentation entry and
valve-stiction-pipeline's STREAMING_EVALUATION.md): the RF model trained
only on ISDB/SACAC (real industrial recordings) was found to fail
completely -- AUC 0.079, worse than random -- when scored against
valve-stiction-simulator's synthetic signal in a live streaming
evaluation, despite scoring well (ROC-AUC 0.865) on real held-out SACAC
data. That's a train/serve distribution-shift failure, not a bug: the
model had simply never seen this family of signal shapes.

This module is the same triangle-wave-OP + stick-slip-valve model as
valve-stiction-simulator's domain/simulator.py, ported here (not imported
as a cross-repo dependency -- it's ~20 lines, and this repo has no other
reason to depend on the pipeline repo) and *randomized* across a wide
range of period/amplitude/stick-band/noise/center combinations, so
training sees a family of similar signals rather than one fixed config.
This is standard domain-randomization practice for closing a train/serve
gap: it is not the same as training and testing on identical data, and a
held-out synthetic set (different random ranges, see
scripts/build_synthetic_features.py) is kept separate from what's
trained on, exactly like SACAC is kept separate from ISDB.

Ground truth here is exact (the caller sets stiction_enabled), unlike
ISDB/SACAC where the classic detector had to serve as a pseudo-label
source in the absence of any true label -- see classic.py's module
docstring and ML_PLAN.md §2/§4 for why that workaround exists for real
data and isn't needed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    stiction_enabled: bool
    period_samples: int
    amplitude: float
    stick_band: float
    noise_std: float
    center: float
    seed: int


def _generate_stream(config: SyntheticConfig, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    valve_position = config.center
    pvs = np.empty(n_samples)
    ops = np.empty(n_samples)
    for n in range(n_samples):
        phase = (n % config.period_samples) / config.period_samples
        op = config.center + config.amplitude * (2 * abs(2 * phase - 1) - 1)
        op += rng.normal(0, config.noise_std)

        if config.stiction_enabled:
            if abs(op - valve_position) > config.stick_band:
                valve_position = op
        else:
            valve_position = op

        pvs[n] = valve_position + rng.normal(0, config.noise_std)
        ops[n] = op
    return pvs, ops


def generate_windows(
    config: SyntheticConfig, window_size: int, n_windows: int, settle: int = 50
) -> list[tuple[np.ndarray, np.ndarray]]:
    """n_windows non-overlapping windows from one config, discarding the
    first `settle` samples so the valve isn't still at its initial
    position (matches valve-stiction-simulator's own settling behavior)."""
    pv, op = _generate_stream(config, settle + window_size * n_windows)
    pv, op = pv[settle:], op[settle:]
    return [
        (pv[i * window_size : (i + 1) * window_size], op[i * window_size : (i + 1) * window_size])
        for i in range(n_windows)
    ]


def sample_configs(rng: np.random.Generator, n_configs: int) -> list[SyntheticConfig]:
    """Randomly sample n_configs parameter combinations across a broad
    range -- not the same as valve-stiction-simulator's one fixed default
    config (period=50, amplitude=20, stick_band=7, noise=0.4, center=50),
    which stays a distinct, unseen check (see build_synthetic_features.py).
    stiction_enabled is an independent 50/50 coin flip per config.

    period_samples caps at 80 (window_size is 100 throughout this
    project) so every generated window contains at least one full
    oscillation cycle. ML_PLAN.md §13 already found, on real ISDB data,
    that kano_pattern_check's majority-vote-over-strokes logic needs a
    complete cycle to work reliably -- letting synthetic periods exceed
    the window size would manufacture windows that are ambiguous by
    construction, not a harder-but-fair test.

    stick_band and noise_std are sampled as *fractions of amplitude*
    (not independently) so a stiction_enabled=True config reliably
    produces a visible stick-slip pattern instead of degenerating into
    "amplitude barely exceeds stick_band" (near-continuous slipping,
    doesn't look like stiction) or "noise swamps stick_band" (looks like
    noise, not sticking) -- found by inspection: independent uniform
    sampling of all five parameters let through many configs where the
    classic detector correctly disagreed with the injected
    stiction_enabled flag because the *signal itself* didn't actually
    exhibit distinguishable stiction, not because the detector was wrong.
    valve-stiction-simulator's one hand-picked default config
    (stick_band=7 is 35% of amplitude=20, noise=0.4 is 2% of amplitude)
    avoids this by construction; sampling relative to amplitude here
    keeps every generated config in the same well-behaved regime."""
    configs = []
    for _ in range(n_configs):
        amplitude = float(rng.uniform(10.0, 40.0))
        configs.append(
            SyntheticConfig(
                stiction_enabled=bool(rng.integers(0, 2)),
                period_samples=int(rng.integers(20, 80)),
                amplitude=amplitude,
                stick_band=float(amplitude * rng.uniform(0.25, 0.45)),
                noise_std=float(amplitude * rng.uniform(0.01, 0.05)),
                center=float(rng.uniform(20.0, 80.0)),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return configs
