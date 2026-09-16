"""Classic, training-free stiction detectors (ellipse-fit + Kano cross-check).

See docs/ML_PLAN.md §3-4, §7 for rationale and how this fits the pipeline.
Zero training data, zero human labels -- every function here judges a single
window's PV/OP signal shape on its own, using process-control theory instead
of learned parameters.

Both shape-detector functions (ellipse_stiction_index, kano_pattern_check)
expect pv/op already per-window z-score normalized (ML_PLAN.md §6), so their
outputs are unit-agnostic across sensor types by construction, not because
either function does any scaling itself.

Known edge case, found during sanity-testing: per-window z-score
normalization divides by that window's own std, so a window where OP barely
moved at all (valve not actively being driven) gets its tiny noise rescaled
up to unit variance -- amplified noise can then look like structure to the
shape detectors. has_sufficient_activity() guards against this: it compares
a window's raw (pre-normalization) OP std against a reference std (e.g. the
full file's OP std) computed by the caller, since "how much is enough
activity" isn't answerable from one window in isolation and can't use an
absolute threshold across sensor types with different physical units. This
is a caller-side pre-filter, not baked into derive_label, since it needs
file-level context that a pure per-window function shouldn't have to take.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def has_sufficient_activity(
    op_raw: np.ndarray, reference_op_std: float, min_relative_activity: float = 0.15
) -> bool:
    """Whether a window's raw OP signal moved enough to judge stick-slip on.

    reference_op_std should be computed by the caller from a wider context
    than one window (e.g. the full source file's OP std) -- see module
    docstring.
    """
    if reference_op_std <= 0:
        return True  # no reference to compare against; don't filter
    return bool(float(np.std(op_raw)) >= min_relative_activity * reference_op_std)


def ellipse_stiction_index(
    pv: np.ndarray,
    op: np.ndarray,
    band_frac: float = 0.1,
    min_band_points: int = 4,
) -> float:
    """Estimate 'apparent stiction' as the OP-axis width of the PV-OP point
    cloud, measured in a horizontal band around PV's center.

    This approximates He et al.'s ellipse-fitting method (the width of a
    fitted ellipse along the OP axis, at the ellipse's center) but measures
    the point cloud's actual width directly instead of fitting a parametric
    ellipse. That's deliberate: the literature itself notes ellipse fitting
    struggles when the true shape is closer to a parallelogram, which is
    what a stick-slip limit cycle often looks like. Measuring the point
    cloud directly sidesteps that failure mode and is more numerically
    stable on small, noisy windows than algebraic conic fitting.
    """
    if len(pv) < min_band_points:
        raise ValueError("window too short for a band-width estimate")

    center_pv = float(np.median(pv))
    pv_std = float(np.std(pv))
    if pv_std == 0:
        return 0.0  # perfectly flat PV: no spread to measure a width from

    band_half_width = band_frac * pv_std
    in_band = np.abs(pv - center_pv) <= band_half_width
    for _ in range(6):  # widen the band until it has enough points
        if in_band.sum() >= min_band_points or band_half_width >= pv_std * 3:
            break
        band_half_width *= 2
        in_band = np.abs(pv - center_pv) <= band_half_width

    if in_band.sum() < 2:
        return 0.0

    return float(op[in_band].max() - op[in_band].min())


def is_likely_quantized(pv: np.ndarray, unique_fraction_threshold: float = 0.2) -> bool:
    """Whether pv looks like it's been through coarse quantization rather
    than a continuous physical measurement.

    Found while validating kano_pattern_check against real SACAC data: a
    quantized PV signal (sensor/DAC resolution limit) produces the exact
    same "flat, then jump" step shape as real stick-slip -- OP moves
    continuously while PV sits at a level then jumps to the next one. What
    tells them apart isn't the step shape, it's that quantization revisits
    a small fixed set of levels over and over, while a genuinely stuck
    valve settles at whatever level friction happened to catch it at --
    effectively continuous, rarely exactly repeated. Verified against SACAC's
    own literature-tagged files: real stick-slip windows had 46-77% unique
    PV values; a `quantisation-*` file had 6%.
    """
    if len(pv) == 0:
        return False
    n_unique = len(np.unique(np.round(pv, 6)))
    return (n_unique / len(pv)) < unique_fraction_threshold


def kano_pattern_check(
    pv: np.ndarray,
    op: np.ndarray,
    op_move_threshold: float = 0.3,
    pv_flat_threshold: float = 0.15,
    min_flat_fraction: float = 0.3,
) -> bool:
    """Heuristic check for Kano's characteristic 'stick, then jump' pattern:
    a stroke where OP moves consistently in one direction while PV stays
    flat for an initial portion, then jumps.

    This is a simplified proxy for Kano's original qualitative pattern
    taxonomy, not a literal reimplementation of it -- it captures the core
    stick-slip signature (OP moving, PV mostly not responding, punctuated
    by sudden catch-up jumps) that actually distinguishes stiction from a
    smoothly-varying loop (e.g. one oscillating from aggressive tuning or
    an external disturbance, which produces continuous PV movement with no
    flat/stuck periods), without reproducing the full pattern
    classification.

    Note this is deliberately a *stricter* signature than "any width in the
    OP-PV plot" (which is what ellipse_stiction_index alone measures, and
    which a lagged-but-healthy oscillation can also produce) -- it requires
    most of a stroke's PV movement to be near-zero, not just some.

    Explicitly excludes quantized-looking signals (see
    is_likely_quantized) -- coarse quantization produces the same
    "flat, then jump" step shape as real stick-slip for an entirely
    different reason (sensor/DAC resolution, not valve friction), found
    while validating this function against SACAC's own
    literature-tagged `quantisation-*` files.

    Returns True if a majority of OP-direction "strokes" in the window show
    the stick-then-jump signature.
    """
    if is_likely_quantized(pv):
        return False

    op_diff = np.diff(op)
    pv_diff = np.diff(pv)

    signs = np.sign(op_diff)
    change_points = np.where(np.diff(signs) != 0)[0] + 1
    strokes = np.split(np.arange(len(op_diff)), change_points)

    hits = 0
    total = 0
    for stroke in strokes:
        if len(stroke) < 3:
            continue
        op_range = op[stroke[-1] + 1] - op[stroke[0]]
        if abs(op_range) < op_move_threshold:
            continue  # not enough OP movement to judge stick-slip on
        total += 1

        pv_stroke_diff = np.abs(pv_diff[stroke])
        flat_mask = pv_stroke_diff < pv_flat_threshold

        # Most of the stroke's PV steps near-zero (stuck), with at least
        # one much larger step somewhere in it (slip) -- catches repeating
        # stick-jump cycles within a single OP stroke, not just one
        # leading flat run.
        flat_fraction = float(flat_mask.mean())
        has_jump = pv_stroke_diff.max() > 3 * pv_flat_threshold

        if flat_fraction >= min_flat_fraction and has_jump:
            hits += 1

    if total == 0:
        return False
    return (hits / total) >= 0.5


def derive_label(
    pv: np.ndarray,
    op: np.ndarray,
    ellipse_threshold: float,
) -> Literal["yes", "no", "uncertain"]:
    """Combine both detectors into a per-window label.

    Windows where the two detectors disagree are "uncertain" and get
    dropped from RF training data -- never imputed, never guessed. See
    ML_PLAN.md §7.
    """
    ellipse_verdict = ellipse_stiction_index(pv, op) >= ellipse_threshold
    kano_verdict = kano_pattern_check(pv, op)

    if ellipse_verdict and kano_verdict:
        return "yes"
    if not ellipse_verdict and not kano_verdict:
        return "no"
    return "uncertain"
