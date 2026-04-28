"""Estimate dynamics signature from a (x, y) sample.

Given a 1-D regression target sampled on an ordered grid, infer
how many oscillation modes and how many decay modes are present.
The output is a coarse two-feature signature

    (n_oscillations, n_decays)

that matches what :func:`eml_cost.regularize` expects in
:attr:`RegularizerConfig.expected_dynamics`.

Method
------

  - **Oscillations:** real FFT, retain frequency bins whose power
    is at least ``min_power_ratio`` of the peak. Group nearby
    bins into a single mode (a 3-bin window). Static / DC bin is
    excluded.
  - **Decays:** Hilbert-transform envelope. If the envelope is
    monotone (within tolerance) and decreasing, count one decay
    mode. If the residual after a single-exponential fit is
    itself near-monotone, count a second mode. Cap at
    ``max_decays``.
  - **Static:** report ``n_static = 1`` when the FFT power is
    overwhelmingly DC (no detectable oscillation, no decay).

Honest framing
--------------

This is a **structural estimator** — the output is a two-integer
signature suitable for steering a symbolic-regression search.
It does not return frequencies or rates with a confidence
interval; the ``confidence`` attribute is a coarse 0..1 score
based on FFT peak prominence and envelope monotonicity.

Edge cases:

  - Samples shorter than ``min_samples`` (default 32) return
    ``DataDynamics(0, 0, 1, ...)`` with confidence 0.
  - Constant inputs return ``DataDynamics(0, 0, 1, ...)``.
  - Pure noise tends to register as 0 oscillation / 0 decay
    because the FFT power is near-flat (no peak above ratio).

Usage
-----

    >>> import numpy as np
    >>> from eml_cost import estimate_dynamics
    >>> x = np.linspace(0, 10, 256)
    >>> y = np.sin(3 * x) * np.exp(-0.2 * x)
    >>> d = estimate_dynamics(x, y)
    >>> d.n_oscillations
    1
    >>> d.n_decays >= 1
    True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "estimate_dynamics requires numpy. "
        "Install with: pip install eml-cost[data]"
    ) from exc

try:
    from scipy.signal import hilbert
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "estimate_dynamics requires scipy. "
        "Install with: pip install eml-cost[data]"
    ) from exc


__all__ = [
    "DataDynamics",
    "estimate_dynamics",
]


@dataclass(frozen=True)
class DataDynamics:
    """Coarse oscillation / decay / static signature inferred from data.

    Attributes
    ----------
    n_oscillations:
        Number of distinct frequency modes detected (peaks above
        the noise floor in the FFT power spectrum).
    n_decays:
        Number of monotone decay components detected via Hilbert
        envelope analysis. Capped at ``max_decays`` (default 2).
    n_static:
        ``1`` when the signal is dominated by the DC component
        with neither oscillation nor decay; ``0`` otherwise.
    estimated_chain_order:
        ``2 * n_oscillations + n_decays`` — the dynamics counter
        prediction for the structural chain order needed to
        reproduce this signature.
    confidence:
        Coarse 0..1 score combining FFT peak prominence and
        envelope monotonicity. Below 0.3, treat the signature as
        weakly identified.
    frequencies:
        Detected angular frequencies (radians per unit x), one per
        oscillation mode, sorted by descending power.
    decay_rates:
        Per-mode estimated decay rates (units of 1/x). Empty if
        ``n_decays == 0``.
    is_monotonic:
        True if the raw signal (not the envelope) is monotone
        within the sampled range.
    is_periodic:
        True if the dominant FFT peak's relative power is high
        enough to call the signal periodic (peak ratio >= 0.4).
    is_bounded:
        True if ``max(|y|)`` is finite and finite difference is
        not blowing up across the sample.
    """

    n_oscillations: int
    n_decays: int
    n_static: int
    estimated_chain_order: int
    confidence: float
    frequencies: tuple[float, ...] = field(default_factory=tuple)
    decay_rates: tuple[float, ...] = field(default_factory=tuple)
    is_monotonic: bool = False
    is_periodic: bool = False
    is_bounded: bool = True


def _detect_oscillations(
    y: np.ndarray,
    dx: float,
    *,
    min_power_ratio: float = 0.10,
    min_peak_to_median: float = 20.0,
    max_modes: int = 4,
) -> tuple[list[float], float]:
    """Return ``(angular_frequencies, dominant_power_ratio)``.

    A bin is accepted when its power exceeds ``min_power_ratio *
    peak_power``. Adjacent bins (within +/-1 of an already-accepted
    peak) are merged into the same mode.

    A noise-floor gate: if the dominant peak's power is less than
    ``min_peak_to_median`` times the median bin power, treat the
    spectrum as noise and return zero modes. For a clean sinusoid
    this ratio is huge; for white noise it is on the order of 5.
    """
    n = len(y)
    if n < 8:
        return [], 0.0

    # Detrend (remove linear trend so a pure decay doesn't masquerade
    # as a low-frequency oscillation).
    idx = np.arange(n)
    coeffs = np.polyfit(idx, y, 1)
    detrended = y - (coeffs[0] * idx + coeffs[1])

    # Window to reduce spectral leakage.
    window = np.hanning(n)
    spec = np.fft.rfft(detrended * window)
    power = np.abs(spec) ** 2

    if len(power) <= 1:
        return [], 0.0

    # Drop DC bin from oscillation search.
    power_ac = power[1:]
    if power_ac.size == 0 or power_ac.max() == 0:
        return [], 0.0

    peak = power_ac.max()
    median_pow = float(np.median(power_ac))
    if median_pow > 0 and peak / median_pow < min_peak_to_median:
        # Spectrum is essentially flat → noise, no real modes.
        return [], 0.0

    threshold = min_power_ratio * peak

    # Accept bins above threshold, suppressing nearby duplicates so
    # one broad peak counts as one mode.
    accepted: list[int] = []
    order = np.argsort(-power_ac)
    for k in order:
        if power_ac[k] < threshold:
            break
        if any(abs(int(k) - a) <= 2 for a in accepted):
            continue
        accepted.append(int(k))
        if len(accepted) >= max_modes:
            break

    accepted.sort()

    # Convert FFT bin indices (offset by 1 since we dropped DC) to
    # angular frequencies.
    sample_rate = 1.0 / dx if dx > 0 else 1.0
    bin_to_omega = 2.0 * np.pi * sample_rate / n
    freqs = [(k + 1) * bin_to_omega for k in accepted]

    total_power = power_ac.sum()
    dominant_ratio = peak / total_power if total_power > 0 else 0.0

    return freqs, float(dominant_ratio)


def _fit_exponential_decay(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit y ~ A * exp(-k * (x - x0)) on the strictly positive
    envelope. Return (k, residual). When the fit is degenerate,
    return (0.0, y - mean(y))."""
    eps = 1e-12
    pos = np.where(y > eps, y, eps)
    log_y = np.log(pos)
    try:
        slope, intercept = np.polyfit(x, log_y, 1)
    except Exception:  # noqa: BLE001
        return 0.0, y - float(np.mean(y))
    fitted = np.exp(intercept) * np.exp(slope * x)
    return float(-slope), y - fitted


def _detect_decays(
    y: np.ndarray,
    x: np.ndarray,
    *,
    max_decays: int = 2,
    tol: float = 1e-3,
) -> tuple[int, list[float]]:
    """Detect monotone decay modes via Hilbert envelope.

    A decay mode is counted when the analytic-signal envelope is
    well-fit by a positive-rate exponential. Multi-mode detection
    is conservative: we re-run on the residual and accept a second
    mode only if the residual envelope is itself monotone-decaying.
    """
    if len(y) < 16:
        return 0, []

    rates: list[float] = []

    # Fast-path: a pure positive decay has no AC component, so the
    # demeaned-Hilbert envelope is unreliable. If the raw signal is
    # strictly positive and globally non-increasing, fit it directly.
    if (np.all(y > 0)
            and float(y[-1]) < float(y[0]) * 0.95
            and float(np.std(np.diff(y))) >= 0):
        dy_neg = np.diff(y)
        if bool(np.all(dy_neg <= 1e-9)):
            rate, _ = _fit_exponential_decay(y, x)
            if rate > 0:
                return 1, [rate]

    # Stage 1: raw envelope.
    analytic = hilbert(y - float(np.mean(y)))
    env = np.abs(analytic)
    if env.max() == 0:
        return 0, []

    # Detect monotone-ish decay: peak occurs early and last sample
    # is materially smaller.
    peak_idx = int(np.argmax(env))
    if peak_idx > len(env) // 2:
        # Peak is near the end → not a decaying signal.
        return 0, []

    if env[-1] >= env[peak_idx] * (1 - tol):
        # Envelope is essentially flat — no decay.
        return 0, []

    rate1, residual = _fit_exponential_decay(env[peak_idx:], x[peak_idx:])
    if rate1 > 0:
        rates.append(rate1)

    if len(rates) == 0 or max_decays == 1:
        return len(rates), rates

    # Stage 2: residual envelope.
    if len(residual) < 16:
        return len(rates), rates
    res_env = np.abs(hilbert(residual - float(np.mean(residual))))
    if res_env.max() == 0:
        return len(rates), rates
    res_peak = int(np.argmax(res_env))
    if res_peak > len(res_env) // 2:
        return len(rates), rates
    if res_env[-1] >= res_env[res_peak] * (1 - tol):
        return len(rates), rates
    rate2, _ = _fit_exponential_decay(
        res_env[res_peak:], x[peak_idx:][res_peak:])
    if rate2 > 0:
        rates.append(rate2)

    return len(rates), rates


def estimate_dynamics(
    x: Sequence[float],
    y: Sequence[float],
    *,
    min_power_ratio: float = 0.10,
    max_modes: int = 4,
    max_decays: int = 2,
    min_samples: int = 32,
) -> DataDynamics:
    """Infer ``(n_oscillations, n_decays, n_static)`` from sampled data.

    Parameters
    ----------
    x, y:
        Sampled coordinates and values. ``x`` should be ordered;
        the spacing is treated as uniform when it is.
    min_power_ratio:
        FFT peak retention threshold (fraction of the dominant
        peak's power).
    max_modes:
        Cap on detected oscillation modes.
    max_decays:
        Cap on detected decay modes.
    min_samples:
        Minimum number of samples required; below this the
        estimator returns a static-only signature with
        confidence 0.

    Returns
    -------
    DataDynamics
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)

    if arr_x.shape != arr_y.shape or arr_x.ndim != 1:
        raise ValueError(
            f"x and y must be 1-D and matching shape; got "
            f"{arr_x.shape} vs {arr_y.shape}")

    is_bounded = bool(np.all(np.isfinite(arr_y))) and bool(
        np.all(np.isfinite(arr_x)))

    n = len(arr_y)
    if n < min_samples or not is_bounded:
        return DataDynamics(
            n_oscillations=0,
            n_decays=0,
            n_static=1,
            estimated_chain_order=0,
            confidence=0.0,
            is_bounded=is_bounded,
        )

    # Constant signal → static.
    if float(np.std(arr_y)) < 1e-12:
        return DataDynamics(
            n_oscillations=0,
            n_decays=0,
            n_static=1,
            estimated_chain_order=0,
            confidence=1.0,
            is_monotonic=True,
            is_bounded=True,
        )

    diffs = np.diff(arr_x)
    dx = float(np.mean(diffs)) if diffs.size > 0 else 1.0

    freqs, dom_ratio = _detect_oscillations(
        arr_y, dx,
        min_power_ratio=min_power_ratio,
        max_modes=max_modes,
    )
    n_osc = len(freqs)

    n_decay, rates = _detect_decays(
        arr_y, arr_x, max_decays=max_decays)

    # Monotonicity and periodicity flags.
    dy = np.diff(arr_y)
    is_monotonic = bool(np.all(dy >= -1e-9)) or bool(np.all(dy <= 1e-9))
    is_periodic = dom_ratio >= 0.4

    n_static = 1 if (n_osc == 0 and n_decay == 0) else 0

    estimated_chain_order = 2 * n_osc + n_decay

    # Confidence: blend FFT peak prominence and decay confidence.
    osc_conf = min(1.0, dom_ratio * 2.0) if n_osc > 0 else 0.0
    dec_conf = 0.6 if n_decay > 0 else 0.0
    if n_osc == 0 and n_decay == 0 and n_static == 1:
        confidence = 0.5
    else:
        confidence = max(osc_conf, dec_conf, 0.3)
    confidence = float(min(1.0, confidence))

    return DataDynamics(
        n_oscillations=n_osc,
        n_decays=n_decay,
        n_static=n_static,
        estimated_chain_order=estimated_chain_order,
        confidence=confidence,
        frequencies=tuple(freqs),
        decay_rates=tuple(rates),
        is_monotonic=is_monotonic,
        is_periodic=is_periodic,
        is_bounded=is_bounded,
    )
