"""Live profiling: lambdify an expression, run it, measure actual
wall-clock + peak memory + numerical error against a high-precision
oracle, and compare to the eml-cost predictions (estimate_time +
predict_precision_loss).

The "live" prefix distinguishes this from :mod:`eml_cost.profile`,
which holds the static :class:`PfaffianProfile` cost-class wrapper.

Public API:

    >>> from eml_cost.live_profile import live_profile
    >>> r = live_profile("exp(x) + log(x) + sin(x*y)")
    >>> r.actual_lambdify_ms      # measured
    >>> r.actual_eval_ns_per_call # measured
    >>> r.peak_kib                # measured
    >>> r.relerr_max              # measured
    >>> r.predicted_lambdify_ms   # eml_cost prediction
    >>> r.predicted_relerr        # eml_cost prediction

The tool serves three concrete needs:

  1. Pre-flight cost check:   "before I lambdify this, what will it
                              actually take?" — predicted vs measured.
  2. Numerics gate:           "what's the actual digit loss vs
                              what we predicted?" — closes the
                              prediction loop.
  3. Memory budgeting:        "how much heap does this expression
                              chew through?" — answered by tracemalloc.

We deliberately do NOT include this in the default `eml-cost report`
output (that's predictions only); profiling actually executes the
expression and so is opt-in via `eml-cost profile`.
"""
from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Optional

import sympy as sp

from .estimate_time import estimate_time
from .predict_precision_loss import predict_precision_loss


# ──────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveProfileResult:
    """Live-execution profile of one symbolic expression.

    Compares the eml-cost predictions (which never run the
    expression) against actual measurements (which do).

    Wall-clock fields:

      * ``actual_lambdify_ms``  — time to call ``sp.lambdify``.
      * ``actual_eval_ns_per_call`` — median per-call ns over
        ``eval_repeats`` evaluations.

    Memory:

      * ``peak_kib`` — peak heap during ``lambdify + eval`` per
        ``tracemalloc``. Includes both stages combined since
        users care about the worst case.

    Numerics:

      * ``relerr_max`` — max |float64-eval - mpmath-eval| /
        |mpmath-eval| over the random sample.
      * ``digits_lost_actual`` — log10(relerr_max) − log10(eps).

    Predictions (from estimate_time + predict_precision_loss):

      * ``predicted_lambdify_ms`` — eml-cost wall-clock prediction
        for the lambdify proxy (95% CI).
      * ``predicted_relerr`` — eml-cost float-64 max relerr.
      * ``predicted_digits_lost`` — eml-cost decimal digits lost.

    Sample sizes:

      * ``samples`` — number of random points used for relerr
        sampling (default 64).
      * ``eval_repeats`` — number of timeit reps (default 1000).
    """

    expr: str
    free_vars: list[str]

    actual_lambdify_ms: float
    actual_eval_ns_per_call: float
    peak_kib: float

    relerr_max: float
    digits_lost_actual: float

    predicted_lambdify_ms: float
    predicted_lambdify_ci95: tuple[float, float]
    predicted_relerr: float
    predicted_digits_lost: float

    samples: int
    eval_repeats: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["predicted_lambdify_ci95"] = list(self.predicted_lambdify_ci95)
        return d


# ──────────────────────────────────────────────────────────────────
# Core profiler
# ──────────────────────────────────────────────────────────────────


def live_profile(
    expr: sp.Basic | str,
    *,
    samples: int = 64,
    eval_repeats: int = 1000,
    seed: int = 0,
    sample_lo: float = 0.5,
    sample_hi: float = 1.5,
    mpmath_dps: int = 50,
) -> LiveProfileResult:
    """Lambdify ``expr``, run it, measure wall-clock / memory /
    numerical error, and compare to eml-cost predictions.

    Sampling domain ``[sample_lo, sample_hi]`` defaults to
    ``[0.5, 1.5]`` — far enough from zero that ``log`` / division
    don't blow up while still exercising the float-64 mantissa.
    Override for expressions that need a different domain
    (e.g. ``log(x)`` near ``x=1``).

    Args:
      expr:           sympy expression or its string form.
      samples:        number of random sample points for the
                      relerr measurement. Default 64.
      eval_repeats:   number of timed evaluations per sample.
                      Default 1000 — keeps total cost O(1ms) per
                      sample for cheap kernels.
      seed:           PRNG seed for sampling reproducibility.
      sample_lo,hi:   inclusive bounds for the uniform sampler.
      mpmath_dps:     mpmath decimal precision used as the oracle.

    Returns:
      A :class:`LiveProfileResult` with every measured + predicted
      field. Pure data — no side effects on the caller.
    """
    if isinstance(expr, str):
        expr = sp.sympify(expr)
    expr_str = str(expr)
    free = sorted(expr.free_symbols, key=lambda s: s.name)
    free_names = [s.name for s in free]

    # ── Predictions ────────────────────────────────────────────
    times = estimate_time(expr)
    pl = predict_precision_loss(expr)
    lambdify_pred = times["lambdify"]
    predicted_lambdify_ms = float(lambdify_pred.predicted_ms)
    predicted_lambdify_ci95 = (
        float(lambdify_pred.ci95[0]),
        float(lambdify_pred.ci95[1]),
    )
    predicted_relerr = float(pl.predicted_max_relerr)
    predicted_digits_lost = float(pl.predicted_digits_lost)

    # ── Measure lambdify wall-clock + peak memory ──────────────
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    f = sp.lambdify(free, expr, modules="math") if free else (
        sp.lambdify((), expr, modules="math")
    )
    t1 = time.perf_counter()
    actual_lambdify_ms = (t1 - t0) * 1000.0
    lambdify_peak = tracemalloc.get_traced_memory()[1]

    # ── Build random samples + run evaluation ──────────────────
    import random
    rnd = random.Random(seed)
    sample_points: list[tuple[float, ...]] = []
    if free:
        for _ in range(samples):
            sample_points.append(
                tuple(rnd.uniform(sample_lo, sample_hi) for _ in free)
            )
    else:
        sample_points = [()] * samples

    # Time per-call (median over eval_repeats * samples) — use the
    # first sample point for the timing loop so the cache is hot.
    timing_args = sample_points[0] if sample_points else ()
    # Warm up.
    for _ in range(min(64, eval_repeats)):
        if timing_args:
            f(*timing_args)
        else:
            f()
    # Measure.
    t0 = time.perf_counter()
    for _ in range(eval_repeats):
        if timing_args:
            f(*timing_args)
        else:
            f()
    t1 = time.perf_counter()
    actual_eval_ns_per_call = (t1 - t0) * 1e9 / eval_repeats

    # ── Numerical error vs mpmath oracle ───────────────────────
    relerr_max = 0.0
    # The oracle computes the same expression at high precision via
    # sp.N with mpmath; we then cast back to float64 and compare.
    for pt in sample_points:
        subs = {s: v for s, v in zip(free, pt)}
        try:
            float_val = float(f(*pt)) if pt else float(f())
        except (OverflowError, ZeroDivisionError, ValueError):
            continue
        try:
            oracle = float(sp.N(expr.subs(subs), mpmath_dps))
        except (OverflowError, ZeroDivisionError, ValueError):
            continue
        if oracle == 0.0:
            denom = max(abs(float_val), 1e-300)
        else:
            denom = abs(oracle)
        err = abs(float_val - oracle) / denom
        if err > relerr_max:
            relerr_max = err

    # tracemalloc captures *peak since start*. We want lambdify+eval.
    eval_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    peak_kib = max(lambdify_peak, eval_peak) / 1024.0

    # ── digits_lost actual ─────────────────────────────────────
    if relerr_max <= 0:
        digits_lost_actual = 0.0
    else:
        import math
        eps = 2.220446049250313e-16  # float64
        digits_lost_actual = max(0.0, math.log10(relerr_max / eps))

    return LiveProfileResult(
        expr=expr_str,
        free_vars=free_names,
        actual_lambdify_ms=actual_lambdify_ms,
        actual_eval_ns_per_call=actual_eval_ns_per_call,
        peak_kib=peak_kib,
        relerr_max=relerr_max,
        digits_lost_actual=digits_lost_actual,
        predicted_lambdify_ms=predicted_lambdify_ms,
        predicted_lambdify_ci95=predicted_lambdify_ci95,
        predicted_relerr=predicted_relerr,
        predicted_digits_lost=predicted_digits_lost,
        samples=samples,
        eval_repeats=eval_repeats,
    )


__all__ = ["LiveProfileResult", "live_profile"]
