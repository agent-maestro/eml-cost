"""analyze_batch — process many expressions with caching + multiprocessing.

Created in E-184 (2026-04-26).

Public API:

    from eml_cost import analyze_batch

    profiles = analyze_batch(
        [sp.exp(x), sp.sin(x), x**2 + 3*x + 1],
        canonicalize=True,
        n_jobs=4,
        cache=True,
        progress=True,
    )
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import sympy as sp

from .canonicalize import canonicalize as _canonicalize
from .profile import PfaffianProfile

# tqdm is optional — falls back to no-op if absent
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

# joblib is optional — falls back to serial if absent or n_jobs == 1
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False


def _profile_one(expr_or_str: Any, do_canonicalize: bool) -> PfaffianProfile:
    """Worker — must be picklable, hence module-level."""
    try:
        return PfaffianProfile.from_expression(expr_or_str,
                                                do_canonicalize=do_canonicalize)
    except Exception as e:
        # Sentinel error profile — keep batch order intact
        return PfaffianProfile(
            r=-1, degree=-1, width=-1,
            cost_class="ERROR",
            oscillatory=False,
            corrections={"c_osc": 0, "c_composite": 0, "delta_fused": 0,
                          "error": str(e)[:200]},
            expression=str(expr_or_str)[:200],
            canonical_form="",
            is_pfaffian_not_eml=False,
        )


def analyze_batch(
    expressions: Iterable[Any],
    *,
    canonicalize: bool = True,
    n_jobs: int = 1,
    cache: bool = True,
    progress: Optional[bool] = None,
    cache_max_size: int = 100_000,
) -> list[PfaffianProfile]:
    """Profile many expressions in one call.

    Parameters
    ----------
    expressions : iterable of sympy.Basic or str
        Expressions to profile.
    canonicalize : bool
        Apply :func:`canonicalize` before profiling. Default ``True``.
    n_jobs : int
        Number of worker processes (joblib). Default ``1`` = serial.
        Use ``-1`` for all CPUs. SymPy is not thread-safe — we use
        processes, never threads.
    cache : bool
        Cache profiles keyed by the **canonical form string**, so
        algebraically-equivalent expressions share a profile. Default
        ``True``.
    progress : bool or None
        Show a tqdm progress bar. Default ``None`` = auto (on for
        len(expressions) > 100).
    cache_max_size : int
        Max number of cached entries before LRU eviction. Default
        100_000.

    Returns
    -------
    list[PfaffianProfile]
        Same order as input. Failed expressions get a sentinel profile
        with ``cost_class == "ERROR"``.

    Notes
    -----
    For ``n_jobs > 1`` the cache is per-batch only (not shared across
    workers) — each process builds its own cache. For maximum cache
    benefit use ``n_jobs=1`` on workloads with many duplicates.
    """
    exprs = list(expressions)
    n = len(exprs)
    if n == 0:
        return []

    show_progress = progress if progress is not None else (n > 100 and _HAVE_TQDM)
    iterator = tqdm(exprs, desc="analyze_batch") if show_progress and _HAVE_TQDM else exprs

    # ------------------------------------------------------------------
    # Path 1: serial with cache (most common, max cache benefit)
    #
    # 0.5.1 two-level cache fix:
    #   Level 1 (raw_cache): keyed by str(input), fast path for exact
    #                         repeats — no canonicalize needed.
    #   Level 2 (canon_cache): keyed by canonical-form string, catches
    #                          algebraically-equivalent forms.
    #
    # On Level-1 hit: return immediately (zero canonicalize cost).
    # On Level-1 miss + Level-2 hit: write to Level-1 too, return.
    # On both miss: compute profile, write to BOTH levels.
    #
    # Old 0.5.0 path ran canonicalize on every call (even cache hits),
    # producing the 0.58x slowdown. This fix makes cache hits ~free.
    # ------------------------------------------------------------------
    if n_jobs == 1:
        from dataclasses import replace
        raw_cache: dict[str, PfaffianProfile] = {}
        canon_cache: dict[str, PfaffianProfile] = {}
        results: list[PfaffianProfile] = []
        for e in iterator:
            if cache:
                # Level 1: exact-string fast path
                raw_key = str(e)
                if raw_key in raw_cache:
                    results.append(raw_cache[raw_key])
                    continue
                # Need to compute canonical form to check Level 2
                try:
                    parsed = sp.sympify(e)
                    canon = _canonicalize(parsed) if canonicalize else parsed
                    canon_key = str(canon)
                except Exception:
                    results.append(_profile_one(e, canonicalize))
                    continue
                if canon_key in canon_cache:
                    p = canon_cache[canon_key]
                    if len(raw_cache) < cache_max_size:
                        raw_cache[raw_key] = p
                    results.append(p)
                    continue
                # Both caches missed — build profile from canonical form
                try:
                    p = PfaffianProfile.from_expression(canon, do_canonicalize=False)
                    p = replace(p, expression=raw_key, canonical_form=canon_key)
                except Exception:
                    p = _profile_one(e, canonicalize)
                if len(canon_cache) < cache_max_size:
                    canon_cache[canon_key] = p
                if len(raw_cache) < cache_max_size:
                    raw_cache[raw_key] = p
                results.append(p)
            else:
                results.append(_profile_one(e, canonicalize))
        return results

    # ------------------------------------------------------------------
    # Path 2: parallel via joblib (no cross-worker cache)
    # ------------------------------------------------------------------
    if not _HAVE_JOBLIB:
        # Graceful fallback to serial
        return analyze_batch(exprs, canonicalize=canonicalize, n_jobs=1,
                              cache=cache, progress=progress,
                              cache_max_size=cache_max_size)

    backend = "loky"  # process-based
    results = Parallel(n_jobs=n_jobs, backend=backend, batch_size="auto")(
        delayed(_profile_one)(e, canonicalize) for e in iterator
    )
    return list(results)


def cache_hit_analysis(expressions: Iterable[Any], canonicalize: bool = True) -> dict[str, Any]:
    """Analyze the theoretical cache hit rate of a workload.

    Returns the unique-canonical-form count and theoretical hit rate.
    """
    exprs = list(expressions)
    n = len(exprs)
    if n == 0:
        return {"n": 0, "n_unique_canonical": 0, "theoretical_hit_rate": 0.0}

    canonical_forms = set()
    for e in exprs:
        try:
            ex = sp.sympify(e)
            if canonicalize:
                ex = _canonicalize(ex)
            canonical_forms.add(str(ex))
        except Exception:
            pass

    n_unique = len(canonical_forms)
    return {
        "n": n,
        "n_unique_canonical": n_unique,
        "theoretical_hit_rate": float(1 - n_unique / n) if n > 0 else 0.0,
    }
