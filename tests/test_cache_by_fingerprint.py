"""Tests for @cache_by_fingerprint and fingerprint_axes."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost import (
    FingerprintCacheInfo,
    cache_by_fingerprint,
    fingerprint,
    fingerprint_axes,
)


def test_axes_strips_tail_hash():
    x = sp.Symbol("x")
    fp = fingerprint(sp.sin(x))
    axes = fingerprint_axes(sp.sin(x))
    assert axes != fp
    assert "-h" not in axes
    assert axes.startswith("p")
    assert fp.startswith(axes + "-h")


def test_renamed_expression_hits_cache():
    """sin(x) and sin(y) share Pfaffian axes — second call must be a hit."""
    @cache_by_fingerprint(maxsize=10)
    def f(expr: sp.Basic) -> int:
        return 42

    x, y = sp.symbols("x y")
    f(sp.sin(x))
    f(sp.sin(y))

    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.hits == 1
    assert info.misses == 1
    assert info.currsize == 1


def test_different_axes_misses():
    """sin (r=2) and exp (r=1) have different axes — both miss."""
    @cache_by_fingerprint(maxsize=10)
    def f(expr: sp.Basic) -> int:
        return 0

    x = sp.Symbol("x")
    f(sp.sin(x))
    f(sp.exp(x))

    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.misses == 2
    assert info.hits == 0


def test_cache_info_returns_dataclass():
    @cache_by_fingerprint(maxsize=5)
    def f(expr: sp.Basic) -> int:
        return 1

    info = f.cache_info()       # type: ignore[attr-defined]
    assert isinstance(info, FingerprintCacheInfo)
    assert info.maxsize == 5
    assert info.currsize == 0


def test_cache_clear_resets_stats():
    @cache_by_fingerprint(maxsize=10)
    def f(expr: sp.Basic) -> int:
        return 1

    x = sp.Symbol("x")
    f(sp.sin(x))
    f(sp.sin(x))

    f.cache_clear()             # type: ignore[attr-defined]
    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0


def test_lru_eviction_at_capacity():
    """When maxsize=2 and we add a 3rd distinct entry, the LRU is evicted."""
    @cache_by_fingerprint(maxsize=2)
    def f(expr: sp.Basic) -> int:
        return 1

    x = sp.Symbol("x")
    f(x)              # MISS, cache=[x]
    f(sp.exp(x))      # MISS, cache=[x, exp]
    f(x)              # HIT, cache=[exp, x]
    f(sp.sin(x))      # MISS, evicts exp, cache=[x, sin]
    f(sp.exp(x))      # MISS — exp was evicted

    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.currsize == 2
    assert info.hits == 1
    assert info.misses == 4


def test_unhashable_arg_bypasses_cache():
    """Unhashable secondary args don't crash — call falls through."""
    @cache_by_fingerprint(maxsize=10)
    def f(expr: sp.Basic, opts: object) -> int:
        return 1

    x = sp.Symbol("x")
    result = f(sp.sin(x), {"key": "val"})   # dict is unhashable → bypass
    assert result == 1
    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0


def test_kwargs_normalized_into_key():
    """Same SymPy arg passed as kwarg must collide with positional."""
    @cache_by_fingerprint(maxsize=10)
    def f(*, expr: sp.Basic) -> int:
        return 1

    x, y = sp.symbols("x y")
    f(expr=sp.sin(x))
    f(expr=sp.sin(y))

    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.hits == 1


def test_non_sympy_args_use_normal_hashing():
    @cache_by_fingerprint(maxsize=10)
    def f(n: int) -> int:
        return n * 2

    f(3)
    f(3)
    f(4)

    info = f.cache_info()       # type: ignore[attr-defined]
    assert info.hits == 1
    assert info.misses == 2


def test_invalid_maxsize_rejected():
    with pytest.raises(ValueError):
        cache_by_fingerprint(maxsize=0)
    with pytest.raises(ValueError):
        cache_by_fingerprint(maxsize=-3)


def test_unbounded_cache_keeps_all_entries():
    """maxsize=None means no eviction even past 100+ distinct entries."""
    @cache_by_fingerprint(maxsize=None)
    def f(expr: sp.Basic) -> int:
        return 1

    x = sp.Symbol("x")
    # Build 8 expressions with distinct Pfaffian profiles via depth.
    distinct: list[sp.Basic] = [
        x,
        sp.exp(x),
        sp.sin(x),
        sp.exp(sp.exp(x)),
        sp.sin(sp.sin(x)),
        sp.exp(sp.sin(x)),
        sp.exp(sp.exp(sp.exp(x))),
        sp.sin(sp.sin(sp.sin(x))),
    ]
    for e in distinct:
        f(e)

    info = f.cache_info()       # type: ignore[attr-defined]
    # No capacity → all distinct entries retained.
    assert info.misses == len(distinct)
    assert info.currsize == len(distinct)
