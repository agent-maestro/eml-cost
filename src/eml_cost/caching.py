"""Memoization keyed on Pfaffian fingerprint of expression arguments.

Standard :func:`functools.lru_cache` keys on argument identity (or
hash + equality). For functions whose output depends only on the
*cost class* of a SymPy expression — not on specific symbol names —
this misses every distinct naming variant of an otherwise-identical
computation.

``@cache_by_fingerprint(maxsize=N)`` instead keys SymPy arguments on
the **axes portion** of their :func:`fingerprint` (the leading
``p…-d…-w…-c…`` block, ignoring the tail hash). Two expressions that
agree on Pfaffian profile share a cache slot:

    >>> import sympy as sp
    >>> from eml_cost import cache_by_fingerprint
    >>> @cache_by_fingerprint(maxsize=128)
    ... def slow(expr):
    ...     return sum(1 for _ in sp.preorder_traversal(expr))
    >>> x, y = sp.symbols("x y")
    >>> slow(sp.sin(x))   # MISS, computes
    2
    >>> slow(sp.sin(y))   # HIT — same Pfaffian axes

Non-SymPy positional and keyword arguments are hashed normally and
must be hashable; if any argument is unhashable the call bypasses
the cache entirely.

**Caveat.** The decorator cannot enforce that the wrapped function's
output depends only on cost class. If the function inspects symbol
names or specific values, hits will be wrong for the new argument.
Use only when the cost-class-only contract is explicit and intentional.
"""
from __future__ import annotations

import functools
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

import sympy as sp

from .analyze import fingerprint as _fingerprint


__all__ = ["cache_by_fingerprint", "fingerprint_axes", "FingerprintCacheInfo"]


_F = TypeVar("_F", bound=Callable[..., Any])

_AXES_RE = re.compile(r"^(p\d+-d\d+-w\d+-c-?\d+)-h[0-9a-f]+$")


@dataclass(frozen=True)
class FingerprintCacheInfo:
    """Snapshot of a fingerprint-keyed cache's runtime state.

    Attributes
    ----------
    hits, misses:
        Cumulative hit / miss counts since the last ``cache_clear()``.
    maxsize:
        Configured capacity (``None`` = unbounded).
    currsize:
        Number of entries currently held.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


def fingerprint_axes(expr: sp.Basic | str) -> str:
    """Return the cost-axes portion of :func:`fingerprint` (no tail hash).

    Two expressions returning the same value here have identical Pfaffian
    profile (``pfaffian_r``, ``max_path_r``, ``eml_depth``,
    correction sum) but may differ in tree shape or symbol names.
    """
    fp = _fingerprint(expr)
    m = _AXES_RE.match(fp)
    return m.group(1) if m else fp


def _normalize_arg(arg: Any) -> Any:
    if isinstance(arg, sp.Basic):
        try:
            return ("__fp_axes__", fingerprint_axes(arg))
        except Exception:   # pragma: no cover — fingerprint is robust
            return ("__fp_fallback__", sp.srepr(arg))
    return arg


def cache_by_fingerprint(
    *,
    maxsize: int | None = 128,
) -> Callable[[_F], _F]:
    """Memoize the decorated function with a fingerprint-axes-keyed LRU.

    Parameters
    ----------
    maxsize:
        LRU capacity. Default 128. Pass ``None`` for unbounded.
        A ``ValueError`` is raised at decoration time for non-positive
        finite values.

    The wrapper exposes ``.cache_info()`` (returns
    :class:`FingerprintCacheInfo`) and ``.cache_clear()`` for
    introspection / reset.
    """
    if maxsize is not None and maxsize <= 0:
        raise ValueError(f"maxsize must be positive or None, got {maxsize!r}")

    def decorator(fn: _F) -> _F:
        cache: OrderedDict[Any, Any] = OrderedDict()
        stats = {"hits": 0, "misses": 0}

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                key = (
                    tuple(_normalize_arg(a) for a in args),
                    tuple(sorted(
                        (k, _normalize_arg(v)) for k, v in kwargs.items()
                    )),
                )
                hash(key)
            except TypeError:
                # Unhashable argument — bypass cache transparently.
                return fn(*args, **kwargs)

            if key in cache:
                cache.move_to_end(key)
                stats["hits"] += 1
                return cache[key]

            stats["misses"] += 1
            result = fn(*args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_info() -> FingerprintCacheInfo:
            return FingerprintCacheInfo(
                hits=stats["hits"],
                misses=stats["misses"],
                maxsize=maxsize,
                currsize=len(cache),
            )

        def cache_clear() -> None:
            cache.clear()
            stats["hits"] = 0
            stats["misses"] = 0

        wrapper.cache_info = cache_info        # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear      # type: ignore[attr-defined]
        return cast(_F, wrapper)

    return decorator
