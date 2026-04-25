"""Type-hint-style enforcement of cost ceilings on functions
returning SymPy expressions.

    >>> from eml_cost import costlimit
    >>> @costlimit(predicted_depth=3)
    ... def gradient(x):
    ...     return sp.exp(x) * sp.cos(x)   # cost <= 3 → OK
    >>> gradient(sp.Symbol("x"))
    exp(x)*cos(x)

When the function's return value exceeds the configured cost ceiling
on ANY of the three named axes (``predicted_depth``, ``max_path_r``,
``pfaffian_r``), the decorator raises :class:`CostLimitExceeded`
with the offending expression and the breached axis attached. Useful
as runtime guards in numerical pipelines, regression tests, and
"this function must stay simple" enforcement.

Non-SymPy returns pass through untouched, so the decorator is safe
to apply to functions that conditionally return Basic.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import sympy as sp

from .analyze import analyze


__all__ = ["CostLimitExceeded", "costlimit"]


_F = TypeVar("_F", bound=Callable[..., Any])


class CostLimitExceeded(Exception):
    """Raised by :func:`costlimit` when a wrapped function returns an
    expression exceeding the configured ceiling.

    Attributes
    ----------
    expression:
        The SymPy expression that breached the limit.
    axis:
        Which cost axis was breached: ``"predicted_depth"``,
        ``"max_path_r"``, or ``"pfaffian_r"``.
    measured:
        The expression's value on that axis.
    limit:
        The configured ceiling on that axis.
    """

    def __init__(
        self,
        expression: sp.Basic,
        axis: str,
        measured: int,
        limit: int,
    ) -> None:
        self.expression = expression
        self.axis = axis
        self.measured = measured
        self.limit = limit
        super().__init__(
            f"expression {expression!s} has {axis}={measured} > limit={limit}"
        )


def costlimit(
    *,
    predicted_depth: int | None = None,
    max_path_r: int | None = None,
    pfaffian_r: int | None = None,
) -> Callable[[_F], _F]:
    """Decorator that enforces a cost ceiling on the return value
    of the wrapped function.

    Parameters
    ----------
    predicted_depth:
        Maximum allowed value of ``analyze(result).predicted_depth``.
        ``None`` (default) disables this check.
    max_path_r:
        Maximum allowed value of ``analyze(result).max_path_r``.
    pfaffian_r:
        Maximum allowed value of ``analyze(result).pfaffian_r``.

    At least one of the three keyword arguments must be set. When
    multiple are set, the function's return value must satisfy
    *all* of them.

    Raises
    ------
    CostLimitExceeded
        At call time, when the function's return value (a SymPy
        expression) exceeds any of the configured limits.
    ValueError
        At decoration time, when no limit is provided.
    """
    if all(v is None for v in (predicted_depth, max_path_r, pfaffian_r)):
        raise ValueError(
            "costlimit() requires at least one of predicted_depth, "
            "max_path_r, or pfaffian_r"
        )

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if not isinstance(result, sp.Basic):
                return result   # non-SymPy returns pass through untouched

            analysis = analyze(result)
            if predicted_depth is not None and analysis.predicted_depth > predicted_depth:
                raise CostLimitExceeded(
                    result, "predicted_depth",
                    analysis.predicted_depth, predicted_depth,
                )
            if max_path_r is not None and analysis.max_path_r > max_path_r:
                raise CostLimitExceeded(
                    result, "max_path_r",
                    analysis.max_path_r, max_path_r,
                )
            if pfaffian_r is not None and analysis.pfaffian_r > pfaffian_r:
                raise CostLimitExceeded(
                    result, "pfaffian_r",
                    analysis.pfaffian_r, pfaffian_r,
                )
            return result
        return wrapper  # type: ignore[return-value]

    return decorator
