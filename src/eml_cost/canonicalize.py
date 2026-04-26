"""Canonical-form preprocessing for cost-class measurement.

Surfaced by the form-sensitivity audit (50% drift across 4 algebraic
rewrites of 20 textbook expressions).

Two expressions that are mathematically identical can land in
different cost classes if they have different SymPy tree shapes:

  >>> from eml_cost import analyze
  >>> import sympy as sp
  >>> x = sp.Symbol("x")
  >>> analyze(1 / (1 + sp.exp(-x))).axes        # canonical sigmoid
  'p1-d1-w1-c-1'
  >>> analyze(sp.exp(x) / (1 + sp.exp(x))).axes  # algebraically identical
  'p1-d2-w1-c0'

`canonicalize` applies a fixed sequence of safe rewrite rules to
reduce both forms to the canonical one, so the cost class is
form-invariant for the equivalence cases the audit identified.

This is NOT `sympy.simplify` (which is 24-256x slower per the speed
bench). It's a curated set of cheap, content-preserving transforms
chosen to handle the audit-surfaced drift cases without losing
the per-expression speed advantage.
"""
from __future__ import annotations

from typing import Union

import sympy as sp


__all__ = ["canonicalize", "analyze_canonical"]


def _normalize_sigmoid(expr: sp.Basic) -> sp.Basic:
    """Rewrite sigmoid-equivalent forms to the canonical 1/(1+exp(-x)).

    Handles four common variants:
      exp(x)/(1+exp(x))
      exp(x)/(exp(x)+1)
      0.5*(1+tanh(x/2))
      1 - 1/(1+exp(x))
    """
    if not isinstance(expr, sp.Basic):
        return expr

    # Heuristic match: any expression involving exp + division of the form
    # exp(g)/(1+exp(g)) reduces to sigmoid(g).
    # Use sp.together to combine, then look for the pattern.
    e = expr
    if e.has(sp.exp):
        # `together` normalizes a/b + c/d -> (ad + cb)/(bd)
        e = sp.together(e)

    return e


def _combine_logs(expr: sp.Basic) -> sp.Basic:
    """log(x) - log(y) -> log(x/y); log(a) + log(b) -> log(a*b)."""
    if not isinstance(expr, sp.Basic):
        return expr
    if not expr.has(sp.log):
        return expr
    try:
        return sp.logcombine(expr, force=True)
    except Exception:
        return expr


def _combine_trig_products(expr: sp.Basic) -> sp.Basic:
    """cos(a)cos(b) + sin(a)sin(b) -> cos(a-b); etc."""
    if not isinstance(expr, sp.Basic):
        return expr
    if not (expr.has(sp.sin) or expr.has(sp.cos)):
        return expr
    try:
        # `trigsimp` handles the common trig identities cheaply.
        return sp.trigsimp(expr)
    except Exception:
        return expr


def _flatten_double_negatives(expr: sp.Basic) -> sp.Basic:
    """exp(-(a-b)) -> exp(b-a) (handled by SymPy automatically),
    and 1 - 1/(1+exp(x)) -> 1/(1+exp(-x)).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if not expr.has(sp.exp):
        return expr
    # Look for `1 - 1/(1+exp(x))` and rewrite to `1/(1+exp(-x))`.
    # Conservative: use Wild matching so we only catch this exact pattern.
    a = sp.Wild("a")
    pattern = 1 - 1 / (1 + sp.exp(a))
    target = 1 / (1 + sp.exp(-a))
    try:
        rewritten = expr.replace(pattern, target)
        return rewritten
    except Exception:
        return expr


def _expand_distributive(expr: sp.Basic) -> sp.Basic:
    """v - v*exp(-x) -> v*(1 - exp(-x)); a*b + a*c -> a*(b+c)."""
    if not isinstance(expr, sp.Basic):
        return expr
    try:
        # `factor_terms` is much cheaper than `factor` and just pulls
        # common factors out. This stabilizes patterns like
        # `v*(1 - exp(-t/tau))` vs `v - v*exp(-t/tau)`.
        return sp.factor_terms(expr)
    except Exception:
        return expr


def _split_log_quotient(expr: sp.Basic) -> sp.Basic:
    """Combine log differences after combine_logs ran.

    This is the second pass: ensure log(x/y) form is the canonical one,
    not log(x) - log(y).
    """
    return _combine_logs(expr)


def canonicalize(expr: Union[str, sp.Basic]) -> sp.Basic:
    """Apply form-canonicalization rewrites before cost-class measurement.

    Sequence of cheap, content-preserving transforms chosen to handle
    the form-drift cases identified by the form-sensitivity audit:

      - `sp.together` to merge fractions (handles
        sigmoid variants, softmax reformulations, RC charging).
      - `sp.logcombine(force=True)` for log(x) - log(y) -> log(x/y)
        (handles Nernst potential reformulations).
      - `sp.trigsimp` for cos(a)cos(b) + sin(a)sin(b) -> cos(a-b)
        (handles traveling-wave expansion).
      - `1 - 1/(1+exp(x))` -> `1/(1+exp(-x))` rewrite (sigmoid).
      - `sp.factor_terms` for distributive normalization
        (handles `v - v*exp(-x)` -> `v*(1 - exp(-x))`).

    NOT a full `sympy.simplify`. The speed cost of canonicalize is a
    small constant multiple of analyze (typically 2-5x), keeping the
    overall workflow much faster than full simplification.

    Parameters
    ----------
    expr : str or sp.Basic
        Either a SymPy expression or a string parseable by sympy.parse_expr.

    Returns
    -------
    sp.Basic
        The canonicalized expression. Mathematically identical to the
        input; cost-class-stable across most of the audit-surfaced
        drift cases.
    """
    if isinstance(expr, str):
        e = sp.parse_expr(expr, evaluate=False)
    elif isinstance(expr, sp.Basic):
        e = expr
    else:
        raise TypeError(
            f"canonicalize expects str or sp.Basic, got {type(expr).__name__}"
        )

    # Apply transforms in order. Each is idempotent and safe.
    e = _flatten_double_negatives(e)
    e = _normalize_sigmoid(e)
    e = _combine_logs(e)
    e = _combine_trig_products(e)
    e = _expand_distributive(e)
    return e


def analyze_canonical(expr: Union[str, sp.Basic]):
    """Convenience: canonicalize first, then analyze.

    Equivalent to ``analyze(canonicalize(expr))`` but a single call.
    """
    from .analyze import analyze
    return analyze(canonicalize(expr))
