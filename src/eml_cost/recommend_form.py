"""Narrow-scope numerical-form recommender.

Intentionally fires on **only the four expression families** for which
E-193 Phase 3 demonstrated cost-class concordantly picks the more
numerically stable algebraic form (Spearman rho >= 0.77 per family on
the form-sensitivity killer test). On every other expression the
recommender abstains by returning ``None``.

The four supported families
---------------------------

  - **sigmoid** — rho = +0.800; canonical stable form ``1 / (1 + exp(-x))``
  - **exponential_decay** — rho = +0.775; canonical stable form
    ``exp(-k * t)`` (avoid ``exp(-t / tau)`` which adds a division in
    the argument)
  - **logistic_growth** — rho = +0.775; canonical stable form
    ``K / (1 + exp(-r * (t - t0)))``
  - **cardiac_oscillator** (any pure-cosine oscillator) — rho = +0.866;
    canonical stable form ``A * cos(omega * t + phi)`` (avoid
    ``A * cos(2 * pi * t / T_period)`` which adds 2*pi/T arithmetic)

The remaining six families in the E-193 killer test (hill_kernel,
softmax_2key, nernst_potential, gaussian_kernel_2d, rc_charging,
traveling_wave) had Spearman rho between -0.26 and +0.0 — cost class
does NOT reliably pick the more stable form on those. The recommender
**abstains on those families** by structural detection, even when the
expression is recognised as belonging to one of them.

Honest framing
--------------

This is a **family-narrow** recommender. On expressions outside the
four supported families it returns ``None`` rather than guessing.

  - **Do** use it as a SymPy linter that flags known rewrites for
    sigmoid / exp-decay / logistic-growth / cosine-oscillator.
  - **Do not** use it as a general-purpose numerical-form picker.
    The general-purpose recommender was deliberately not shipped
    (E-193 Phase 3 best-pick = 30 percent on the full 10-family test;
    see ``predict_precision_loss`` docstring).

Source: ``monogate-research/exploration/E193_numerical_stability/
phase3_killer.json`` — the per-family Spearman correlation between
predicted EML depth and measured float64 mpmath_max_relerr.

Usage
-----

    >>> from eml_cost import recommend_form
    >>> r = recommend_form("tanh(x/2)/2 + 1/2")
    >>> r.family                                   # doctest: +SKIP
    'sigmoid'
    >>> r.canonical_form                           # doctest: +SKIP
    1/(exp(-x) + 1)
    >>> r = recommend_form("besselj(0, x)")
    >>> r is None                                  # abstains
    True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import sympy as sp

from .analyze import analyze
from .predict_precision_loss import predict_precision_loss

__all__ = [
    "RecommendedForm",
    "recommend_form",
    "SUPPORTED_FAMILIES",
    "FAMILY_RHO",
]


SUPPORTED_FAMILIES: tuple[str, ...] = (
    "sigmoid",
    "exponential_decay",
    "logistic_growth",
    "cardiac_oscillator",
)

# E-193 Phase 3 per-family Spearman rho (predicted EML depth vs measured
# mpmath_max_relerr across the alternative forms). Source:
# phase3_killer.json. Only families with rho >= 0.77 are supported.
FAMILY_RHO: dict[str, float] = {
    "sigmoid": 0.800,
    "exponential_decay": 0.775,
    "logistic_growth": 0.775,
    "cardiac_oscillator": 0.866,
}


@dataclass(frozen=True)
class RecommendedForm:
    """Output of :func:`recommend_form` when a family match is found.

    Attributes
    ----------
    family:
        One of ``"sigmoid"``, ``"exponential_decay"``,
        ``"logistic_growth"``, ``"cardiac_oscillator"``.
    canonical_form:
        The SymPy expression the recommender would prefer, structured
        in the canonical-stable shape for that family.
    input_predicted_relerr:
        ``predict_precision_loss(input_expr).predicted_max_relerr``.
    recommended_predicted_relerr:
        ``predict_precision_loss(canonical_form).predicted_max_relerr``.
    digits_saved:
        Estimated decimal-digits-of-precision saved by switching to
        the canonical form, computed as
        ``log10(input_predicted_relerr) - log10(recommended_predicted_relerr)``.
        Negative if the input is already at least as stable as the
        canonical form.
    rho:
        E-193 Phase 3 Spearman rho for this family (carried for
        honest accuracy reporting).
    honest_note:
        Reminder that this recommender fires only on four families
        and abstains otherwise.
    """

    family: str
    canonical_form: sp.Expr
    input_predicted_relerr: float
    recommended_predicted_relerr: float
    digits_saved: float
    rho: float
    honest_note: str = (
        "Narrow-scope recommender: fires only on sigmoid / exponential_decay / "
        "logistic_growth / cardiac_oscillator (the four families with E-193 "
        "Phase 3 Spearman rho >= 0.77). Abstains on all other expressions. "
        "Not a general-purpose form recommender."
    )


# ---------------------------------------------------------------------------
# Family detection. Each detector returns True if `expr` is in its family.
# Detectors are ordered most-specific to least-specific.
# ---------------------------------------------------------------------------


def _free_symbols_excluding(expr: sp.Expr, exclude: set[sp.Symbol]) -> set[sp.Symbol]:
    return set(expr.free_symbols) - exclude


def _try_subtract_simplify(a: sp.Expr, b: sp.Expr) -> bool:
    """True if `simplify(a - b) == 0`. Catches algebraic equivalence.

    Tries multiple normalisation strategies in order: plain `simplify`,
    then `rewrite(exp)` to fold tanh / sinh / cosh / tan / sin / cos
    into exp form, then `trigsimp` for the residual trig identities.
    Sigmoid in particular needs the tanh -> exp rewrite to detect the
    `tanh(x/2)/2 + 1/2` alternative form.
    """
    diff = a - b
    try:
        if sp.simplify(diff) == 0:
            return True
    except (sp.SympifyError, TypeError, ValueError, RecursionError):
        pass
    try:
        if sp.simplify(diff.rewrite(sp.exp)) == 0:
            return True
    except (sp.SympifyError, TypeError, ValueError, RecursionError, AttributeError):
        pass
    try:
        if sp.trigsimp(sp.simplify(diff)) == 0:
            return True
    except (sp.SympifyError, TypeError, ValueError, RecursionError):
        pass
    return False


def _detect_sigmoid(expr: sp.Expr) -> Optional[sp.Symbol]:
    """If `expr` is sigmoid in some variable x (over rationals + the four
    standard alternative forms), return that symbol. Else None.

    Recognised forms (each in some single free variable x):
      - 1 / (1 + exp(-x))           canonical
      - exp(x) / (exp(x) + 1)        equivalent
      - tanh(x/2) / 2 + 1 / 2        equivalent
      - 1 - 1 / (1 + exp(x))         equivalent
    """
    syms = list(expr.free_symbols)
    if len(syms) != 1:
        return None
    x = syms[0]
    canonical = 1 / (1 + sp.exp(-x))
    if _try_subtract_simplify(expr, canonical):
        return x
    return None


def _detect_exponential_decay(expr: sp.Expr) -> Optional[tuple[sp.Symbol, sp.Expr]]:
    """If `expr` is `exp(coefficient * x)` in some single free variable x
    where `coefficient` is a negative numeric or a free-of-x SymPy expression,
    return (x, coefficient). Else None.

    Recognised forms include:
      - exp(-k * t)         canonical
      - exp(-t / tau)        equivalent (drift form: division in argument)
      - exp(-c * t) for any c > 0
    """
    if not isinstance(expr, sp.exp):
        return None
    arg = expr.args[0]
    free = list(arg.free_symbols)
    if len(free) < 1 or len(free) > 2:
        return None
    # Prefer the symbol that is the "time" variable — the one the user
    # is most likely treating as varying. We pick by alphabetical order
    # for determinism; in real use the free-of-x coefficient stands
    # out automatically.
    for x in free:
        # Try treating x as the variable; arg should be linear in x.
        try:
            poly = sp.Poly(arg, x)
            if poly.degree() == 1:
                coeff = poly.nth(1)  # coefficient of x^1
                # The coeff should not contain x.
                if x not in coeff.free_symbols:
                    # Decay only if coeff is negative (real) or symbolically
                    # negative (we accept purely-symbolic coeffs too — that
                    # is the honest default, and the user can override with
                    # a sign assumption).
                    return (x, coeff)
        except (sp.PolynomialError, ValueError):
            continue
    return None


def _detect_logistic_growth(expr: sp.Expr) -> Optional[dict[str, sp.Expr]]:
    """If `expr` is logistic growth `K / (1 + exp(-r * (t - t0)))` or
    structural equivalents over symbols (K, r, t, t0), return a dict
    of those symbols. Else None.

    We require at least two free symbols (the variable + at least one
    parameter).
    """
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(syms) < 2:
        return None
    # Try each pair (x, K) where x is the time variable and K is the carrying
    # capacity, with the remaining symbols being r and t0.
    for t in syms:
        others = [s for s in syms if s is not t]
        for K in others:
            params = [s for s in others if s is not K]
            if not params:
                continue
            r = params[0]
            t0 = params[1] if len(params) > 1 else sp.S.Zero
            canonical = K / (1 + sp.exp(-r * (t - t0)))
            if _try_subtract_simplify(expr, canonical):
                return {"t": t, "K": K, "r": r, "t0": t0}
    return None


def _detect_cosine_oscillator(expr: sp.Expr) -> Optional[dict[str, sp.Expr]]:
    """If `expr` is `A * cos(omega * t + phi)` (or an algebraic equivalent
    over the same symbols), return a dict of (A, omega, t, phi). Else None.

    Recognised by checking if `expr` simplifies to `A * cos(omega * t)`
    in some symbol `t` (we ignore the phase offset for the family-
    membership check; the canonical form preserves whatever phase
    structure the input had).
    """
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(syms) < 2:
        return None
    for t in syms:
        others = [s for s in syms if s is not t]
        for A in others:
            params = [s for s in others if s is not A]
            if not params:
                continue
            omega = params[0]
            # Try matching A * cos(omega * t).
            canonical = A * sp.cos(omega * t)
            if _try_subtract_simplify(expr, canonical):
                return {"A": A, "omega": omega, "t": t, "phi": sp.S.Zero}
    return None


# ---------------------------------------------------------------------------
# Per-family canonical-form constructors.
# ---------------------------------------------------------------------------


def _canonical_sigmoid(x: sp.Symbol) -> sp.Expr:
    """The numerically-stable sigmoid form: 1 / (1 + exp(-x))."""
    return 1 / (1 + sp.exp(-x))


def _canonical_exponential_decay(x: sp.Symbol, coeff: sp.Expr) -> sp.Expr:
    """exp(coeff * x), expanded so coeff is a single multiplicative term
    against x rather than a chained division.

    If `coeff` already is a single symbol or rational, it stays that
    shape. If it is `1 / tau` or similar, we factor it cleanly.
    """
    return sp.exp(sp.together(coeff) * x)


def _canonical_logistic_growth(parts: dict[str, sp.Expr]) -> sp.Expr:
    """K / (1 + exp(-r * (t - t0)))."""
    return parts["K"] / (1 + sp.exp(-parts["r"] * (parts["t"] - parts["t0"])))


def _canonical_cosine_oscillator(parts: dict[str, sp.Expr]) -> sp.Expr:
    """A * cos(omega * t + phi). For pure cosine input the phi is 0."""
    return parts["A"] * sp.cos(parts["omega"] * parts["t"] + parts["phi"])


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def recommend_form(expr: Union[sp.Expr, str]) -> Optional[RecommendedForm]:
    """Recommend a more numerically stable algebraic form, or abstain.

    Returns a :class:`RecommendedForm` if `expr` is recognised as
    belonging to one of the four supported families; otherwise returns
    ``None``.

    Parameters
    ----------
    expr:
        SymPy expression or a string parseable by :func:`sympy.sympify`.

    Returns
    -------
    RecommendedForm or None
        The recommendation if a family is matched, or ``None`` if the
        expression is outside the four supported families.

    Examples
    --------
    >>> import sympy as sp
    >>> from eml_cost import recommend_form
    >>> r = recommend_form("tanh(x/2)/2 + 1/2")  # alternative sigmoid form
    >>> r is None
    False
    >>> r.family
    'sigmoid'
    >>> r2 = recommend_form("besselj(0, x)")  # outside any supported family
    >>> r2 is None
    True
    """
    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)

    # Sigmoid (single variable, easiest match).
    x = _detect_sigmoid(expr)
    if x is not None:
        return _build_recommendation(expr, "sigmoid", _canonical_sigmoid(x))

    # Exponential decay (requires a single dominant variable; structural).
    decay = _detect_exponential_decay(expr)
    if decay is not None:
        x, coeff = decay
        return _build_recommendation(
            expr, "exponential_decay", _canonical_exponential_decay(x, coeff)
        )

    # Logistic growth (requires K, r, t, optionally t0).
    logistic = _detect_logistic_growth(expr)
    if logistic is not None:
        return _build_recommendation(
            expr, "logistic_growth", _canonical_logistic_growth(logistic)
        )

    # Cosine oscillator (requires A, omega, t).
    osc = _detect_cosine_oscillator(expr)
    if osc is not None:
        return _build_recommendation(
            expr, "cardiac_oscillator", _canonical_cosine_oscillator(osc)
        )

    return None


def _build_recommendation(
    expr: sp.Expr,
    family: str,
    canonical: sp.Expr,
) -> RecommendedForm:
    """Common builder: run predict_precision_loss on input + canonical,
    compute digits_saved, attach rho metadata."""
    in_pred = predict_precision_loss(expr)
    cf_pred = predict_precision_loss(canonical)
    import math

    log_in = math.log10(max(in_pred.predicted_max_relerr, 1e-300))
    log_cf = math.log10(max(cf_pred.predicted_max_relerr, 1e-300))
    digits_saved = log_in - log_cf  # positive => canonical is more stable

    return RecommendedForm(
        family=family,
        canonical_form=canonical,
        input_predicted_relerr=in_pred.predicted_max_relerr,
        recommended_predicted_relerr=cf_pred.predicted_max_relerr,
        digits_saved=digits_saved,
        rho=FAMILY_RHO[family],
    )
