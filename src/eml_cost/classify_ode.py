"""classify_ode — EML class of a function defined by a 2nd-order linear ODE.

Every other entry point in :mod:`eml_cost` takes a *closed-form expression*
(``analyze``, ``eml_depth``, ``is_pfaffian_not_eml`` look up names in
``PFAFFIAN_NOT_EML_R``). This module adds a new input modality: classify a
function given *implicitly* by a differential equation

    y'' + p(x) y' + q(x) y = 0,   p, q in Q(x),

into ``EML-infinity`` vs ``EML-finite``, with no closed form needed.

How
---
1. Reduce to the Schroedinger / normal form ``u'' = r u`` via ``y = u·exp(-½∫p)``,
   ``r = p²/4 + p'/2 - q`` (``r`` is the differential invariant).
2. Kovacic Step-1 structural data: finite poles + orders, order at infinity.
3. The Kovacic Case-1 *necessary*-condition screen (rigorous for ruling Case 1
   OUT — e.g. Airy: ``order at infinity = -1``).
4. The EML verdict by OSCILLATION (Sturm theory on ``r`` = the Infinite-Zeros
   Barrier = the compact differential-Galois torus factor). The unifying fact:
   at an order-2 singular point the local indicial exponents ``½(1 ± √(1+4b))``
   are complex **iff b < -1/4**, the same threshold as Sturm oscillation for
   ``r ~ b/x²``; for a constant ray ``r → L`` the exponents ``±√L`` are imaginary
   iff ``L < 0``. So "Kovacic complex local exponent" = "Sturm oscillatory" =
   "compact torus factor present" — one computation.

Background and the differential-Galois <-> EML correspondence this implements:
monogate-research ``exploration/differential_galois_eml_depth_2026_06_24/``.

Scope (honest)
--------------
The EML-infinity verdict (oscillation) and the Case-1-IMPOSSIBLE screen are
RIGOROUS. A *definitive* EML-finite verdict is only returned for the torus normal
forms (``r`` constant or ``r = c/x²``), where the solutions are real
exponentials / powers. For a non-oscillatory ``r`` of mixed/polynomial shape
(genuine special functions: modified Bessel, erf) the verdict is
``EML-finite?`` — a CANDIDATE: elementary iff Kovacic Case 1 actually constructs
a solution, which this module does not do (the d-degree polynomial search and
Cases 2/3 are out of scope). So the classifier never *contradicts*
``PFAFFIAN_NOT_EML_R``; it confirms the oscillatory entries and flags the rest.

    >>> classify_ode(0, 1).eml_class            # y'' + y = 0 (harmonic)
    'EML-infinity'
    >>> classify_ode(0, -1).eml_class           # y'' - y = 0 (hyperbolic)
    'EML-finite'
    >>> import sympy as sp
    >>> X = sp.Symbol('x')
    >>> classify_ode(0, -X).eml_class           # y'' - x y = 0 (Airy)
    'EML-infinity'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import sympy as sp

__all__ = ["OdeClass", "classify_ode"]

_x = sp.Symbol("x", real=True)


@dataclass(frozen=True)
class OdeClass:
    """Result of :func:`classify_ode`.

    Attributes
    ----------
    eml_class:
        ``'EML-infinity'``, ``'EML-finite'`` (definitive, torus form), or
        ``'EML-finite?'`` (non-oscillatory candidate; elementary iff Kovacic
        Case 1 constructs — not verified here).
    definitive:
        ``True`` when the verdict is rigorous (oscillation, or a torus normal
        form); ``False`` for an ``'EML-finite?'`` candidate.
    oscillatory:
        Does some solution have infinitely many zeros (the EML-infinity cause)?
    r:
        The normal-form invariant ``r`` in ``u'' = r u``.
    poles:
        Finite real poles of ``r`` as ``(point, order)`` pairs.
    order_at_infinity:
        ``deg(den) - deg(num)`` of ``r`` (``None`` for ``r = 0``).
    case1_possible:
        Whether Kovacic Case-1 *necessary* conditions hold (necessary, not
        sufficient).
    reason:
        One-line justification of ``eml_class``.
    detail:
        Per-ray oscillation reasoning + the Case-1 screen note.
    """

    eml_class: str
    definitive: bool
    oscillatory: bool
    r: sp.Expr
    poles: tuple
    order_at_infinity: Union[int, None]
    case1_possible: bool
    reason: str
    detail: tuple = field(default_factory=tuple)


# ── normal form + structural data ────────────────────────────────────────────
def _normal_form(p: sp.Expr, q: sp.Expr) -> sp.Expr:
    return sp.simplify(p**2 / 4 + sp.diff(p, _x) / 2 - q)


def _structural_data(r: sp.Expr):
    rr = sp.together(sp.cancel(r))
    num, den = sp.fraction(rr)
    num_p, den_p = sp.Poly(num, _x), sp.Poly(den, _x)
    o_inf = den_p.degree() - num_p.degree()
    poles = []
    for root, mult in sp.roots(den_p, _x).items():
        if root.is_real is not False:
            poles.append((sp.nsimplify(root), int(mult)))
    return tuple(poles), int(o_inf)


def _case1_necessary(poles, o_inf) -> tuple[bool, str]:
    for c, o in poles:
        if not (o in (1, 2) or (o >= 4 and o % 2 == 0)):
            return False, f"pole x={c} has order {o} (odd >=3) — forbidden for Case 1"
    if not (o_inf >= 2 or (o_inf <= 0 and o_inf % 2 == 0)):
        return False, f"order at infinity {o_inf} (odd <=1) — forbidden for Case 1"
    return True, "Case-1 necessary conditions hold (necessary, not sufficient)"


def _is_torus(r: sp.Expr) -> bool:
    """``r`` constant (-> const-coeff torus) or ``r = c/x²`` (-> Euler torus):
    the cases whose split solutions (e^{rx}, x^r) are elementary, real-exponent."""
    if not r.free_symbols:
        return True
    return not sp.simplify(r * _x**2).free_symbols


def _exp_disc(b: sp.Expr) -> sp.Expr:
    """Discriminant 1+4b of the local indicial exponents ½(1 ± √(1+4b))."""
    return sp.nsimplify(1 + 4 * b)


def _oscillation(r: sp.Expr, domain: str) -> tuple[bool, list[str]]:
    """Sturm / local-exponent oscillation test. domain ``'R'`` or ``'pos'``."""
    detail: list[str] = []
    osc = False
    rays = [("+oo", sp.oo, False)]
    if domain == "R":
        rays.append(("-oo", -sp.oo, False))
    else:
        rays.append(("0+", 0, True))  # the regular singular point of (0,oo) ODEs

    for name, pt, near_zero in rays:
        if near_zero:
            b = sp.limit(_x**2 * r, _x, 0)
            if b in (sp.oo, -sp.oo):
                continue
            disc = _exp_disc(b)
            if disc.is_negative:
                osc = True
                detail.append(f"x->{name}: r~({b})/x², 1+4b={disc}<0 => complex local "
                              f"exponents => oscillatory")
            else:
                detail.append(f"x->{name}: r~({b})/x², 1+4b={disc}>=0 => real exponents")
            continue
        L = sp.limit(r, _x, pt)
        if L == sp.oo:
            detail.append(f"x->{name}: r->+oo => non-oscillatory (Sturm r>=0)")
        elif L == -sp.oo:
            osc = True
            detail.append(f"x->{name}: r->-oo => oscillatory (Sturm)")
        elif L.is_negative:
            osc = True
            detail.append(f"x->{name}: r->{L}<0 => exponents ±√{L} imaginary => oscillatory")
        elif L.is_positive:
            detail.append(f"x->{name}: r->{L}>0 => exponents ±√{L} real => non-oscillatory")
        else:  # L == 0 borderline
            b = sp.limit(_x**2 * r, _x, pt)
            disc = _exp_disc(b)
            if disc.is_negative:
                osc = True
                detail.append(f"x->{name}: r~({b})/x², 1+4b={disc}<0 => oscillatory")
            else:
                detail.append(f"x->{name}: r~({b})/x², 1+4b={disc}>=0 => non-oscillatory")
    return osc, detail


def classify_ode(p: Union[sp.Expr, str, int],
                 q: Union[sp.Expr, str, int],
                 domain: str = "R") -> OdeClass:
    """Classify ``y'' + p y' + q y = 0`` into its EML class.

    Parameters
    ----------
    p, q:
        Coefficients (SymPy expressions, strings, or numbers) in ``x``.
    domain:
        ``'R'`` (whole line) or ``'pos'`` ((0, oo) — also checks x -> 0+ for the
        regular singular point of equidimensional / Bessel-type equations).

    Returns
    -------
    OdeClass
    """
    p = sp.sympify(p, locals={"x": _x})
    q = sp.sympify(q, locals={"x": _x})
    # Unify any symbol literally named "x" (e.g. a plain ``sp.Symbol('x')`` with
    # different assumptions) onto the module's real symbol, so the limits see it.
    for s in (p.free_symbols | q.free_symbols):
        if s.name == "x" and s != _x:
            p, q = p.subs(s, _x), q.subs(s, _x)
    r = _normal_form(p, q)

    if r == 0:
        return OdeClass(
            eml_class="EML-finite", definitive=True, oscillatory=False,
            r=r, poles=(), order_at_infinity=None, case1_possible=True,
            reason="r=0 (zero potential): u''=0 => solutions {1, x} => EML-finite",
            detail=("r=0 => u''=0 => polynomial * e^{-½∫p}",))

    poles, o_inf = _structural_data(r)
    c1_ok, c1_reason = _case1_necessary(poles, o_inf)
    osc, detail = _oscillation(r, domain)
    detail = tuple(detail) + (c1_reason,)

    if osc:
        return OdeClass(
            eml_class="EML-infinity", definitive=True, oscillatory=True,
            r=r, poles=poles, order_at_infinity=o_inf, case1_possible=c1_ok,
            reason="oscillatory => infinitely many zeros "
                   "(Infinite-Zeros Barrier / compact torus)",
            detail=detail)
    if _is_torus(r):
        return OdeClass(
            eml_class="EML-finite", definitive=True, oscillatory=False,
            r=r, poles=poles, order_at_infinity=o_inf, case1_possible=c1_ok,
            reason="non-oscillatory torus normal form (r const or c/x²) => "
                   "real exponentials / powers => EML-finite",
            detail=detail)
    return OdeClass(
        eml_class="EML-finite?", definitive=False, oscillatory=False,
        r=r, poles=poles, order_at_infinity=o_inf, case1_possible=c1_ok,
        reason="non-oscillatory but non-torus => elementary iff Kovacic Case 1 "
               "constructs a solution (not verified here)",
        detail=detail)
