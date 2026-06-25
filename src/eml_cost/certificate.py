"""certificate — machine-checked non-oscillation certificate for an ODE.

The "verified kernel" surface: when :func:`eml_cost.classify_ode.classify_ode`
finds a function (given by ``y'' + p y' + q y = 0``) is EML-finite because it does
not oscillate, this attaches a certificate citing the Lean theorem that PROVES
the underlying analytic fact — so the verdict is backed by a machine-checked
proof, not a heuristic.

The Lean theorem (``MachLib.SturmNonOscillation.sturm_no_positive_bump``, no
sorryAx) is the Sturm result: for ``u'' = r u`` with ``r >= 0`` there is no
"positive arch" between two zeros, hence solutions do not oscillate.

Honest scope: ``r >= 0`` is *sufficient but not necessary* for non-oscillation
(an Euler equation with ``r = c/x^2``, ``-1/4 <= c < 0``, is non-oscillatory yet
``r < 0`` — the threshold near a regular singular point is ``-1/4``, not ``0``).
So a certificate is ``certified=True`` only when ``r >= 0`` is actually verified
on the domain; the EML-finite cases with ``r < 0`` get ``certified=False`` with a
note that their non-oscillation needs the sharper comparison (not yet in Lean).

    >>> c = certify_non_oscillation(0, -1)      # y'' - y = 0, r = 1 >= 0
    >>> c.certified, c.lean_theorem
    (True, 'MachLib.SturmNonOscillation.sturm_no_positive_bump')
    >>> certify_non_oscillation(0, 1) is None   # y'' + y = 0 oscillates -> no cert
    True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import sympy as sp

from .classify_ode import classify_ode

__all__ = ["NonOscillationCertificate", "certify_non_oscillation"]

_x = sp.Symbol("x", real=True)
_LEAN_THEOREM = "MachLib.SturmNonOscillation.sturm_no_positive_bump"


@dataclass(frozen=True)
class NonOscillationCertificate:
    """A non-oscillation / bounded-zeros certificate for an ODE-defined function.

    Attributes
    ----------
    certified:
        ``True`` when the Sturm Lean theorem applies (``r >= 0`` verified on the
        domain) — the property is machine-checked. ``False`` when the function is
        non-oscillatory but by the sharper regular-singular threshold not covered
        by the current Lean theorem.
    eml_class:
        The :func:`classify_ode` verdict (always ``'EML-finite'`` here).
    property:
        The certified property, in words.
    lean_theorem:
        Fully-qualified Lean theorem name when ``certified``; else ``None``.
    condition:
        The verified hypothesis (e.g. ``r = 1 >= 0 on R``).
    domain:
        ``'R'`` or ``'pos'``.
    note:
        Caveat for the non-``certified`` case.
    """

    certified: bool
    eml_class: str
    property: str
    lean_theorem: Optional[str]
    condition: str
    domain: str
    note: str = ""


def _nonneg_on_domain(r: sp.Expr, domain: str) -> bool:
    """Verify ``r >= 0`` on the domain for the shapes classify_ode produces
    (constant, ``c/x^2``, or 0) — the Sturm sufficient condition."""
    r = sp.simplify(r)
    if r == 0:
        return True
    if not r.free_symbols:                  # constant
        return bool(r >= 0)
    rx2 = sp.simplify(r * _x**2)            # r = c/x^2  ->  r*x^2 = c
    if not rx2.free_symbols:
        return bool(rx2 >= 0)               # c >= 0  =>  r >= 0 on x != 0
    return False                            # not provably nonneg here


def certify_non_oscillation(
    p: Union[sp.Expr, str, int],
    q: Union[sp.Expr, str, int],
    domain: str = "R",
) -> Optional[NonOscillationCertificate]:
    """Certificate that ``y'' + p y' + q y = 0`` solutions do not oscillate.

    Returns ``None`` when the function is EML-infinity (oscillatory) or a
    non-definitive candidate — there is no non-oscillation property to certify.
    Otherwise returns a :class:`NonOscillationCertificate`; ``certified`` is
    ``True`` exactly when ``r >= 0`` (the machine-checked Sturm hypothesis) holds.
    """
    res = classify_ode(p, q, domain)
    if res.eml_class != "EML-finite":
        return None
    r = res.r
    if _nonneg_on_domain(r, domain):
        return NonOscillationCertificate(
            certified=True,
            eml_class=res.eml_class,
            property=("no oscillation / bounded zeros: solutions of u''=r·u have no "
                      "positive arch between two zeros, hence finitely many zeros"),
            lean_theorem=_LEAN_THEOREM,
            condition=f"r = {sp.nsimplify(r)} >= 0 on {domain} "
                      f"(Sturm sufficient condition, machine-checked, no sorryAx)",
            domain=domain,
        )
    return NonOscillationCertificate(
        certified=False,
        eml_class=res.eml_class,
        property="non-oscillatory by the regular-singular -1/4 threshold",
        lean_theorem=None,
        condition=f"r = {sp.nsimplify(r)} is not >= 0",
        domain=domain,
        note=("non-oscillation holds by the sharper Sturm comparison near a regular "
              "singular point (threshold -1/4), not covered by sturm_no_positive_bump "
              "(which requires r >= 0) — not yet Lean-certified"),
    )
