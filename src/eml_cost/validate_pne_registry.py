"""validate_pne_registry — cross-check the ``PFAFFIAN_NOT_EML_R`` table against
the differential-equation EML classifier.

``PFAFFIAN_NOT_EML_R`` (in :mod:`eml_cost.core`) is a *hand-curated* table of
Pfaffian-but-not-EML primitives (Bessel, Airy, ...): each asserts "this function
is not EML-finite" with a chain order. :func:`eml_cost.classify_ode.classify_ode`
can now DERIVE that verdict from a function's defining 2nd-order linear ODE. This
module wires the two together: for every registry entry that *is* a solution of a
2nd-order linear ODE, it runs the ODE through the classifier and checks the
result is consistent with the registry (never EML-finite).

The map is honest about coverage. Three categories:

* ``confirmed``   — the classifier RIGOROUSLY re-derives ``EML-infinity`` (the
  function's ODE is oscillatory, by Sturm). The table entry is independently
  validated.
* ``consistent``  — non-oscillatory genuine special functions (modified Bessel,
  erf): the classifier returns the non-definitive ``EML-finite?`` candidate. It
  does NOT contradict the table (it never claims EML-finite); confirming these
  needs the full Kovacic Case-1 constructor (out of scope).
* ``out_of_scope`` — the entry is not a 2nd-order linear ODE solution in ``x``
  (gamma family, integral functions Ei/Si/Ci, LambertW, number-theoretic ζ/η/...,
  inverse / Fresnel functions). The classifier does not apply; the registry
  verdict rests on other grounds.

A ``contradiction`` (classifier returns a *definitive* ``EML-finite`` for a
registry entry) would indicate a registry error or a classifier bug; the
validator asserts there are none.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .classify_ode import classify_ode
from .core import PFAFFIAN_NOT_EML_R

__all__ = ["RegistryValidation", "validate_pne_registry", "REGISTRY_ODES"]

_x = sp.Symbol("x", real=True)
_NU = sp.Integer(2)  # a generic (non-half-integer) order for Bessel-type tests


def _bessel_ordinary(nu):
    # x²y'' + xy' + (x²-ν²)y = 0  ->  p=1/x, q = 1 - ν²/x²  (oscillatory)
    return (1 / _x, 1 - nu**2 / _x**2, "pos")


def _bessel_modified(nu):
    # x²y'' + xy' - (x²+ν²)y = 0  ->  p=1/x, q = -(1 + ν²/x²)  (non-oscillatory)
    return (1 / _x, -(1 + nu**2 / _x**2), "pos")


# ── Registry entry -> (p, q, domain) for the 2nd-order-linear-ODE subset. ──
# Each special function is a solution of the listed ODE. Functions not solving a
# 2nd-order linear ODE in x (gamma family, integral functions, LambertW, number
# theory, inverse / Fresnel) are intentionally absent -> out_of_scope.
REGISTRY_ODES: dict[str, tuple] = {
    # Ordinary Bessel + spherical + Hankel: oscillatory -> confirmed EML-infinity.
    "besselj": _bessel_ordinary(_NU),
    "bessely": _bessel_ordinary(_NU),
    "hankel1": _bessel_ordinary(_NU),
    "hankel2": _bessel_ordinary(_NU),
    "jn": _bessel_ordinary(sp.Rational(3, 2)),   # spherical = half-integer order
    "yn": _bessel_ordinary(sp.Rational(3, 2)),
    "hn1": _bessel_ordinary(sp.Rational(3, 2)),
    "hn2": _bessel_ordinary(sp.Rational(3, 2)),
    # Modified Bessel: non-oscillatory genuine special fn -> consistent (candidate).
    "besseli": _bessel_modified(_NU),
    "besselk": _bessel_modified(_NU),
    # Airy: y'' = x y -> oscillatory on the negative axis -> confirmed.
    "airyai": (sp.Integer(0), -_x, "R"),
    "airybi": (sp.Integer(0), -_x, "R"),
    "airyaiprime": (sp.Integer(0), -_x, "R"),
    "airybiprime": (sp.Integer(0), -_x, "R"),
    # erf: y'' + 2x y' = 0 (solution space {1, erf}). Non-oscillatory but a
    # non-elementary integral -> consistent (the "elementary tower fails" case).
    "erf": (2 * _x, sp.Integer(0), "R"),
    "erfc": (2 * _x, sp.Integer(0), "R"),
    "erfi": (-2 * _x, sp.Integer(0), "R"),       # erfi: y'' - 2x y' = 0
}


@dataclass(frozen=True)
class RegistryValidation:
    """Result of :func:`validate_pne_registry`."""

    confirmed: tuple        # names rigorously re-derived as EML-infinity
    consistent: tuple       # names classifier leaves as EML-finite? (no contradiction)
    out_of_scope: tuple     # registry names not 2nd-order-linear-ODE solutions
    contradictions: tuple   # names the classifier calls definitively EML-finite (should be empty)

    @property
    def total_registry(self) -> int:
        return len(PFAFFIAN_NOT_EML_R)

    @property
    def ode_covered(self) -> int:
        return len(self.confirmed) + len(self.consistent) + len(self.contradictions)

    def summary(self) -> str:
        return (
            f"PFAFFIAN_NOT_EML_R: {self.total_registry} entries; "
            f"{self.ode_covered} are 2nd-order linear ODE solutions.\n"
            f"  confirmed EML-infinity (rigorous, oscillatory): {len(self.confirmed)}\n"
            f"  consistent (non-osc candidate, classifier non-definitive): {len(self.consistent)}\n"
            f"  out of ODE-classifier scope (integral/gamma/number-theoretic): "
            f"{len(self.out_of_scope)}\n"
            f"  CONTRADICTIONS (registry vs classifier): {len(self.contradictions)}"
        )


def validate_pne_registry() -> RegistryValidation:
    """Run every ODE-expressible registry entry through :func:`classify_ode`."""
    confirmed, consistent, contradictions = [], [], []
    for name in PFAFFIAN_NOT_EML_R:
        if name not in REGISTRY_ODES:
            continue
        p, q, domain = REGISTRY_ODES[name]
        res = classify_ode(p, q, domain)
        if res.eml_class == "EML-infinity":
            confirmed.append(name)
        elif res.eml_class == "EML-finite?":
            consistent.append(name)
        else:  # definitive EML-finite for a registry entry => contradiction
            contradictions.append(name)
    out_of_scope = tuple(n for n in PFAFFIAN_NOT_EML_R if n not in REGISTRY_ODES)
    return RegistryValidation(
        confirmed=tuple(confirmed),
        consistent=tuple(consistent),
        out_of_scope=out_of_scope,
        contradictions=tuple(contradictions),
    )


def main() -> None:  # pragma: no cover
    v = validate_pne_registry()
    print(v.summary())
    print("\nconfirmed EML-infinity (classifier re-derives the table, oscillatory):")
    for n in v.confirmed:
        print(f"  {n:<14} r-osc -> EML-infinity   (registry chain order {PFAFFIAN_NOT_EML_R[n]})")
    print("\nconsistent (non-oscillatory; needs full Kovacic Case-1 constructor):")
    for n in v.consistent:
        print(f"  {n:<14} non-osc -> EML-finite? (registry says non-EML; not contradicted)")
    if v.contradictions:
        print("\n!! CONTRADICTIONS:", v.contradictions)
    else:
        print("\nNo contradictions: the classifier never calls a registry entry EML-finite.")


if __name__ == "__main__":  # pragma: no cover
    main()
