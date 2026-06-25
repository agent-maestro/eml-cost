"""lint_ode — EML-readiness linter for functions defined by a 2nd-order ODE.

Before synthesising an EML kernel (or running EML symbolic regression) for a
function given by a differential equation ``y'' + p(x) y' + q(x) y = 0``, lint it
to learn whether the function is even *representable* as a finite EML tree:

* **EML-finite**   — representable exactly; safe for EML kernel synthesis.
* **EML-infinity** — solutions oscillate (compact differential-Galois torus /
  Infinite-Zeros Barrier); no finite EML tree exists — an EML kernel can only
  approximate on a bounded interval.
* **non-Liouvillian** — Kovacic Case-1 impossible (Galois group SL2); no
  elementary closed form at all (Airy, generic Bessel).
* **EML-candidate** — non-oscillatory but non-torus; elementary only if Kovacic
  Case 1 constructs a solution (not verified) — treat as possibly non-elementary
  (modified Bessel, erf).

Thin wrapper over :func:`eml_cost.classify_ode.classify_ode` that turns the
verdict into actionable lint findings.

    >>> [f.code for f in lint_ode(0, 1)]            # y'' + y = 0 (oscillatory)
    ['EML-INF-OSC']
    >>> [f.code for f in lint_ode(0, -1)]           # y'' - y = 0 (representable)
    ['EML-FINITE']
    >>> sorted(f.code for f in lint_ode(0, "-x"))   # Airy: both flags
    ['EML-INF-OSC', 'NON-LIOUVILLIAN']
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import sympy as sp

from .classify_ode import classify_ode

__all__ = ["OdeFinding", "lint_ode", "lint_odes", "format_ode_findings"]


@dataclass(frozen=True)
class OdeFinding:
    """One ODE-linter result.

    Attributes
    ----------
    code:
        ``'EML-FINITE'`` (representable), ``'EML-INF-OSC'`` (oscillatory),
        ``'NON-LIOUVILLIAN'`` (no elementary solution), or ``'EML-CANDIDATE'``
        (non-definitive).
    severity:
        ``'info'`` (representable, all clear) or ``'warn'`` (not exactly
        representable / uncertain).
    message:
        Actionable one-liner for the EML-kernel author.
    eml_class:
        The underlying :class:`~eml_cost.classify_ode.OdeClass` verdict.
    name:
        Optional label for the ODE (echoed in the message).
    """

    code: str
    severity: str
    message: str
    eml_class: str
    name: Union[str, None] = None


def lint_ode(p: Union[sp.Expr, str, int],
             q: Union[sp.Expr, str, int],
             domain: str = "R",
             *, name: Union[str, None] = None) -> list[OdeFinding]:
    """Lint ``y'' + p y' + q y = 0`` for EML representability.

    Parameters
    ----------
    p, q, domain:
        As in :func:`eml_cost.classify_ode.classify_ode`.
    name:
        Optional label echoed in finding messages.

    Returns
    -------
    list[OdeFinding]
        Usually one finding (the verdict); oscillatory non-Liouvillian ODEs
        (Airy, generic Bessel) get a second ``NON-LIOUVILLIAN`` finding.
    """
    res = classify_ode(p, q, domain)
    tag = f"{name}: " if name else ""
    findings: list[OdeFinding] = []

    if res.eml_class == "EML-infinity":
        findings.append(OdeFinding(
            code="EML-INF-OSC", severity="warn", eml_class=res.eml_class, name=name,
            message=(f"{tag}solutions oscillate ({res.reason}). NOT a finite EML tree — "
                     f"an EML kernel can only approximate on a bounded interval."),
        ))
        if res.case1_possible is False:
            findings.append(OdeFinding(
                code="NON-LIOUVILLIAN", severity="warn", eml_class=res.eml_class, name=name,
                message=(f"{tag}Kovacic Case-1 impossible (order at infinity "
                         f"{res.order_at_infinity}): solutions are non-Liouvillian "
                         f"(Galois group SL2) — no elementary closed form exists."),
            ))
    elif res.eml_class == "EML-finite":
        findings.append(OdeFinding(
            code="EML-FINITE", severity="info", eml_class=res.eml_class, name=name,
            message=(f"{tag}solutions are EML-finite ({res.reason}). Representable as a "
                     f"finite EML tree — safe for EML kernel synthesis."),
        ))
    else:  # EML-finite?
        findings.append(OdeFinding(
            code="EML-CANDIDATE", severity="warn", eml_class=res.eml_class, name=name,
            message=(f"{tag}non-oscillatory but non-torus: elementary only if Kovacic "
                     f"Case-1 constructs a solution (not verified). Treat as possibly "
                     f"non-elementary (e.g. modified Bessel, erf)."),
        ))
    return findings


def lint_odes(odes: dict) -> list[OdeFinding]:
    """Lint a catalogue ``{name: (p, q[, domain])}`` and concatenate findings."""
    out: list[OdeFinding] = []
    for name, spec in odes.items():
        if len(spec) == 3:
            p, q, domain = spec
        else:
            (p, q), domain = spec, "R"
        out.extend(lint_ode(p, q, domain, name=name))
    return out


def format_ode_findings(findings: list[OdeFinding]) -> str:
    """One line per finding, ``SEVERITY  CODE  message`` — shell-friendly."""
    return "\n".join(f"{f.severity.upper():<5} {f.code:<16} {f.message}" for f in findings)
