"""find_siblings — cross-domain structural neighbour search.

Given any symbolic expression, return the structurally most similar
expressions across the bundled 578-row cross-domain corpus. Distance
is the weighted Euclidean metric on (r, degree, width, c_osc) defined
by :class:`PfaffianProfile`. The metric obeys the triangle inequality
(verified in ``tests/test_profile_metric.py``).

The corpus spans 12 subdomains (bench-300 + E-196 + Robotics + Human
Body + Music & Sound + Olfactory + Color Science) — see
``monogate-research/data/bench.md`` for the cohort breakdown. Final
shape: 576 / 578 rows parseable; 2 outliers skipped at packaging time.

Evidence
--------

  - The headline cross-domain class ``p2-d5-w2-c1`` holds at 38 corpus
    members vs 135 null members at 2.43× enrichment (BH-q = 1.49e-4)
    after the Color Science merge.
  - Phase 4f Spearman ρ = +0.890 (n=175) on the dynamics counter — so
    structural similarity in the Pfaffian profile reliably co-tracks
    physical-mode similarity in the underlying expression.
  - Source: ``data/bench.md``,
    ``exploration/color_science_subdomain/phase4f_extended_v5_summary.json``.

Usage
-----

    >>> from eml_cost import find_siblings
    >>> for s in find_siblings("A * exp(-zeta*omega*t) * cos(omega_d*t)"):
    ...     print(f"{s.distance:5.2f}  {s.domain:<14} {s.name}")    # doctest: +SKIP
     0.00  engineering   damped harmonic oscillator
     0.00  bio           cardiac action potential decay
     0.00  music_sound   exponential reverb
     1.41  olfactory     receptor potential
     2.83  signal        Gabor wavelet

Honest framing
--------------

  - **OBSERVATION** tier: structural similarity is not the same as
    physical equivalence. Two expressions with identical
    ``cost_class`` may model unrelated physical systems.
  - The corpus is curated, not exhaustive — absence of a sibling
    means "not in our 578-row sample," not "no sibling exists."
  - Distance is in cost-class units, not normalised: a distance of
    2.0 on an oscillator and on a polynomial mean different things.
"""
from __future__ import annotations

import csv
import functools
import re
from dataclasses import dataclass
from importlib.resources import files as _files
from typing import Any, Optional, Union

import sympy as sp

from .profile import DEFAULT_WEIGHTS, PfaffianProfile

__all__ = ["Sibling", "find_siblings", "corpus_size", "corpus_domains"]


@dataclass(frozen=True)
class Sibling:
    """One structural neighbour returned by :func:`find_siblings`.

    Attributes
    ----------
    name:
        Human-readable label of the corpus expression.
    domain:
        Subdomain tag (e.g. ``"engineering_physics"``, ``"music_sound"``,
        ``"olfactory"``, ``"color_science"``).
    expression:
        The original SymPy expression string from the corpus.
    cost_class:
        Canonical axes string ``"p{r}-d{degree}-w{width}-c{c_osc}"``.
    distance:
        Weighted Euclidean distance from the query profile (lower is
        more similar; 0 means same cost class).
    profile:
        The corpus row's :class:`PfaffianProfile`.
    """

    name: str
    domain: str
    expression: str
    cost_class: str
    distance: float
    profile: PfaffianProfile


# Permissive symbol overrides so corpus expressions parse correctly.
_PERMISSIVE = {
    "zeta": sp.Symbol("zeta"),
    "I": sp.Symbol("I"),
    "N": sp.Symbol("N"),
    "E": sp.Symbol("E"),
    "beta": sp.Symbol("beta"),
    "gamma": sp.Symbol("gamma"),
    "Lambda": sp.Symbol("Lambda"),
    "S": sp.Symbol("S"),
    "Q": sp.Symbol("Q"),
    "O": sp.Symbol("O"),
    "lam": sp.Symbol("lam"),
}


def _safe_expr(s: str) -> str:
    """Avoid Python-keyword `lambda` collision by renaming to `lam`."""
    return re.sub(r"\blambda\b", "lam", s)


@functools.lru_cache(maxsize=1)
def _corpus_rows() -> tuple[dict[str, Any], ...]:
    """Load the 576-row precomputed corpus from the package data file.

    Cached after first call. Each row is a dict with keys: name,
    domain, expr, cost_class, profile (PfaffianProfile object).
    """
    data_path = _files("eml_cost.data") / "corpus_578.csv"
    rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            profile = PfaffianProfile(
                r=int(r["r"]),
                degree=int(r["degree"]),
                width=int(r["width"]),
                cost_class=r["cost_class"],
                oscillatory=int(r["c_osc"]) > 0,
                corrections={
                    "c_osc": int(r["c_osc"]),
                    "c_composite": int(r["c_composite"]),
                    "delta_fused": int(r["delta_fused"]),
                },
                expression=r["expr"],
                canonical_form=r["expr"],
                is_pfaffian_not_eml=bool(int(r["is_pfaffian_not_eml"])),
            )
            rows.append({
                "name": r["name"],
                "domain": r["domain"],
                "expr": r["expr"],
                "cost_class": r["cost_class"],
                "profile": profile,
            })
    return tuple(rows)


def corpus_size() -> int:
    """Number of corpus rows available for sibling search."""
    return len(_corpus_rows())


def corpus_domains() -> tuple[str, ...]:
    """Sorted tuple of distinct domain tags in the corpus."""
    return tuple(sorted({r["domain"] for r in _corpus_rows()}))


def find_siblings(
    expr: Union[sp.Expr, str],
    k: int = 5,
    *,
    domain: Optional[str] = None,
    max_distance: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
    exclude_self: bool = True,
) -> list[Sibling]:
    """Find the top-``k`` structural neighbours of ``expr`` in the corpus.

    Parameters
    ----------
    expr:
        SymPy expression or parseable string. The query is canonicalised
        via :meth:`PfaffianProfile.from_expression`.
    k:
        Number of neighbours to return (default 5).
    domain:
        If set, restrict the search to corpus rows tagged with this
        subdomain (e.g. ``"music_sound"``). Use :func:`corpus_domains`
        to enumerate available tags.
    max_distance:
        If set, return only siblings within this weighted-Euclidean
        distance of the query.
    weights:
        Optional override of :data:`DEFAULT_WEIGHTS` for the distance
        metric. Pass a dict with keys ``"r", "d", "w", "c"``.
    exclude_self:
        If ``True`` (default), drop corpus rows whose expression string
        exactly matches the query (after canonicalisation) — useful
        when querying with a known corpus member.

    Returns
    -------
    list[Sibling]
        Sorted by ascending distance; up to ``k`` entries.

    Raises
    ------
    sympy.SympifyError
        If ``expr`` cannot be parsed.

    Examples
    --------
    >>> from eml_cost import find_siblings
    >>> sibs = find_siblings("A * cos(omega * t)", k=3)
    >>> all(isinstance(s.distance, float) for s in sibs)
    True
    """
    if isinstance(expr, str):
        parsed = sp.sympify(_safe_expr(expr), locals=_PERMISSIVE)
    else:
        parsed = expr

    # Profile the query with do_canonicalize=False so that the cost-
    # class axes match the convention used at corpus-build time. The
    # bundled corpus rows were profiled with do_canonicalize=False
    # because canonicalize() takes exponential time on a small subset
    # of authored expressions; uniform handling is more important than
    # canonicalisation gains for the sibling search.
    query_profile = PfaffianProfile.from_expression(
        parsed, do_canonicalize=False
    )
    query_str = query_profile.canonical_form
    w = weights or DEFAULT_WEIGHTS

    candidates = _corpus_rows()
    if domain is not None:
        candidates = tuple(r for r in candidates if r["domain"] == domain)

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in candidates:
        if exclude_self and row["expr"] == query_str:
            continue
        d = query_profile.distance(row["profile"], weights=w)
        if max_distance is not None and d > max_distance:
            continue
        scored.append((d, row))

    scored.sort(key=lambda kv: kv[0])
    out: list[Sibling] = []
    for d, row in scored[:k]:
        out.append(Sibling(
            name=row["name"],
            domain=row["domain"],
            expression=row["expr"],
            cost_class=row["cost_class"],
            distance=d,
            profile=row["profile"],
        ))
    return out
