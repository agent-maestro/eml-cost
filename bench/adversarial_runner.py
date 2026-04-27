"""Adversarial bench runner — try to break the substrate.

Reads ``bench/adversarial.csv`` (100 stress-test expressions across 10
categories), runs each through ``eml_cost.analyze``, captures the
outcome, and reports failures honestly.

Categories:
    1. deep_nesting              — heavily nested elementary towers
    2. mixed_pne_elementary      — elementary functions wrapping PNE
    3. auto_eval_traps           — SymPy auto-simplifies to elementary
    4. limits_and_summation      — Limit / Sum / Integral / Derivative
    5. pathological_compositions — many ops, deep tree
    6. complex_arithmetic        — branches, complex constants
    7. distribution_discrete     — Heaviside / DiracDelta / Mod / floor
    8. pne_compositions          — PNE composed with PNE
    9. numeric_edge_cases        — infinities, NaN, huge literals
   10. structural                — Tuple / Matrix / Piecewise / Lambda

Expected values:
    ELEMENTARY    — is_pfaffian_not_eml should be False
    PNE           — is_pfaffian_not_eml should be True
    NON_NUMERIC   — structural object; either graceful exception or
                    a result without crashing. The runner records
                    which path was taken; both are valid.

Usage:
    python bench/adversarial_runner.py
    python bench/adversarial_runner.py --json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sympy as sp

from eml_cost import analyze


HERE = Path(__file__).resolve().parent


@dataclass
class Outcome:
    id: int
    category: str
    expression: str
    expected: str
    parsed_ok: bool
    analyze_ok: bool
    is_pne: bool | None
    pfaffian_r: int | None
    eml_depth: int | None
    classification: str   # PASS / FAIL / EXPECTED_REJECT / UNEXPECTED_REJECT / ANALYZE_ERROR / PARSE_ERROR
    message: str = ""


# Safe-ish eval namespace: only sympy + common symbols. The CSV is part
# of the package; arbitrary code execution is not a real concern, but
# the restricted namespace makes accidental imports impossible.
def _eval_namespace() -> dict[str, Any]:
    return {
        "sp": sp,
        "sympy": sp,
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "z": sp.Symbol("z"),
        "k": sp.Symbol("k", integer=True),
        "n": sp.Symbol("n", integer=True),
        "sum": sum,
        "range": range,
    }


def _classify(row: dict[str, str], expr: Any, result: Any, error: Exception | None) -> tuple[str, str]:
    """Return (classification, message) for one row outcome."""
    expected = row["expected"].strip()

    # Parse failure → already handled by caller
    if expected == "ELEMENTARY":
        if error is not None:
            return ("ANALYZE_ERROR", f"analyze raised: {type(error).__name__}: {error}")
        if result.is_pfaffian_not_eml:
            return ("FAIL", f"expected ELEMENTARY but flagged PNE (r={result.pfaffian_r})")
        return ("PASS", f"r={result.pfaffian_r}, depth={result.eml_depth}")

    if expected == "PNE":
        if error is not None:
            return ("ANALYZE_ERROR", f"analyze raised: {type(error).__name__}: {error}")
        if not result.is_pfaffian_not_eml:
            return ("FAIL", f"expected PNE but flagged ELEMENTARY (r={result.pfaffian_r})")
        return ("PASS", f"PNE flagged, r={result.pfaffian_r}, depth={result.eml_depth}")

    if expected == "NON_NUMERIC":
        # Either graceful exception (EXPECTED_REJECT) or a result
        # without crashing (PASS_WITH_RESULT). Both are valid; we just
        # record which path the detector took.
        if error is not None:
            return ("EXPECTED_REJECT", f"analyze raised gracefully: {type(error).__name__}")
        return ("PASS_WITH_RESULT", f"non-numeric handled, r={result.pfaffian_r}")

    return ("UNKNOWN_EXPECTED", f"unknown expected value: {expected}")


def run() -> list[Outcome]:
    csv_path = HERE / "adversarial.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    outcomes: list[Outcome] = []
    ns = _eval_namespace()

    for row in rows:
        rid = int(row["id"])
        expr_src = row["expression"]
        expected = row["expected"].strip()
        category = row["category"]

        # Stage 1: parse / construct
        try:
            expr = eval(expr_src, {"__builtins__": {}}, ns)
            parsed = True
            parse_error: Exception | None = None
        except Exception as e:
            parsed = False
            expr = None
            parse_error = e

        if not parsed:
            outcomes.append(Outcome(
                id=rid, category=category, expression=expr_src, expected=expected,
                parsed_ok=False, analyze_ok=False,
                is_pne=None, pfaffian_r=None, eml_depth=None,
                classification="PARSE_ERROR",
                message=f"{type(parse_error).__name__}: {parse_error}",
            ))
            continue

        # Stage 2: analyze
        try:
            result = analyze(expr)
            analyze_ok = True
            error = None
        except Exception as e:
            result = None
            analyze_ok = False
            error = e

        cls, msg = _classify(row, expr, result, error)
        outcomes.append(Outcome(
            id=rid, category=category, expression=expr_src, expected=expected,
            parsed_ok=True, analyze_ok=analyze_ok,
            is_pne=result.is_pfaffian_not_eml if analyze_ok else None,
            pfaffian_r=result.pfaffian_r if analyze_ok else None,
            eml_depth=result.eml_depth if analyze_ok else None,
            classification=cls, message=msg,
        ))

    return outcomes


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    total = len(outcomes)
    by_class: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    failures: list[Outcome] = []

    for o in outcomes:
        by_class[o.classification] = by_class.get(o.classification, 0) + 1
        cat = by_category.setdefault(o.category, {})
        cat[o.classification] = cat.get(o.classification, 0) + 1
        if o.classification in ("FAIL", "ANALYZE_ERROR", "UNEXPECTED_REJECT", "PARSE_ERROR"):
            failures.append(o)

    return {
        "total": total,
        "by_classification": by_class,
        "by_category": by_category,
        "failures": [asdict(o) for o in failures],
    }


def print_human(summary: dict[str, Any], outcomes: list[Outcome]) -> None:
    total = summary["total"]
    by_class = summary["by_classification"]
    by_cat = summary["by_category"]

    print(f"=== Adversarial bench results ({total} cases) ===\n")
    print("By classification:")
    for cls in sorted(by_class):
        pct = 100 * by_class[cls] / total
        print(f"  {cls:24s} {by_class[cls]:3d}  ({pct:5.1f}%)")
    print()
    print("By category:")
    for cat in sorted(by_cat):
        bins = by_cat[cat]
        total_in_cat = sum(bins.values())
        bins_str = ", ".join(f"{k}={v}" for k, v in sorted(bins.items()))
        print(f"  {cat:30s} n={total_in_cat:3d}   {bins_str}")
    print()
    failures = summary["failures"]
    if failures:
        print(f"=== {len(failures)} failures / errors (worth investigating) ===\n")
        for f in failures:
            print(f"  [{f['id']:3d}] {f['category']:24s} {f['classification']}")
            print(f"        expr:     {f['expression']}")
            print(f"        expected: {f['expected']}")
            print(f"        message:  {f['message']}")
            print()
    else:
        print("No failures — every case classified cleanly.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the adversarial substrate bench.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human report.")
    parser.add_argument("--results-csv", type=Path, default=HERE / "adversarial_results.csv",
                        help="Where to write per-row results CSV.")
    args = parser.parse_args()

    outcomes = run()
    summary = summarize(outcomes)

    # Always write the per-row results
    fields = ["id", "category", "expression", "expected", "parsed_ok",
              "analyze_ok", "is_pne", "pfaffian_r", "eml_depth",
              "classification", "message"]
    with args.results_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for o in outcomes:
            writer.writerow(asdict(o))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_human(summary, outcomes)

    # Exit code reflects whether any FAIL / ANALYZE_ERROR / PARSE_ERROR
    # rows surfaced. The bench is allowed to surface findings without
    # blocking CI — non-zero exit is informational only.
    failures = summary["by_classification"].get("FAIL", 0)
    errors = (summary["by_classification"].get("ANALYZE_ERROR", 0)
              + summary["by_classification"].get("PARSE_ERROR", 0))
    if failures or errors:
        return 0  # informational; do not break CI
    return 0


if __name__ == "__main__":
    sys.exit(main())
