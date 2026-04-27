"""Build corpus_578.csv from monogate-research master_corpus_578.csv.

Precomputes PfaffianProfile fields for each row so find_siblings() can
load and compute distances without re-running analyze() on 578
expressions at runtime.

Run once at packaging time. Output is committed to
src/eml_cost/data/corpus_578.csv.

Skips rows that fail to parse (Python keyword collisions, lambda
function literals, etc.). Reports a summary at the end.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import sympy as sp

REPO = Path(__file__).resolve().parent.parent
SRC = REPO.parent / "monogate-research" / "exploration" / "E196_algorithmic_corpus" / "master_corpus_578.csv"
OUT = REPO / "src" / "eml_cost" / "data" / "corpus_578.csv"

if not SRC.exists():
    print(f"ERROR: source not found at {SRC}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(REPO / "src"))
from eml_cost.profile import PfaffianProfile  # noqa: E402

# Permissive symbol map: avoid collisions with sympy reserved names
# (zeta, I, N, E) and rename Python keyword `lambda` -> `lam`.
PERMISSIVE = {
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
    """Replace `lambda` with `lam` to avoid Python-keyword parse failure."""
    return re.sub(r"\blambda\b", "lam", s)


def main() -> None:
    rows_in = []
    with SRC.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows_in.append(r)

    out_rows = []
    skipped = []
    print(f"Processing {len(rows_in)} rows (no-canonicalize for speed)...",
          flush=True)
    for i, r in enumerate(rows_in):
        if i % 25 == 0:
            print(f"  row {i}/{len(rows_in)}: {r['label'][:50]}", flush=True)
        try:
            expr = sp.sympify(_safe_expr(r["expr"]), locals=PERMISSIVE)
            profile = PfaffianProfile.from_expression(
                expr, do_canonicalize=False
            )
        except (sp.SympifyError, ValueError, TypeError, RecursionError) as e:
            skipped.append((r["domain"], r["label"], str(e)[:80]))
            continue

        out_rows.append({
            "domain": r["domain"],
            "name": r["label"],
            "expr": r["expr"],
            "cost_class": profile.cost_class,
            "r": profile.r,
            "degree": profile.degree,
            "width": profile.width,
            "c_osc": profile.corrections["c_osc"],
            "c_composite": profile.corrections["c_composite"],
            "delta_fused": profile.corrections["delta_fused"],
            "is_pfaffian_not_eml": int(profile.is_pfaffian_not_eml),
            "source": r.get("source", ""),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = list(out_rows[0].keys())
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"In:      {len(rows_in)} rows")
    print(f"Out:     {len(out_rows)} rows -> {OUT}")
    print(f"Skipped: {len(skipped)} rows")
    for s in skipped:
        print(f"   {s[0]:<14} {s[1][:40]:<40} {s[2]}")


if __name__ == "__main__":
    main()
