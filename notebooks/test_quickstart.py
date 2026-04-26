"""Headless equivalent of quickstart.ipynb for end-to-end verification.

Runs every cell's logic. If this script passes, the notebook works.
"""
from __future__ import annotations

import csv
from importlib.resources import files
from collections import Counter

import numpy as np
import sympy as sp

import eml_cost
from eml_cost import PfaffianProfile, analyze, canonicalize


print("=" * 70)
print("Cell 1+2: import + first analysis")
print("=" * 70)
print("eml-cost version:", eml_cost.__version__)

x = sp.Symbol("x")
for expr in [sp.exp(x), sp.sin(x), x**2 + 3*x + 1, sp.erf(x)]:
    p = PfaffianProfile.from_expression(expr)
    elem = "EML-elementary" if p.is_elementary() else "Pfaffian-not-EML"
    print(f"  {str(expr):<25s}  {p.cost_class:<14s}  {elem}")


print("\n" + "=" * 70)
print("Cell 3: canonicalize on sigmoid forms")
print("=" * 70)
forms = [
    1 / (1 + sp.exp(-x)),
    sp.exp(x) / (sp.exp(x) + 1),
    sp.Rational(1, 2) * (1 + sp.tanh(x / 2)),
    1 - 1 / (1 + sp.exp(x)),
]
print("Without canonicalize:")
classes_raw = set()
for f in forms:
    p = PfaffianProfile.from_expression(f, do_canonicalize=False)
    classes_raw.add(p.cost_class)
    print(f"  {str(f):<35s} -> {p.cost_class}")
print(f"  unique classes: {len(classes_raw)}")
print()
print("With canonicalize (default):")
classes_canon = set()
for f in forms:
    p = PfaffianProfile.from_expression(f)
    classes_canon.add(p.cost_class)
    print(f"  {str(f):<35s} -> {p.cost_class}")
print(f"  unique classes: {len(classes_canon)}")
print(f"  canonicalize reduces drift: {len(classes_raw)} -> {len(classes_canon)}")


print("\n" + "=" * 70)
print("Cell 4: distance metric")
print("=" * 70)
logistic_growth = 1 / (1 + sp.exp(-x))
ml_sigmoid = sp.exp(x) / (sp.exp(x) + 1)
p1 = PfaffianProfile.from_expression(logistic_growth)
p2 = PfaffianProfile.from_expression(ml_sigmoid)
print(f"logistic growth:     {p1.cost_class}")
print(f"ML sigmoid:          {p2.cost_class}")
print(f"Distance:            {p1.distance(p2):.4f}")
print(f"Compare:             {p1.compare(p2)}")


print("\n" + "=" * 70)
print("Cell 5: load + profile demo corpus")
print("=" * 70)
corpus_path = files("eml_cost").joinpath("data/demo_corpus.csv")
with open(corpus_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows)} expressions across "
      f"{len(set(r['domain'] for r in rows))} domains")

profiles = []
failed = []
for r in rows:
    try:
        p = PfaffianProfile.from_expression(r["sympy_expr"])
        profiles.append({**r, "profile": p})
    except Exception as e:
        failed.append((r["name"], str(e)[:80]))
print(f"Successfully profiled: {len(profiles)} / {len(rows)}")
if failed:
    print(f"Failed expressions:")
    for name, err in failed[:5]:
        print(f"  {name}: {err}")

class_counts = Counter(p["profile"].cost_class for p in profiles)
print("Top cost classes:")
for cls, n in class_counts.most_common(10):
    print(f"  {cls:<16s} x{n}")


print("\n" + "=" * 70)
print("Cell 6: distance matrix on the demo corpus")
print("=" * 70)
n = len(profiles)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = profiles[i]["profile"].distance(profiles[j]["profile"])
        D[i, j] = d
        D[j, i] = d
print(f"Distance matrix: {n}x{n}")
print(f"  range: {D.min():.3f} to {D.max():.3f}")
print(f"  mean:  {D.mean():.3f}")
print(f"  median (off-diag): {np.median(D[D > 0]):.3f}")


print("\n" + "=" * 70)
print("ALL NOTEBOOK CELLS EXECUTABLE")
print("=" * 70)
