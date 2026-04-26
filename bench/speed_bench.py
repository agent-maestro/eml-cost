"""Speed benchmark — eml-cost.analyze vs sympy.simplify(measure=).

Measures median + p95 ms per expression across 5 corpora (200 each):
  - polynomials (constant chain order = 0)
  - transcendental (single chain element)
  - Pfaffian-not-EML (Bessel, erf, Gamma)
  - deep-nested (e.g. exp(exp(exp(x))))
  - Transformer-component-shaped (sigmoid, GELU, LayerNorm, attention)

Output: bench/speed.csv  + bench/speed_summary.json + bench/speed.png

Reproducer: pip install eml-cost sympy matplotlib, then python this file.
Seed-pinned so two runs produce the same expressions and comparable
medians.

Note: sympy.simplify is run with measure=count_ops as the standard
baseline. We do NOT pass measure=eml_cost.measure (that would call
analyze inside simplify, which is circular). The fair comparison is
"two paths to a cost number: full simplify vs analyze-only."
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import time
from pathlib import Path

import sympy as sp

from eml_cost import analyze


HERE = Path(__file__).parent
SEED = 20260426
N_PER_CORPUS = 200


# ---------------------------------------------------------------------------
# 5 corpora
# ---------------------------------------------------------------------------
def gen_polynomial(rng):
    x, y, z = sp.symbols("x y z")
    syms = [x, y, z]
    n_terms = rng.randint(2, 6)
    expr = 0
    for _ in range(n_terms):
        c = rng.randint(-5, 5)
        if c == 0:
            c = 1
        s = rng.choice(syms)
        p = rng.randint(1, 4)
        expr = expr + c * s ** p
    return expr


def gen_transcendental(rng):
    x = sp.Symbol("x")
    funcs = [sp.exp, sp.log, sp.sin, sp.cos, sp.tanh, sp.sqrt]
    f = rng.choice(funcs)
    inner = x * rng.randint(1, 5) + rng.randint(-3, 3)
    if f is sp.log or f is sp.sqrt:
        inner = sp.Abs(inner) + sp.Rational(1, 100)
    return f(inner)


def gen_pfaffian_not_eml(rng):
    x = sp.Symbol("x", positive=True)
    funcs = [sp.besselj, sp.bessely, sp.airyai, sp.LambertW,
             sp.erf, sp.erfc, sp.gamma, sp.loggamma, sp.polylog,
             sp.zeta, sp.elliptic_k, sp.Ei, sp.Si]
    f = rng.choice(funcs)
    if f in (sp.besselj, sp.bessely):
        return f(rng.randint(0, 3), x)
    if f is sp.polylog:
        return f(rng.randint(2, 4), x)
    return f(x)


def gen_deep_nested(rng, depth=None):
    x = sp.Symbol("x")
    if depth is None:
        depth = rng.randint(3, 6)
    expr = x
    for _ in range(depth):
        op = rng.choice(["exp", "log", "sin", "tanh"])
        if op == "exp":
            expr = sp.exp(expr)
        elif op == "log":
            expr = sp.log(sp.Abs(expr) + 1)
        elif op == "sin":
            expr = sp.sin(expr)
        else:
            expr = sp.tanh(expr)
    return expr


def gen_transformer_component(rng):
    """Sigmoid, GELU, LayerNorm, attention softmax, etc."""
    x = sp.Symbol("x")
    y = sp.Symbol("y", positive=True)
    pick = rng.choice([
        "sigmoid", "gelu_approx", "tanh_act", "layernorm_axis",
        "softmax_one", "rmsnorm", "swish", "scaled_dot",
    ])
    if pick == "sigmoid":
        return 1 / (1 + sp.exp(-x))
    if pick == "gelu_approx":
        return sp.S.Half * x * (1 + sp.tanh(sp.sqrt(2 / sp.pi) * (x + sp.Rational(447, 10000) * x ** 3)))
    if pick == "tanh_act":
        return sp.tanh(x)
    if pick == "layernorm_axis":
        mu = sp.Symbol("mu")
        sigma = sp.Symbol("sigma", positive=True)
        eps = sp.Symbol("eps", positive=True)
        return (x - mu) / sp.sqrt(sigma ** 2 + eps)
    if pick == "softmax_one":
        return sp.exp(x) / (sp.exp(x) + sp.exp(y))
    if pick == "rmsnorm":
        ms = sp.Symbol("ms", positive=True)
        eps = sp.Symbol("eps", positive=True)
        return x / sp.sqrt(ms + eps)
    if pick == "swish":
        return x * (1 / (1 + sp.exp(-x)))
    # scaled_dot
    d = sp.Symbol("d", positive=True)
    return x * y / sp.sqrt(d)


CORPORA = [
    ("polynomial", gen_polynomial),
    ("transcendental", gen_transcendental),
    ("pfaffian-not-eml", gen_pfaffian_not_eml),
    ("deep-nested", gen_deep_nested),
    ("transformer-component", gen_transformer_component),
]


def build_corpora():
    rng = random.Random(SEED)
    corpora = {}
    for name, gen in CORPORA:
        exprs = []
        seen = set()
        attempts = 0
        while len(exprs) < N_PER_CORPUS and attempts < N_PER_CORPUS * 5:
            attempts += 1
            try:
                e = gen(rng)
                s = str(e)
                if s in seen or len(s) < 3 or len(s) > 400:
                    continue
                seen.add(s)
                exprs.append(e)
            except Exception:
                continue
        corpora[name] = exprs
    return corpora


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------
def bench(fn, exprs):
    """Return list of per-expression elapsed milliseconds."""
    times_ms = []
    for e in exprs:
        t0 = time.perf_counter()
        try:
            fn(e)
        except Exception:
            pass
        elapsed = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed)
    return times_ms


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("eml-cost speed benchmark — analyze vs sympy.simplify")
    print("=" * 72)
    print(f"seed={SEED}, N per corpus = {N_PER_CORPUS}")
    print()

    corpora = build_corpora()
    for name, exprs in corpora.items():
        print(f"  {name}: {len(exprs)} expressions")
    print()

    rows = []
    for name, exprs in corpora.items():
        print(f"Benchmarking {name}...")

        # eml-cost.analyze
        analyze_times = bench(analyze, exprs)

        # sympy.simplify (count_ops baseline)
        def simplify_with_count(e):
            return sp.simplify(e, measure=sp.count_ops)
        simplify_times = bench(simplify_with_count, exprs)

        # sympy.count_ops alone (fastest sympy "cost" path; not a full simplify)
        count_ops_times = bench(sp.count_ops, exprs)

        analyze_med = statistics.median(analyze_times)
        analyze_p95 = percentile(analyze_times, 95)
        simplify_med = statistics.median(simplify_times)
        simplify_p95 = percentile(simplify_times, 95)
        count_med = statistics.median(count_ops_times)
        count_p95 = percentile(count_ops_times, 95)

        speedup_vs_simplify = simplify_med / analyze_med if analyze_med > 0 else float("inf")
        ratio_vs_count_ops = count_med / analyze_med if analyze_med > 0 else float("inf")

        print(f"  analyze:                med={analyze_med:.3f} ms, p95={analyze_p95:.3f} ms")
        print(f"  sympy.simplify:         med={simplify_med:.3f} ms, p95={simplify_p95:.3f} ms")
        print(f"  sympy.count_ops:        med={count_med:.3f} ms, p95={count_p95:.3f} ms")
        print(f"  speedup vs simplify:    {speedup_vs_simplify:.1f}x median")
        print(f"  speedup vs count_ops:   {ratio_vs_count_ops:.2f}x median (smaller is closer)")
        print()

        rows.append({
            "corpus": name,
            "n": len(exprs),
            "analyze_med_ms": round(analyze_med, 3),
            "analyze_p95_ms": round(analyze_p95, 3),
            "simplify_med_ms": round(simplify_med, 3),
            "simplify_p95_ms": round(simplify_p95, 3),
            "count_ops_med_ms": round(count_med, 3),
            "count_ops_p95_ms": round(count_p95, 3),
            "speedup_vs_simplify_med": round(speedup_vs_simplify, 2),
            "ratio_vs_count_ops_med": round(ratio_vs_count_ops, 2),
        })

    # CSV
    out_csv = HERE / "speed.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv.name}")

    # JSON summary
    out_json = HERE / "speed_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n_per_corpus": N_PER_CORPUS,
            "rows": rows,
            "headline": {
                "all_corpora_median_speedup_vs_simplify": round(
                    statistics.median(r["speedup_vs_simplify_med"] for r in rows), 1
                ),
            },
        }, f, indent=2)
    print(f"Wrote {out_json.name}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["corpus"] for r in rows]
        analyze_meds = [r["analyze_med_ms"] for r in rows]
        simplify_meds = [r["simplify_med_ms"] for r in rows]
        count_meds = [r["count_ops_med_ms"] for r in rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(names)))
        w = 0.27
        ax.bar([i - w for i in x], analyze_meds, w,
               label="eml_cost.analyze", color="#d4a76a")
        ax.bar(x, count_meds, w, label="sympy.count_ops", color="#7e9bd1")
        ax.bar([i + w for i in x], simplify_meds, w,
               label="sympy.simplify", color="#e07070")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Median ms / expression")
        ax.set_yscale("log")
        ax.set_title(f"eml-cost speed benchmark (n={N_PER_CORPUS} per corpus)")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        plt.savefig(HERE / "speed.png", dpi=120, bbox_inches="tight")
        print(f"Wrote speed.png")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
