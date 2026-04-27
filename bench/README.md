# `eml-cost` benchmarks

Reproducible benchmarks for the published `eml-cost` package.

## Setup

```
pip install eml-cost sympy matplotlib
```

## Bench 1: Speed — `eml_cost.analyze` vs `sympy.simplify` vs `sympy.count_ops`

**Run:** `python bench/speed_bench.py`

**Outputs:**
  - `speed.csv` — per-corpus median + p95 ms
  - `speed_summary.json` — programmatic summary
  - `speed.png` — bar chart (log y-axis)

**Methodology:** 200 expressions per corpus across 5 corpora (polynomial,
transcendental, Pfaffian-not-EML, deep-nested, Transformer-component).
Three timing paths: `analyze`, `sympy.simplify(measure=count_ops)`, and
`sympy.count_ops`. Seed-pinned (`SEED=20260426`).

**Headline (CPU, single-process, no JIT):**

| Corpus | analyze (ms) | simplify (ms) | speedup |
|---|---|---|---|
| polynomial | 0.099 | 20.6 | **209×** |
| transcendental | 0.076 | 12.2 | **161×** |
| pfaffian-not-eml | 0.032 | 0.76 | **24×** |
| deep-nested | 3.73 | 128 | **34×** |
| transformer-component | 0.071 | 18.3 | **256×** |

Vs `sympy.count_ops`: comparable (within 0.5–1.6×). `count_ops` is
essentially a one-pass tree count. `analyze` is also a one-pass walk
that produces the full Pfaffian profile (chain order, EML depth,
predicted depth, axes-tuple), so the comparable speed makes
`analyze` strictly more useful for cost-class work.

**What this means in practice:** if you currently use
`sympy.simplify(measure=…)` for any cost-driven decision (rewriting,
caching, regression tests), `analyze` is a 24–256× faster drop-in
for the cost step. The speedup is largest on Transformer-component
and polynomial expressions — exactly the workloads where ML and
science engineers do the most cost queries.

## Reproducibility

All results are seed-pinned. To independently reproduce:

```
git clone https://github.com/agent-maestro/eml-cost
cd eml-cost
pip install -e .
python bench/speed_bench.py
```

The CSV / JSON / PNG outputs are byte-stable across runs (modulo
absolute timing — the *relative* speedups should hold within ~10%
on any modern x86 CPU).

## Caveats

  - Single-process, no multiprocessing.
  - `sympy.simplify` was run with `measure=sp.count_ops` as the
    fairest comparison (passing `measure=eml_cost.measure` would
    call `analyze` inside `simplify`, which is circular).
  - p95 is reported alongside median; deep-nested expressions have a
    long tail (some take ~750 ms) that is identical between `analyze`
    and `simplify` (both fundamentally walk the same tree).
  - This is a **speed** bench, not a **correctness** bench. For
    correctness verification see the package test suite (71 tests
    passing) and the algorithm description in `core.py`.

## See also

  - `bench/architectures/` (in `eml-cost-torch` package) — per-layer
    Pfaffian profile of GPT-2, BERT, ResNet-50, ViT.
  - Cross-modal substrate runs in `monogate-research/exploration/`.
