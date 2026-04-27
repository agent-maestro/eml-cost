# eml-cost

**Stable beta. Patent pending.** Source-available; see LICENSE.

Pfaffian chain order and EML routing depth for symbolic expressions —
a programmatic complexity measure on SymPy expression trees.

## Installation

```bash
pip install eml-cost
```

For local development:

```bash
git clone https://github.com/agent-maestro/eml-cost
cd eml-cost
pip install -e ".[dev]"
pytest
```

## Quick start

Three things you can do in under 10 lines each.

### 1. Get a complexity profile for any expression

```python
from eml_cost import analyze

result = analyze("exp(exp(x)) + sin(x**2)")
print(result.pfaffian_r, result.max_path_r, result.predicted_depth)
# 5 5 7
```

### 2. Plug into SymPy's simplify as a cost function

```python
import sympy as sp
from eml_cost import measure

x = sp.Symbol("x", real=True)
sp.simplify(sp.cos(x)**2 + sp.sin(x)**2, measure=measure)
# 1
```

### 3. Detect Pfaffian-but-not-EML expressions (Bessel, Airy, Lambert W)

```python
import sympy as sp
from eml_cost import is_pfaffian_not_eml

is_pfaffian_not_eml(sp.besselj(0, sp.Symbol("x")))   # True
is_pfaffian_not_eml(sp.exp(sp.Symbol("x")))          # False
```

### 4. Canonicalize before profiling (eliminate form-fragility)

50% of textbook expressions yield different cost classes when written in
algebraically equivalent forms. `canonicalize()` is a curated, content-
preserving rewrite-rule sequence that drops drift to ~35% in our audit.

```python
import sympy as sp
from eml_cost import PfaffianProfile

x = sp.Symbol("x")
forms = [
    1 / (1 + sp.exp(-x)),
    sp.exp(x) / (sp.exp(x) + 1),
    1 - 1 / (1 + sp.exp(x)),
]
for f in forms:
    p = PfaffianProfile.from_expression(f)  # canonicalize=True is default
    print(f"{f}  ->  {p.cost_class}")
# All three collapse to the same cost class.
```

### 5. Compare two expressions with a real distance metric

```python
from eml_cost import PfaffianProfile

a = PfaffianProfile.from_expression("exp(x)")
b = PfaffianProfile.from_expression("sin(x)")
a.distance(b)        # weighted Euclidean in (r, d, w, c) space
a.compare(b)         # per-axis deltas + same_class flag
a.is_elementary()    # True (not Pfaffian-not-EML)
```

The metric satisfies identity, symmetry, and the triangle inequality
(verified in `tests/test_profile_metric.py`). Default weights:
`r=4, d=1, w=2, c=1` — chain order dominates.

### 6. Run on the bundled 50-expression cross-domain corpus

```python
import csv
from importlib.resources import files
from eml_cost import PfaffianProfile

corpus_path = files("eml_cost").joinpath("data/demo_corpus.csv")
with open(corpus_path) as f:
    rows = list(csv.DictReader(f))
profiles = [PfaffianProfile.from_expression(r["sympy_expr"]) for r in rows]

# 50 expressions, 9 domains (polynomial, exp_log, trig, pfaffian_not_eml,
# ml_activation, physics, biology, engineering, random_null), all with
# citations.
```

For an interactive walk-through with plots, see
[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).

## Command-line interface (0.7.1+)

`eml-cost` ships with a console command that exposes the prediction
trilogy at the shell:

```bash
# Pretty report for one or more expressions
eml-cost report "exp(exp(x))" "sin(x**2)"

# JSON for machine consumers
eml-cost report --json "exp(x) + log(x+1)"

# Read one expression per line (use - for stdin; # comments allowed)
eml-cost report --file expressions.txt

# Pre-commit gate: exits non-zero if any budget is exceeded
eml-cost check "exp(exp(x)) + sin(x**2)" \
  --max-simplify-ms 200 \
  --max-digits-lost 3
```

What `report` shows for each expression:
  - Pfaffian profile: `pfaffian_r`, `max_path_r`, `eml_depth`,
    `predicted_depth`, PNE flag
  - `estimate_time` per proxy: simplify / factor / cse / lambdify
    predicted ms with 95% CI
  - `predict_precision_loss`: predicted relerr, predicted decimal
    digits lost, 95% CI

Exit codes for `check`:

| code | meaning |
|------|---------|
| 0    | all checks passed |
| 1    | at least one threshold violation (`check` only) |
| 2    | at least one expression failed to parse |
| 64   | usage error (no expressions given) |

### Pre-commit hook recipe

Block any commit that introduces a SymPy expression exceeding your
team's compile-time or numerical-precision budgets. Drop a list of
the expressions you care about into a tracked file (one per line,
`#` comments allowed), then add the hook to your repo's
`.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: eml-cost
      name: eml-cost numerical / compile-cost gate
      entry: eml-cost check --file
      language: system
      files: ^expressions\.txt$
      args: [--max-simplify-ms, "200", --max-digits-lost, "3"]
```

`expressions.txt` contents:

```
# Each line is one SymPy-parseable expression.
# Blank lines and # comments are skipped.
# This file is the authoritative list of expressions the team has
# committed to keeping under the budget below.

exp(exp(x)) + sin(x**2)
1/(1 + exp(-x))
log(x + sqrt(x**2 + 1))

# Add more as you go. Removing an expression is the only way to
# legitimately drop a budget breach below the threshold.
```

Both files live alongside the rest of your repo. The hook runs
locally on every commit (and in CI if you use `pre-commit run --all-files`
in the workflow). A complete working sample lives at
[`examples/.pre-commit-config.example.yaml`](examples/.pre-commit-config.example.yaml).

### CI integration (GitHub Actions, etc.)

Same `eml-cost check` invocation works in any CI:

```yaml
# .github/workflows/eml-cost.yml
name: eml-cost
on: [push, pull_request]
jobs:
  budget:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install eml-cost
      - run: eml-cost check --file expressions.txt --max-simplify-ms 200 --max-digits-lost 3
```

The check is deterministic — same expression file + same `eml-cost`
version produces the same exit code on every run.

## Result shape

```python
from eml_cost import analyze

result = analyze("exp(exp(x)) + sin(x**2)")

result.pfaffian_r           # total Pfaffian chain order
result.max_path_r           # chain order along the deepest path
result.eml_depth            # EML routing tree depth
result.structural_overhead  # tree-structural depth
result.corrections          # Corrections(c_osc, c_composite, delta_fused)
result.predicted_depth      # max_path_r + corrections + structural
result.is_pfaffian_not_eml  # True for Bessel, Airy, Lambert W, ...
```

Drop-in measure for SymPy's `simplify`:

```python
import sympy as sp
from eml_cost import measure

x = sp.Symbol("x", real=True)
sp.simplify(sp.cos(x)**2 + sp.sin(x)**2, measure=measure)
# 1
```

## Public API

```python
from eml_cost import (
    analyze,                # main entry point
    measure,                # SymPy simplify(..., measure=...) helper
    AnalyzeResult,          # frozen dataclass (result type)
    Corrections,            # frozen dataclass (correction terms)
    pfaffian_r,             # total chain order
    max_path_r,             # path-restricted chain order
    eml_depth,              # routing tree depth
    structural_overhead,    # Add/Mul/poly-Pow tree depth
    is_pfaffian_not_eml,    # True for Bessel/Airy/Lambert W/hyper
    PFAFFIAN_NOT_EML_R,     # registry: name -> chain order
)
```

## What gets counted

Khovanskii r-counting throughout:

| Operator | Chain contribution |
|---|---|
| `exp(g)` | 1 |
| `log(g)` | 1 |
| `sin(g)`, `cos(g)` (pair) | 2 |
| `tan(g)` | 1 |
| `tanh`, `atan`, `atanh`, `asinh`, `acosh` | 1 each |
| `sinh(g)`, `cosh(g)` (pair) | 2 |
| `sqrt(g)`, `Pow(g, non-integer)` | 1 |
| `Pow(g, integer)`, `Add`, `Mul` | 0 |
| Bessel J/Y/I/K, Airy Ai/Bi, Lambert W, hyper | per registry |

`max_path_r` differs from `pfaffian_r` only at `Add` and `Mul` nodes:
`pfaffian_r` sums children, `max_path_r` takes the max. For independent-
variable products like atomic orbital wavefunctions
(`R(r) * Y(theta) * Phi(phi)`), the path-restricted count is dramatically
smaller than the total — capturing the parallel-composition behavior.

## EML routing depth

The `eml_depth` function models SuperBEST routing:

| Operator | Depth contribution |
|---|---|
| `exp`, `log` | 1 |
| `sin`, `cos` | 3 (Euler bypass) |
| `tan` | 4 |
| `tanh`, `atan`, `sinh`, `cosh` | 1 (F-family primitive) |
| `Pow`, `Add`, `Mul` | 1 + max over children |

F-family fusion patterns are recognized:

- `log(c + exp(g))` (LEAd / softplus shape) -> depth 1 + depth(g)
- `1/(1 + exp(-g))` (sigmoid shape) -> depth 1 + depth(g)

## Pfaffian-but-not-EML class

Bessel J/Y/I/K, Hankel, Airy Ai/Bi, hypergeometric, and Lambert W are
Pfaffian (admit polynomial-coefficient ODE chains) but lie outside the
EML-elementary class. They are flagged by `is_pfaffian_not_eml(expr)`
and contribute their registered chain order under `pfaffian_r`.

## Links

- Project home: [monogate.org](https://monogate.org)
- Source: [github.com/agent-maestro/eml-cost](https://github.com/agent-maestro/eml-cost)
- Package: [pypi.org/project/eml-cost](https://pypi.org/project/eml-cost/)

## License

`PROPRIETARY-PRE-RELEASE`. See LICENSE.

## Citation

Citation form will be locked at public release.
