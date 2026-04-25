# eml-cost

**Pre-release alpha. Patent pending.** Source-available; see LICENSE.

Pfaffian chain order and EML routing depth for symbolic expressions —
a programmatic complexity measure on SymPy expression trees.

## Installation

```bash
pip install --pre eml-cost
```

The `--pre` flag is required while we're on `0.1.0a0`. Once the first
stable release publishes, plain `pip install eml-cost` will work.

For local development:

```bash
git clone https://github.com/almaguer1986/eml-cost
cd eml-cost
pip install -e ".[dev]"
pytest
```

## Quick start

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
- Source: [github.com/almaguer1986/eml-cost](https://github.com/almaguer1986/eml-cost)
- Package: [pypi.org/project/eml-cost](https://pypi.org/project/eml-cost/)

## License

`PROPRIETARY-PRE-RELEASE`. See LICENSE.

## Citation

Citation form will be locked at public release.
