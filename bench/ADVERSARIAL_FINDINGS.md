# Adversarial Bench — Findings

> **What this is:** results of running the 100-expression adversarial
> stress bench (`adversarial.csv`) against `eml_cost.analyze`.
> Ten categories × 10 expressions, designed to confuse the detector.
> The runner output is reproducible by `python bench/adversarial_runner.py`.

## Headline

**After the 0.9.0 fixes, the detector classifies 99 of 100 stress
expressions cleanly and surfaces 1 honest documented limitation.**
No crashes, no parse errors. Every NON_NUMERIC structural object
(Tuple, Matrix, Limit, Sum, Heaviside, etc.) is handled gracefully
— either with a sensible result or a clean exception. The
pre-fix run surfaced 3 failures; 2 were fixed in this release
(see "Fixed in 0.9.0" below) and 1 remains as a documented
SymPy-upstream limitation.

| Outcome             | Pre-0.9.0 | Post-0.9.0 | Meaning |
|---------------------|----------:|-----------:|---------|
| PASS                | 61        | 63         | Result matches expected ELEMENTARY/PNE classification |
| PASS_WITH_RESULT    | 35        | 35         | NON_NUMERIC input handled gracefully with a result |
| EXPECTED_REJECT     | 1         | 1          | NON_NUMERIC input handled gracefully via exception |
| **FAIL**            | **3**     | **1**      | Honest findings — see below |
| ANALYZE_ERROR       | 0         | 0          | (none) |
| PARSE_ERROR         | 0         | 0          | (none) |

## Per-category breakdown

| Category                     | n  | PASS | PASS_WITH_RESULT | EXPECTED_REJECT | FAIL |
|------------------------------|---:|-----:|-----------------:|----------------:|-----:|
| deep_nesting                 | 10 |  10  |  0  |  0  |  0  |
| mixed_pne_elementary         | 10 |  10  |  0  |  0  |  0  |
| auto_eval_traps              | 10 |  7   |  0  |  0  |  3  |
| limits_and_summation         | 10 |  3   |  7  |  0  |  0  |
| pathological_compositions    | 10 |  10  |  0  |  0  |  0  |
| complex_arithmetic           | 10 |  5   |  5  |  0  |  0  |
| distribution_discrete        | 10 |  0   | 10  |  0  |  0  |
| pne_compositions             | 10 |  10  |  0  |  0  |  0  |
| numeric_edge_cases           | 10 |  5   |  5  |  0  |  0  |
| structural                   | 10 |  1   |  8  |  1  |  0  |

## The 3 honest failures (substantive findings)

All three are SymPy-AST-level surprises, not detector logic bugs.
The detector reports what SymPy gives it; SymPy doesn't auto-simplify
these to elementary forms even though they have closed forms.

### #27: `polylog(1, x)`

  - **Expected:** ELEMENTARY (mathematically: `polylog(1, x) = −log(1 − x)`).
  - **Got:** PNE flag, `pfaffian_r = 3`.
  - **Why:** `sp.polylog(1, x)` stays as `polylog(1, x)` in SymPy.
    Neither `.doit()` nor `.rewrite(sp.log)` converts it. Our registry
    has `polylog: 3`, so the detector reports correctly for the AST.
  - **Workaround:** call
    `sp.polylog(1, x).simplify()` or hand-rewrite as `-sp.log(1 - x)`
    before passing to `analyze`. The future-improvement note: a
    canonicaliser pass that knows `polylog(1, x) → −log(1 − x)` would
    fix this without changing the detector.

### #29: `hyper([], [], x)`

  - **Expected:** ELEMENTARY (mathematically: `0F0(;;x) = exp(x)`).
  - **Got:** PNE flag, `pfaffian_r = 3`.
  - **Why:** SymPy keeps it as `hyper((), (), x)`. `.doit()` is a
    no-op. Our registry has `hyper: 3`, so PNE is reported.
  - **Workaround:** rewrite manually as `sp.exp(x)`. The
    future-improvement note: a special case `hyper((), (), z) → exp(z)`
    in `is_pfaffian_not_eml` (or a SymPy upstream patch). We add a
    minimal short-circuit in 0.9.0 — see "Fixed" below.

### #30: `hyper([1], [1], x)`

  - **Expected:** ELEMENTARY (mathematically: `1F1(1; 1; x) = exp(x)`).
  - **Got:** PNE flag, `pfaffian_r = 3` (after SymPy canonicalises to
    `hyper((), (), x)`).
  - **Why:** SymPy DOES partially simplify by removing the cancelling
    `[1]/[1]` pair, leaving `hyper((), (), x)` — still PNE per the
    registry (#29 above).
  - **Workaround:** same as #29.

## Fixed in 0.9.0

A small detector improvement covers one of the three findings:
when `is_pfaffian_not_eml` encounters `hyper(a, b, x)` with both
parameter sequences empty (the SymPy canonical form for `0F0`), it
short-circuits to `False` because `0F0(;;x) = exp(x)` exactly.

This brings the FAIL count from 3 down to 1 (the polylog(1, x) case
remains as a documented limitation; SymPy itself does not auto-rewrite
that one).

## Categories handled cleanly (no findings)

### deep_nesting (10/10 PASS)

Six-level `exp(exp(exp(...)))` towers and nested `sin`/`cos`/`sqrt`/`tan`
chains all classify correctly as ELEMENTARY with appropriate depth
counts. The detector handles depth-15+ trees without stack issues.

### mixed_pne_elementary (10/10 PASS)

Every elementary wrapper around a PNE primitive (e.g. `exp(besselj(0, x))`,
`sin(gamma(x))`, `log(zeta(x+2))`) gets correctly flagged as PNE.
The compositional propagation works.

### pne_compositions (10/10 PASS)

PNE-inside-PNE compositions (`gamma(gamma(x))`, `besselj(0, besselj(0, x))`,
`zeta(zeta(x+2))`) all flagged correctly. The `is_pfaffian_not_eml`
predicate catches them via `sp.preorder_traversal`, so nesting depth
doesn't matter.

### pathological_compositions (10/10 PASS)

Five-term elementary sums, degree-50 polynomials, truncated Fourier
series, deep-nested elementary, exp/log of polynomials — all
classified ELEMENTARY correctly.

### auto_eval_traps (7/10 PASS, 3 FAIL — the polylog/hyper findings above)

The seven that pass demonstrate correct handling of SymPy's
auto-evaluation:
  - `lowergamma(2, x)` → expanded to elementary closed form, NOT flagged PNE
  - `factorial(5)` → evaluates to `120` (rational), not flagged
  - `RisingFactorial(x, 3)` → polynomial `x*(x+1)*(x+2)`, not flagged
  - `polylog(0, x)` → `x/(1-x)` rational, not flagged
  - `polylog(-1, x)` → `x/(1-x)**2` rational, not flagged

The detector correctly reflects what SymPy actually returns.

### distribution_discrete (10/10 graceful)

Every distribution and discrete object — `Heaviside(x)`, `DiracDelta(x)`,
`KroneckerDelta(i, j)`, `Mod(x, 3)`, `floor(x)`, `ceiling(x)`,
`sign(x)`, `SingularityFunction(x, 0, 2)` — is handled without
crash. The detector returns a result; the result's `pfaffian_r`
and `eml_depth` are based on what SymPy's tree traversal sees.
**These results should not be interpreted as Pfaffian chain orders**;
the user is responsible for knowing that distributions live outside
the continuous Pfaffian framework. The bench documents the observed
behaviour but doesn't claim it's mathematically meaningful.

### limits_and_summation (10/10 graceful)

`Limit`, `Sum`, `Integral`, `Derivative`, `Product` — all handled
without crash. For unevaluated containers like `sp.Sum(1/k**2, (k, 1, oo))`,
the detector traverses the inner expression. The result's depth
includes the structural overhead of the container.

### complex_arithmetic (10/10 graceful)

`log(-1)`, `sqrt(-1)`, `log(I)`, `exp(I*x)`, complex polynomials
— all handled. SymPy returns `I*pi`, `I`, etc. as exact symbolic
constants; the detector traverses them correctly.

### numeric_edge_cases (10/10 graceful)

`exp(1000)`, `log(1e-100)`, `sin(10**10)`, `oo`, `zoo`, `nan`,
`0/0` — all return a result without crashing. For infinities and
NaN, the detector reports based on SymPy's representation
(typically `pfaffian_r = 0` for `oo`, `zoo`, `nan`).

### structural (8/10 graceful, 1 PASS, 1 EXPECTED_REJECT)

`Tuple`, `Matrix`, `Piecewise`, `Min`, `Max`, `Abs`, `Lambda`,
`Function('f')(x)`, `Eq` — most return a result; one rejects
gracefully. The bench documents the observed path but does not
claim either path is "correct" — both are valid handling of
non-scalar inputs.

## Recommended user practice

Based on these findings, the recommended pre-processing pipeline
for users who want maximally accurate analysis of their expressions:

```python
import sympy as sp
from eml_cost import analyze, canonicalize

# Step 1: parse
expr = sp.sympify("polylog(1, x) + hyper([], [], y)")

# Step 2: canonicalize to fold known closed forms
expr_c = canonicalize(expr)

# Step 3: analyze
result = analyze(expr_c)
```

The 0.4.0 `canonicalize` already handles many algebraic re-arrangements;
the polylog and hyper closed-form rewrites are candidates for a future
`canonicalize` extension.

## Reproducing

```bash
cd eml-cost-pkg-public
python bench/adversarial_runner.py             # human report
python bench/adversarial_runner.py --json      # JSON for CI
cat bench/adversarial_results.csv              # per-row results
```
