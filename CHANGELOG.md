# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project will adhere to [Semantic Versioning](https://semver.org/) once
the public 1.0.0 release ships.

## [0.6.0] — 2026-04-26 — `estimate_time(expr)` SymPy compile-time predictor

This release ships the regression model from research session E-191:
a SymPy compile-time linter that predicts wall-time of canonical
compiler-style passes (`simplify`, `factor`, `cse`, `lambdify`) from
the expression's Pfaffian profile. Empirical basis: 202 expressions
from cross-domain-3 R v2 (2026-04-26). **No breaking API changes**.

### Added

- **`estimate_time(expr, proxy="all")`** — returns predicted wall-time
  in milliseconds for one or all four compile-time proxies. Each
  prediction includes a 95% confidence interval (asymmetric in linear
  space, symmetric in log-space) derived from the residual std of the fit.
- **`TimeEstimate`** dataclass with fields: `proxy`, `predicted_ms`,
  `ci95`, `log10_ms`, `log10_std`, `features`, `cv_r2`.
- **`model_metadata()`** — provenance for the shipped regression model
  (n=202, source, features, response, session).
- **`PROXIES`** tuple constant (`"simplify"`, `"factor"`, `"cse"`,
  `"lambdify"`).

### Empirical performance

5-fold cross-validated R^2 on log10(milliseconds):

| proxy    | CV R^2 | residual sigma (log10 ms) | typical ratio |
|----------|--------|---------------------------|---------------|
| simplify | 0.68   | 0.45                      | ~2.8x         |
| factor   | 0.76   | 0.16                      | ~1.4x         |
| cse      | 0.83   | 0.14                      | ~1.4x         |
| lambdify | 0.84   | 0.08                      | ~1.2x         |

Features: `eml_depth`, `max_path_r`, `log10(count_ops+1)`,
`log10(tree_size+1)`. Fit method: OLS on the full 202-row dataset
after CV validation.

### Caveats (documented in module docstring)

- Trained on Python 3.x + sympy >= 1.12 on a single CPU class.
  Absolute times shift with hardware/version; relative ordering is robust.
- Tail behavior beyond ~5s simplify time is extrapolation.
- `solve`, `integrate`, `series` are NOT modeled.

## [0.5.1] — 2026-04-26 — `analyze_batch` two-level cache (28x speedup on high-duplicate workloads)

### Changed

- **`analyze_batch(cache=True)` two-level cache.** The 0.5.0 cache
  ran `canonicalize()` on every input (even cache hits) to compute
  the cache key — producing the documented 0.58x slowdown on
  low-duplicate workloads. 0.5.1 adds a Level-1 raw-string fast path:
    - Level 1 (`raw_cache`): keyed by `str(input)`. Exact-string
      repeats are O(1) — zero canonicalize cost.
    - Level 2 (`canon_cache`): keyed by canonical-form string.
      Catches algebraically-equivalent forms.
- Empirical impact:
    - Low-duplicate (403 bio corpus): 0.58x → **0.94x** (near break-even).
    - High-duplicate (96% dup, deep transcendentals, n=200): 1.01x →
      **28.25x**.
- No API changes. Cache results identical to `cache=False` (verified by
  test).

## [0.5.0] — 2026-04-26 — `PfaffianProfile` + distance metric + `analyze_batch`

This release ships the work from research sessions E-183 (distance
metric), E-184 (analyze_batch + caching), and E-185 (demo notebook +
50-expression corpus). E-182 (`canonicalize()`) shipped earlier today
in 0.4.0 and is not duplicated here. **No breaking API changes** —
`analyze()` and `AnalyzeResult` still work exactly as before.

### Added

- **`PfaffianProfile`** (in `eml_cost.profile`) — rich frozen dataclass
  with `r`, `degree`, `width`, `cost_class`, `oscillatory`, `corrections`,
  `expression`, `canonical_form`, `is_pfaffian_not_eml`. Construct via
  `PfaffianProfile.from_expression(expr, do_canonicalize=True)` or wrap
  an existing `AnalyzeResult` via `from_analysis()`.
- **`PfaffianProfile.distance(other)`** — weighted Euclidean metric in
  (r, d, w, c) coordinate space. Default weights `r=4, d=1, w=2, c=1`
  (chain order dominates; configurable per call). Verified to satisfy
  identity, symmetry, and the triangle inequality across 13K+ test
  triples (`tests/test_profile_metric.py`).
- **`PfaffianProfile.compare(other)`** — per-axis deltas + `same_class`
  flag.
- **`PfaffianProfile.is_elementary()`** — convenience predicate.
- **`PfaffianProfile.to_dict()` / `to_row()` / `csv_header()`** — JSON +
  CSV serialization helpers.
- **`analyze_batch(expressions, n_jobs=1, cache=True, progress=None,
  canonicalize=True)`** — batch profiler with `joblib`-based
  multiprocessing, canonical-form-keyed caching, and `tqdm` progress
  bars. SymPy is not thread-safe, so we use processes (`backend="loky"`).
  Failed expressions get an `ERROR` sentinel profile rather than
  crashing the batch. Order is preserved.
- **`cache_hit_analysis(expressions)`** — theoretical cache hit rate
  for a workload (counts unique canonical forms).
- **`eml_cost/data/demo_corpus.csv`** — bundled 50-expression cross-
  domain corpus across 9 domains (polynomial, exp/log, trig,
  Pfaffian-not-EML, ML activations, physics, biology, engineering,
  random null) with citations on every row. Loadable via
  `importlib.resources.files("eml_cost").joinpath("data/demo_corpus.csv")`.
- **`notebooks/quickstart.ipynb`** — 8-cell guided tour. Tested
  end-to-end via `notebooks/test_quickstart.py`. All 50 corpus
  expressions profile successfully.

### Empirical basis

- Triangle-inequality verification: 13,824 test triples across 24
  representative expressions. Zero violations (`tests/test_profile_metric.py`).
- 403-expression cross-domain bio corpus distance matrix +
  hierarchical clustering shipped as `bench/distance/` reproducer
  (heatmap PNG, dendrogram PNG, full numeric matrix).
- 10,100-expression regression of canonicalize() on the cumulative
  random + null corpus: **84.76% same cost class with vs without
  canonicalize** (n=15,000), **15.24% reclassified** under canonical
  form. Most reclassifications are documented in `bench/regression/`.

### Tests

- 12 new in `tests/test_batch.py` (analyze_batch behavior + cache
  hit analysis).
- 17 new in `tests/test_profile_metric.py` (PfaffianProfile API +
  metric axioms).
- 91 prior + 29 new = **120 tests passing.**

### Optional dependencies

- `joblib` — required for `n_jobs > 1`. Falls back to serial if absent.
- `tqdm` — progress bar. Falls back to no-op if absent.

## [0.4.0] — 2026-04-26 — `canonicalize()` preprocessing for form-fragility

### Added

- `eml_cost.canonicalize(expr)` — applies a curated sequence of cheap,
  content-preserving rewrite rules to convert algebraically-equivalent
  expressions to a canonical form before cost-class measurement.
- `eml_cost.analyze_canonical(expr)` — convenience wrapper for
  `analyze(canonicalize(expr))`.

### Why

A form-sensitivity audit on 20 textbook expressions × 4 algebraically-
equivalent forms each (80 substrate evaluations) found **50% of
expressions yielded different cost classes for different forms**.
The worst offender: sigmoid, where four equivalent forms produced four
different cost classes:

  - `1 / (1 + exp(-x))`     → `p1-d1-w1-c-1`
  - `exp(x) / (exp(x) + 1)` → `p1-d2-w1-c0`
  - `0.5 * (1 + tanh(x/2))` → `p1-d4-w1-c0`
  - `1 - 1 / (1 + exp(x))`  → `p1-d3-w1-c0`

`canonicalize` applies:
  - `sympy.together` to merge fractions
  - `sympy.logcombine(force=True)` for `log(x) - log(y)` → `log(x/y)`
  - `sympy.trigsimp` for `cos(a)cos(b) + sin(a)sin(b)` → `cos(a-b)`
  - explicit pattern rewrite `1 - 1/(1+exp(x))` → `1/(1+exp(-x))`
  - `sympy.factor_terms` for distributive normalization

### Impact (re-run of form-sensitivity audit)

  - Drift rate **50% → 35%** (10 → 7 of 20 tests still drift)
  - **Fully fixed:** traveling_wave, logistic_growth, rc_charging
  - **Reduced (still drift but less):** sigmoid (4 → 2 unique classes),
    gaussian_kernel_2d (3 → 2)
  - **Still drift:** sigmoid, hill_kernel, exponential_decay,
    gaussian_kernel_2d, softmax_2key, nernst_potential, cardiac_oscillator
    — these reflect genuine substrate sensitivity to inner-argument
    shape (e.g., `exp(-k*t)` vs `exp(-t/tau)` use different SymPy
    Mul structures), not bugs.

### Performance

`canonicalize` adds a small constant multiple (typically 2-5x) to
`analyze` cost. Total `analyze_canonical` cost is still in the
sub-millisecond range for typical expressions and **still 5-50x faster
than `sympy.simplify`** per the bench/speed_bench.py results.

### Tests

- 13 new tests in `tests/test_canonicalize.py`. Full suite: **91 passing**.

### When to use

  - **Use `analyze_canonical`** when measuring cost classes across a
    user-supplied corpus where you don't control the expression form.
  - **Use plain `analyze`** when you control the form yourself and
    want maximum speed (e.g., in a tight inner loop with hand-curated
    expressions).
  - **The patent claim shifts** from "Pfaffian profile of an
    expression" to "Pfaffian profile of the canonical form of an
    expression"; the canonicalize() rule set is part of the disclosed
    method.

## [0.3.0] — 2026-04-25 — `PFAFFIAN_NOT_EML_R` registry expansion

### Added — 20 new non-elementary primitives recognised

S/R-134 deep research revealed that 24 named non-elementary SymPy
functions were silently treated as depth-0 atoms by the cost
detector — `is_pfaffian_not_eml` returned False for `erf`, `gamma`,
`polylog`, `zeta`, `elliptic_k`, `Ei`, etc., despite these being
genuinely non-elementary. This release adds 20 new entries to
`PFAFFIAN_NOT_EML_R` covering 6 special-function families:

| Family | Entries | Chain orders |
|---|---|---|
| erf-family | erf, erfc, erfi, fresnels, fresnelc | 2-3 |
| Gamma family | gamma, loggamma, polygamma, beta | 2-3 |
| Exp / cos integrals | Ei, li, Si, Ci, Shi, Chi | 2-3 |
| Polylog / zeta | polylog, zeta | 3-4 |
| Elliptic | elliptic_k, elliptic_e, elliptic_f | 3-4 |

(Note: `digamma` reduces to `polygamma(0, x)` in SymPy, so a
single `polygamma` entry covers digamma + trigamma + general
polygamma_n.)

Each chain order is a conservative estimate justified inline at
`core.py:37-90` against standard analytic facts (Frobenius series,
defining ODEs, Pfaffian chain elements). The substrate's
`analyze()` and `fingerprint()` now report meaningful depths for
these functions instead of treating them as constants.

### Behaviour change

  - **`is_pfaffian_not_eml(sp.erf(x))`** — was `False`, now `True`.
  - **`fingerprint(sp.erf(x))`** — axes change from `p0-d0-w0-c0`
    to `p2-d1-w2-c0` (or similar; depends on the chain order).
  - **`predicted_depth(sp.gamma(x))`** — was `0`, now `2-3`.
  - Downstream `eml-witness 0.2.1+` and `monogate 2.4.3+` continue
    to return `verified_in_lean=False` for these functions (their
    strict allow-list check is independent of this registry; the
    two gates are now mutually reinforcing rather than redundant).

### Tests

- 6 new tests in `tests/test_extended_registry.py` covering one
  representative from each new family + a regression test that
  every new entry is independently `is_pfaffian_not_eml=True`.

### Migration notes

If any downstream code was using `is_pfaffian_not_eml=False` as a
proxy for "this is elementary", it now needs to handle the
correct case (the function IS Pfaffian-not-EML). Strict
elementary-class detection remains in `monogate.witness` /
`eml-witness` allow-list.

## [0.2.0] — 2026-04-25 — `@cache_by_fingerprint` (cost-class memoization)

### Added
- `@cache_by_fingerprint(maxsize=N)`: LRU memoization decorator
  keyed on the **axes portion** of the Pfaffian fingerprint for
  SymPy arguments (and normal hash for everything else). Two
  expressions in the same cost equivalence class share a cache
  slot — so `slow(sin(x))` and `slow(sin(y))` are a single entry.
  Wrapper exposes `.cache_info()` returning `FingerprintCacheInfo`
  and `.cache_clear()`. Unhashable secondary args bypass the
  cache transparently rather than crashing.
- `fingerprint_axes(expr) -> str`: helper returning the leading
  `p…-d…-w…-c…` block of `fingerprint(expr)` (no tail hash).
  Useful as a manual cache key when the decorator's calling
  convention isn't a fit.
- `FingerprintCacheInfo` frozen dataclass exporting through the
  package root.

Use case: cost-driven analyses where the result depends only on
the Pfaffian profile (e.g., per-cost-class measurement, model
profiling on synthetic expression families). Standard
`functools.lru_cache` would miss on every renamed variant; this
decorator collapses them.

**Caveat documented.** The decorator can't enforce that the wrapped
function depends only on cost class. If the function inspects
specific symbol names or values, hits will return wrong results.
Use only when the cost-class-only contract is intentional.

### Tests
- 11 new cases in `tests/test_cache_by_fingerprint.py`. Full
  suite: 71 passing.

## [0.1.2] — 2026-04-25 — `@costlimit` decorator

### Added
- `@costlimit(predicted_depth=N, max_path_r=N, pfaffian_r=N)`:
  decorator that enforces a cost ceiling on the return value of
  the wrapped function. At least one axis must be configured;
  multiple are AND'd. Non-SymPy returns pass through untouched.
  When the limit is exceeded, raises `CostLimitExceeded` with
  `.expression`, `.axis`, `.measured`, `.limit` attached.
- `CostLimitExceeded` exception class (exported from package root).

Use case: type-hint-style cost contracts on functions returning
SymPy expressions. Useful in numerical pipelines, regression
tests, and "this function must stay simple" enforcement.

### Tests
- 10 new cases in `tests/test_costlimit.py`. Full suite: 60 passing.

## [0.1.1] — 2026-04-25 — Pfaffian fingerprint

### Added
- `fingerprint(expr) -> str`: compact structural-cost hash of the
  form `p<r>-d<depth>-w<max_path>-c<correction_sum>-h<6hex>`.
  Two expressions colliding on this hash are guaranteed to have
  identical Pfaffian profile values; the 6-hex tail folds in
  Pfaffian-not-EML status + tree-shape signature so Bessel and
  Airy don't accidentally collide despite identical numeric axes.
  Enables expression-dedup at scale, equivalence-class caching,
  and "have I analyzed this before?" lookup in O(1).

### Tests
- 8 new cases in `tests/test_fingerprint.py`. Full suite: 50 passing.

## [0.1.0] — 2026-04-25 — First stable release

**Status.** Stable beta. Patent pending.

### Highlights
- Domain-aware analysis: `is_pfaffian_not_eml` correctly classifies
  Bessel, Airy, Lambert W, and other extension primitives that fall
  outside the strict EML class but inside the Pfaffian class.
- v5 max-path Pfaffian chain order: parallel composition through
  Add / Mul / poly-Pow takes the max over independent paths instead
  of the sum, giving sharp predictions on independent-variable
  products like `sin(x) * cos(y)` (r=1, not 2).
- Extension primitive registry: 12 named primitives with
  hand-derived chain orders.

### Added vs 0.1.0a0
- 10 new smoke tests: nested compositions (`exp(sin(x))`,
  `log(exp(exp(x)))`), Pfaffian-but-not-EML detection
  (`besselj`, `airyai`, `LambertW`), independent-variable
  max-path behavior, and edge cases (single constant, single
  variable, two-variable independence).
- README "Quick Start" section with three runnable examples.
- `Development Status` classifier promoted from Alpha to Beta.

## [0.1.0a0] — 2026-04-25 — Pre-release skeleton

**Status.** Engineering build. NOT released. Patent pending.

### Added
- Core detector module (`eml_cost.core`) with:
  - `pfaffian_r(expr)` — total Pfaffian chain order (Khovanskii)
  - `max_path_r(expr)` — path-restricted chain order
  - `eml_depth(expr)` — EML routing tree depth
  - `structural_overhead(expr)` — Add/Mul/poly-Pow tree depth
  - `is_pfaffian_not_eml(expr)` — extension-class detector
  - `PFAFFIAN_NOT_EML_R` registry (12 named primitives)
- Top-level `analyze(expr)` API returning `AnalyzeResult` dataclass
- `measure(expr)` — drop-in for `sympy.simplify(…, measure=…)`
- F-family fusion patterns recognized: LEAd `log(c + exp(g))`, sigmoid `1/(1+exp(-g))`
- Strict mypy typing (`py.typed` marker shipped)
- 30 tests: 22 API contract + 8 smoke bench (hand-derived rows)

### Constraints
- License is `PROPRIETARY-PRE-RELEASE`. Do not redistribute.
- Patents #11 and #12 (filed 2026-04-25) cover the methods this
  package exposes. Public release follows post-prosecution licensing.
- Package documents the API only. Empirical validation results live
  in separate research artifacts and a forthcoming paper.
