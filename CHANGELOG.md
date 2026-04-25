# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project will adhere to [Semantic Versioning](https://semver.org/) once
the public 1.0.0 release ships.

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
