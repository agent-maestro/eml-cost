# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project will adhere to [Semantic Versioning](https://semver.org/) once
the public 1.0.0 release ships.

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
