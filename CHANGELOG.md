# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project will adhere to [Semantic Versioning](https://semver.org/) once
the public 1.0.0 release ships.

## [0.17.1] — 2026-04-28 — two-sided chain regularizer + benchmark condition D

A focused iteration on the chain-order regularizer in response
to the 0.17.0 finding that one-sided (upper-bound-only)
regularization cannot prevent underfitting.

### Added

  - **`RegularizerConfig.target_chain_order`** — when set,
    switches the chain-order penalty to **two-sided** absolute-
    deviation mode: ``penalty = lambda_chain * |chain_order -
    target_chain_order|``. Penalises both overshoot and
    undershoot. ``is_feasible`` becomes
    ``chain_order == target_chain_order`` in this mode.

  - **Benchmark condition D** — fourth condition in
    :func:`eml_cost.regression.benchmark.run_benchmark`. Uses
    two-sided regularization with ``target_chain_order`` derived
    from :func:`estimate_dynamics` on the regression target. Pass
    ``include_two_sided=False`` to drop back to the 0.17.0
    three-condition harness.

  - **Noise-floor gate in `_detect_oscillations`** —
    ``min_peak_to_median=20.0`` ratio gate prevents the FFT
    peak-detection from registering pure noise as oscillations.
    Was triggering false positives on chain-0 problems with
    sample-grid jitter.

### Validation rerun

The 4-condition rerun is documented in
`monogate-research/exploration/eml-symbolic-regression-2026-04-28-v2/FINDINGS.md`.

### Default policy

The 0.17.0 one-sided behaviour remains the default (no API
break). Users who can derive a target chain order from data
should pass ``target_chain_order=...`` to switch on two-sided
mode.

### Tests

  - +6 tests for two-sided regularizer mode in
    ``test_regularizer.py``.
  - +1 test for ``include_two_sided=False`` benchmark mode.


## [0.17.0] — 2026-04-28 — GP search engine + Feynman benchmark

Phase 2 of EML-native symbolic regression. The 0.16.0 regularizer
gets a search engine to steer and a benchmark to validate against.

### Added

  - **`eml_cost.regression.EMLNode`** — symbolic-regression tree
    representation. Binary `{+, -, *, /}` and unary `{sin, cos,
    exp, log, sqrt, neg}` operators. `evaluate(env)` is numpy-
    backed (with safe overflow / divide-by-zero handling);
    `to_sympy()` bridges to the rest of `eml_cost`.

  - **`eml_cost.regression.search`** — genetic-programming SR
    engine. Tournament selection, subtree crossover + mutation,
    constant Gaussian perturbation. Optional smart initialisation
    via `estimate_dynamics` for single-variable problems. Every
    candidate is scored against an optional
    `RegularizerConfig`.

  - **`eml_cost.regression.random_baseline`** — compute-matched
    random-tree sampler. Used as the C-condition reference in
    the benchmark.

  - **`eml_cost.regression.benchmark`** — 10-problem Feynman-
    flavoured SR bench spanning chain orders 0 through 3.
    `run_benchmark` runs three conditions (with regularizer /
    without / random) and returns a `BenchmarkTable` with
    per-problem and overall hit rates.

### Validation result

On the bench, A (regularized GP) achieves **53.3%** chain-order
match rate vs **43.3%** for B (unregularized GP) over 30 runs
per condition (+10pp). C (random tree, compute-matched) scores
56.7% — a metric artifact, since random sampling regresses to
the chain-order distribution mean (biased toward 0 and 2 in this
suite). On the hardest problem (damped oscillator, true chain 3),
A is the only method that recovers chain 3.

Full details in
`monogate-research/exploration/eml-symbolic-regression-2026-04-28/FINDINGS.md`.

### Honest framing

  - The chain-order metric is coarse — many distinct expressions
    share a chain order. The +10pp regularizer lift is real but
    modest.
  - The current regularizer formulation only penalises chain
    order **above** `max_chain_order`. A two-sided variant
    (target chain order from `estimate_dynamics`, penalise both
    over- and under-shoot) is the obvious next iteration.
  - Smart-init via `estimate_dynamics` does material work on the
    chain-3 problem; the regularizer alone vs smart-init alone
    is not yet ablated.

### Tests

  - +43 tests across `test_nodes.py`, `test_gp.py`, and
    `test_benchmark.py` (smoke runs of the harness with tiny
    population/generation budgets to keep CI fast).
  - Suite total: 589 passing.


## [0.16.0] — 2026-04-28 — chain-order regularizer + data dynamics estimator

Phase 1 of the EML-Native Symbolic Regression effort. Two new
public modules wire the existing `analyze` /
`predict_chain_order_via_additivity` / `recommend_form` machinery
into a search-loop-friendly regularizer.

### Added

  - **`eml_cost.regularize`** — score a candidate expression by
    chain-order excess, node count, dynamics mismatch, and
    recommend_form distance. Each component is independently
    weighted via `RegularizerConfig`; output is the frozen
    `RegularizerResult` with per-component breakdown and a
    one-line `explanation`.

  - **`eml_cost.estimate_dynamics`** — infer the
    `(n_oscillations, n_decays, n_static)` signature from a
    sampled `(x, y)` series via FFT (oscillation modes) and
    Hilbert envelope analysis (decay modes). Returns a frozen
    `DataDynamics` dataclass usable as
    `RegularizerConfig.expected_dynamics`. Numpy + scipy are
    optional; install via `pip install eml-cost[data]`.

  - **Optional extra `[data]`**: numpy>=1.24, scipy>=1.10. The
    core package remains sympy-only.

### Honest framing

Chain order is a **structural** measure — Spearman ρ ≈ 0.4
against Chebyshev polynomial-approximation degree on a 35-expression
probe (see `monogate-research/exploration/zk-circuit-cost-2026-04-28/`).
The regularizer biases search toward simpler EML forms; it does
not guarantee numerical stability or polynomial-degree minimality.

### Tests

  - +33 tests across `test_regularizer.py` and
    `test_data_analyzer.py`. Suite total: 546 passing.


## [0.14.0] — 2026-04-27 — registry consistency · `besselk` and Hankel 3→5

Fixes three more chain-order undercounts in PFAFFIAN_NOT_EML_R,
discovered during the Y_n higher-order analysis of 2026-04-27
(`monogate-research/exploration/yn-higher-order-2026-04-27/`).

The registry has `bessely` at chain 5 with comment "log-singular
at origin". `besselk` at integer n has the SAME log-singularity
structure (DLMF 10.31.2):

  K_n(x) = (1/2)·(-1)^(n+1)·I_n(x)·[2·ln(x/2) + 2γ - ψ(n+1)]
          + finite series

It was registered at 3 — same as J_n / I_n which are analytic at 0.
By the chain-order additivity rule and the structural parallel to
bessely, K_n should be 5 — matching the spherical analog `hn1`/`hn2`
which the registry already correctly tags at 5.

Hankel functions H_n^(1) = J_n + i·Y_n and H_n^(2) = J_n - i·Y_n
inherit Y_n's log-singularity (one of their two components is
chain 5). Same fix applies.

### Changed

  - **`PFAFFIAN_NOT_EML_R["besselk"]`: 3 → 5.** Aligns with
    bessely under the log-singularity argument at integer n.
  - **`PFAFFIAN_NOT_EML_R["hankel1"]`: 3 → 5.** Composition
    inheriting Y_n's chain.
  - **`PFAFFIAN_NOT_EML_R["hankel2"]`: 3 → 5.** Same.
  - Inline comment expanded to cite DLMF 10.31.2 and the
    empirical-discovery exploration directory.

### Migration notes

Code that asserted `eml_cost.analyze(sp.besselk(0, x)).pfaffian_r == 3`,
or the same for hankel1/hankel2, will break. Corrected value is 5.

The asymmetry between hn1/hn2 (already correct at 5, added in 0.9.0
with comment "log-singular like bessely") and hankel1/hankel2
(previously 3) was the giveaway that the latter was undercounted.

### Source

  - Discovery: `monogate-research/exploration/yn-higher-order-2026-04-27/`
  - Companion: `monogate-research/exploration/full-corpus-additivity-2026-04-27/`
  - Validates: chain-order additivity rule across log-singular
    Bessel-family entries.

---

## [0.13.0] — 2026-04-27 — registry consistency fix · `riemann_xi` 5→6

Fixes a chain-order miscount discovered during the 9th-tier
promotion + chain-5 hunt of 2026-04-27. The PFAFFIAN_NOT_EML_R
registry assigned `riemann_xi` chain order 5; the chain-order
additivity rule that's now corpus-validated (28/28 PNE rows hit
exactly per `monogate-research/exploration/full-corpus-additivity-
2026-04-27/`) gives chain 6 for `riemann_xi`:

  ξ(s) = (s(s−1)/2) · π^(−s/2) · Γ(s/2) · ζ(s)

The Γ(s/2) contributes chain 2, the ζ(s) contributes chain 4 (T_Lerch
supertower, also promoted today), and the additivity rule sums to 6.
Polynomial prefactors and `π^(−s/2)` (a non-integer-exponent power
treated as +0 chain order at the registry level) don't add.

### Changed

  - **`PFAFFIAN_NOT_EML_R["riemann_xi"]`: 5 → 6.** Aligns with the
    chain-order additivity rule and the per-row C-207 PNE verification.
  - Inline docstring expanded to cite the additivity-rule derivation.

### Migration notes

Any code that asserted `eml_cost.analyze(sp.riemann_xi(s)).pfaffian_r == 5`
will break. The corrected value is 6. This is a SEMVER minor bump
because the registry's behaviour for `riemann_xi` is now strictly
more consistent with the documented additivity rule, but downstream
callers that pinned the old value need a one-line update.

### Source

  - Discovery: `monogate-research/exploration/chain-5-hunt-2026-04-27/`
    (registry-vs-additivity discrepancy noted)
  - Confirmation: `monogate-research/exploration/full-corpus-additivity-
    2026-04-27/` (28/28 PNE rows hit exactly under the additivity rule)
  - 9th-tier context: `exploration/9th-tower-promotion-2026-04-27/`
    promoted T_Lerch (chain 4) and T_Mathieu (chain 4) to CONFIRMED;
    riemann_xi sits in T_Lerch's neighbourhood as a Γ × ζ composite.

---

## [0.12.0] — 2026-04-27 — `eml-cost lint` pre-commit linter

Surfaces ``estimate_time`` at the source-file level. Scans Python
files for SymPy ``simplify``/``factor``/``expand``/``cse``/
``lambdify`` calls, infers the expression argument from string
literals or ``sp.sympify("...")`` calls, and predicts wall-time
before the call would actually run.

### Added

  - ``lint_file(path) -> list[Finding]`` — file-based linter API.
  - ``lint_source(text, filename=...) -> list[Finding]`` — string-
    based variant for tooling integration.
  - ``Finding`` dataclass — file, line, col, function, proxy,
    expression_repr, predicted_seconds, severity (info/warn/error),
    message.
  - ``eml-cost lint <paths...>`` CLI subcommand with
    ``--max-seconds`` (pre-commit budget) and ``--severity``
    (filter level) flags.
  - 17 new tests covering pattern detection (sp.simplify,
    sp.factor, sp.expand, bare simplify, attribute call, .subs
    not flagged), severity classification, source-only
    resolution (no runtime context), syntax-error handling,
    file-interface, line-numbers, and missing-file diagnostic.

### Honest framing

  - The linter ONLY resolves expressions it can infer at parse
    time: string literals (``simplify("exp(x)")``) and
    ``sympify("...")`` wrappers. Bare variable names and method
    chains are SKIPPED — no whole-program flow analysis.
  - The estimator was fit on E-191 / E-192 corpora at 5-fold
    CV R² 0.68–0.84. Predicted times are an ORDERING signal, not
    an absolute timeout.
  - SymPy's auto-Symbol behaviour is mitigated: ``sympify("e")``
    returning Euler's E does not cause false positives because
    bare names are not resolved.
  - Source: ``estimate_time`` model in ``estimate_time.py``;
    corpus in ``monogate-research/exploration/E191_estimate_time``.

## [0.11.0] — 2026-04-27 — `find_siblings()` cross-domain structural search

The 578-row cross-domain corpus is now bundled inside the package.
`find_siblings(expr)` returns the structurally most similar
expressions across 12 subdomains (bench-300 + E-196 + Robotics +
Human Body + Music & Sound + Olfactory + Color Science) ranked by
weighted Euclidean distance on the Pfaffian profile.

### Added

  - ``find_siblings(expr, k=5, *, domain=None, max_distance=None,
    weights=None, exclude_self=True) -> list[Sibling]`` — top-k
    structural neighbour search.
  - ``Sibling`` dataclass — name, domain, expression, cost_class,
    distance, profile.
  - ``corpus_size()`` and ``corpus_domains()`` introspection helpers.
  - ``data/corpus_578.csv`` packaged with precomputed PfaffianProfile
    fields per row (one-time cost paid at build, not at runtime).
  - 17 new tests covering corpus introspection, k-cap, sort order,
    domain filter, max_distance, custom weights, and structural
    sanity (cosine → cosine neighbours, polynomial → polynomial).

### Honest framing

  - **OBSERVATION** tier. Structural similarity is not physical
    equivalence — two expressions with identical ``cost_class``
    can model unrelated systems.
  - The corpus is curated, not exhaustive. Absence of a sibling
    means "not in our 578 sample," not "no sibling exists."
  - Distance is in cost-class units, not normalised across
    expression types.
  - Source: ``monogate-research/data/bench.md`` and
    ``exploration/E196_algorithmic_corpus/master_corpus_578.csv``.

## [0.10.0] — 2026-04-27 — `analyze_dynamics()` headline feature

E-196 Phase 4f cross-domain study (n=175 paired rows across 12
subdomains) validated the slope-2 rule
``r ≈ 2 * n_oscillations + 1 * n_decays`` at Spearman ρ = +0.890,
Pearson r = +0.929. This release exposes that rule as a public API.

### Added

  - ``analyze_dynamics(expr) -> DynamicsProfile`` — count
    oscillation modes, decay modes, and static components in a
    symbolic expression. Returns the slope-2 prediction of
    ``pfaffian_r`` plus a confidence label and human-readable
    description.
  - ``DynamicsProfile`` dataclass — exposes ``n_oscillations``,
    ``n_decays``, ``n_static``, ``predicted_r``, ``actual_r``,
    ``confidence`` (high/moderate/low), ``description``, and
    ``domain_fit`` (strong/moderate/weak).
  - 17 new tests covering polynomial / oscillation / decay /
    damped-oscillator / triple-Gaussian / PNE primitive cases.

### Honest framing

  - **OBSERVATION** tier. ρ=0.890 is empirical, not a theorem.
  - The slope-2 rule is calibrated for OSCILLATORY expressions.
    It under-predicts for parallel-Gaussian or sqrt-heavy
    expressions where Pfaffian's DAG-summed chain count exceeds
    the simple per-mode contribution (per-domain ρ on olfactory
    drops to 0.46).
  - For PNE primitives (Bessel, Airy, Gamma, Lambert W), the
    counter abstains via ``confidence = "low"``.
  - Source: ``monogate-research/exploration/E196_algorithmic_corpus/``
    and ``color_science_subdomain/phase4f_extended_v5_summary.json``.

## [0.9.0] — 2026-04-27 — `PFAFFIAN_NOT_EML_R` second expansion (32 → 68 entries)

S/R-#10 substrate-coverage audit. Direct enumeration of
``sympy.functions.special`` surfaced 36 named non-elementary
SymPy classes that the 0.3.0 expansion had missed. They were being
silently treated as depth-0 atoms by the cost detector (same
class of bug the 0.3.0 expansion fixed). This release adds them
with chain orders derived from each function's defining ODE or
closure relation under the Khovanskii convention.

The original expansion target was 60+; the actual landing is 68.

### Added — 36 new ``PFAFFIAN_NOT_EML_R`` entries

  - **Spherical Bessel & Hankel** (4): ``jn``, ``yn``, ``hn1``, ``hn2``.
    Same chain orders as the unscaled Bessel siblings (3 / 5 / 5 / 5).
  - **Marcum Q** (1): ``marcumq`` — Bessel-derived integral, chain 3.
  - **Erf variants** (4): ``erf2``, ``erfinv``, ``erfcinv``, ``erf2inv``.
  - **Exponential integrals** (3): ``expint`` (E_n), ``E1``, ``Li``.
  - **Gamma family** (8): ``digamma``, ``trigamma``, ``lowergamma``,
    ``uppergamma``, ``multigamma``, ``harmonic``, ``factorial``,
    ``factorial2``.
  - **Pochhammer / subfactorial** (3): ``RisingFactorial``,
    ``FallingFactorial``, ``subfactorial``.
  - **Elliptic third kind** (1): ``elliptic_pi``.
  - **Zeta family** (4): ``dirichlet_eta``, ``lerchphi``,
    ``stieltjes``, ``riemann_xi``.
  - **Hypergeometric extensions** (2): ``meijerg``, ``appellf1``.
  - **Mathieu functions** (4): ``mathieuc``, ``mathieus``,
    ``mathieucprime``, ``mathieusprime``.
  - **Spherical harmonics** (2): ``Ynm``, ``Znm``.

### Honest auto-evaluation note

  - SymPy auto-evaluates several of these to elementary forms when
    a parameter is a positive integer literal:
    - ``lowergamma(2, x)`` → ``1 − (x+1)·exp(−x)`` (elementary)
    - ``RisingFactorial(x, 2)`` → ``x·(x + 1)`` (polynomial)
    - ``factorial(3)`` → ``6`` (rational)

    The detector behaves correctly here: once SymPy has rewritten
    the expression into an elementary form, the registry entry no
    longer matches, and ``is_pfaffian_not_eml`` returns ``False``
    — because the *resulting expression* really is elementary.
    The new entries fire on the genuinely-non-elementary cases
    (symbolic parameters, indices that keep the call
    unevaluated). This is documented in the new test file.

### Fixed (S/R-#9 adversarial bench finding)

  - ``is_pfaffian_not_eml(hyper((), (), x))`` previously returned
    ``True`` because ``hyper`` is in the registry. SymPy keeps
    ``0F0(;;x) = exp(x)`` in this canonical form and never
    auto-simplifies (verified: ``.doit()`` is a no-op,
    ``.rewrite(exp)`` is a no-op). The detector now short-circuits
    empty-parameter ``hyper`` to elementary. Same fix applies to
    ``sp.hyper([1], [1], x)`` because SymPy cancels ``[1]/[1]`` →
    ``hyper((), (), x)``. Found by ``bench/adversarial_runner.py``,
    rows 29 + 30; documented in ``bench/ADVERSARIAL_FINDINGS.md``.

### Adversarial bench (S/R-#9 deliverable)

100-expression stress bench at ``bench/adversarial.csv`` covering
10 categories (deep nesting, mixed PNE/elementary, auto-eval traps,
limits/sums, complex arithmetic, distributions/discrete, PNE-of-PNE
compositions, numeric edge cases, structural objects). The runner
``bench/adversarial_runner.py`` produces a per-row results CSV plus
a human-readable / JSON summary. Findings are written up honestly in
``bench/ADVERSARIAL_FINDINGS.md``: 99/100 cases classify cleanly
post-fix, with one documented limitation (``polylog(1, x)`` flags as
PNE on AST level even though it equals ``-log(1 - x)`` mathematically
— SymPy itself doesn't auto-rewrite that one).

### Tests

19 new tests in ``tests/test_v090_registry_extension.py``; full
eml-cost suite **240/240 green** (was 221 before this release).
mypy --strict clean.

### Migration

No API changes. Code that previously got
``is_pfaffian_not_eml=False`` for these functions will now get
``True``. This is a correctness fix — the previous behaviour was
silently treating them as depth-0 atoms. Conversely,
``hyper((), (), x)`` previously got ``True`` (false positive); it
now correctly returns ``False``.

### Source

  - Registry expansion source: derived from direct enumeration of
    ``sympy.functions.special`` modules.
  - Adversarial bench source: ``bench/adversarial.csv`` and
    ``bench/ADVERSARIAL_FINDINGS.md`` (both shipped publicly with
    the wheel).

## [0.8.0] — 2026-04-27 — `recommend_form(expr)` narrow-scope numerical-form recommender

This release ships a deliberately-narrow recommender that fires on
**only** the four expression families with E-193 Phase 3 Spearman
rho >= 0.77 between predicted EML depth and measured float64
mpmath_max_relerr. On every other expression it returns ``None``.

The general-purpose form recommender was deliberately not shipped in
0.7.0 (E-193 Phase 3 best-pick = 30 percent on the full 10-family
test, well below the 70 percent product threshold). This release adds
the narrow-scope variant for the four families that did concord.

### Added

- **`recommend_form(expr)`** — returns a ``RecommendedForm`` if the
  expression is in one of:
  - `sigmoid` (rho = +0.800) — canonical form `1 / (1 + exp(-x))`
  - `exponential_decay` (rho = +0.775) — canonical form `exp(coeff * x)`
    with the coefficient folded into a single multiplicative term
  - `logistic_growth` (rho = +0.775) — canonical form
    `K / (1 + exp(-r * (t - t0)))`
  - `cardiac_oscillator` (rho = +0.866) — canonical form
    `A * cos(omega * t + phi)`
  Returns `None` for everything else.
- **`RecommendedForm`** dataclass with fields: `family`,
  `canonical_form`, `input_predicted_relerr`,
  `recommended_predicted_relerr`, `digits_saved`, `rho`, `honest_note`.
- **`SUPPORTED_FAMILIES`** tuple constant (the four families).
- **`FAMILY_RHO`** dict mapping family name -> Spearman rho.

### Honest framing baked in

  - Every `RecommendedForm` carries a non-empty `honest_note` field
    explaining the four-family scope and naming the un-shipped
    general-purpose recommender as a deliberate choice.
  - Family detection is structural: `recommend_form("besselj(0, x)")`,
    `recommend_form("x**3 + 2*x")`, and the six non-concordant E-193
    families (`hill_kernel`, `softmax_2key`, `nernst_potential`,
    `gaussian_kernel_2d`, `rc_charging`, `traveling_wave`) all return
    `None`.
  - The Spearman rho values come from `phase3_killer.json` and are
    asserted by the test suite to catch silent drift.

### Tests

27 new `recommend_form` tests; full eml-cost suite 221/221 green.
mypy --strict clean (12 source files).

Source data: `monogate-research/exploration/E193_numerical_stability/
phase3_killer.json` + `form_with_stability.csv`.

## [0.7.2] — 2026-04-27 — pre-commit recipe + working example files

Documentation-driven patch: makes the 0.7.1 CLI immediately usable in
a pre-commit hook or CI without the user having to write the YAML
themselves.

### Added

- **README "Command-line interface" section** documenting `eml-cost
  report` / `eml-cost check` / `--file` / exit codes.
- **README "Pre-commit hook recipe"** showing the
  `.pre-commit-config.yaml` block + companion `expressions.txt` file
  pattern (one expression per line, `#` comments allowed).
- **README "CI integration" snippet** for GitHub Actions.
- **`examples/.pre-commit-config.example.yaml`** — drop-in pre-commit
  config, with `--max-simplify-ms 200 --max-digits-lost 3` defaults
  and a commented optional companion hook to verify SymPy parses each
  line before the budget check.
- **`examples/expressions.example.txt`** — sample expression list
  exercising sigmoid / composition / sqrt / polynomial cases. All four
  pass the documented `200ms / 3 digits` budgets.

### Verified

- Running the example `expressions.txt` through `eml-cost check
  --file ...` with the documented budgets produces `exit=0` (4 OK).

### No code changes

This is a docs-and-examples release. The 0.7.1 CLI module, the
prediction trilogy, and all model coefficients are unchanged.

## [0.7.1] — 2026-04-26 — `eml-cost` CLI (UX layer for the trilogy)

This release ships an installable `eml-cost` console command that makes
the existing 0.7.0 prediction trilogy usable from the shell and from a
pre-commit hook. **No model coefficients changed**; this is purely a
UX layer.

### Added

- **`eml-cost report EXPR [EXPR ...]`** — pretty-prints
  `analyze` + `estimate_time` + `predict_precision_loss` for each
  expression. Add `--json` for machine-readable output.
- **`eml-cost check EXPR [EXPR ...]`** — same as `report` but exits
  non-zero if any expression exceeds a budget. Use
  `--max-simplify-ms N` to gate compile-time and `--max-digits-lost N`
  to gate runtime numerical loss.
- **`eml-cost report --file PATH` / `eml-cost check --file PATH`** —
  read one expression per line from PATH (use `-` for stdin). Lines
  starting with `#` and blank lines are skipped, so the file can hold
  comments.
- **`eml-cost version`** — prints version + the trilogy summary.
- **`[project.scripts] eml-cost = "eml_cost.cli:_entrypoint"`** in
  `pyproject.toml`.

### Pre-commit recipe

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

### Exit codes

| code | meaning |
|------|---------|
| 0    | all checks passed |
| 1    | at least one threshold violation (check only) |
| 2    | at least one expression failed to parse |
| 64   | usage error (no expressions given) |

### Tests

16 new CLI tests (`test_cli.py`) — all green; full suite 194/194.
mypy --strict clean (11 source files).

## [0.7.0] — 2026-04-26 — `predict_precision_loss(expr)` runtime numerical predictor

This release ships the regression model from research session E-193 and
**completes the eml-cost prediction trilogy**: compile-time
(`estimate_time`, 0.6.0), runtime numerical precision
(`predict_precision_loss`, 0.7.0), and per-layer activation behavior
(`eml_cost_torch.diagnose`, 0.5.0). **No breaking API changes**.

### Added

- **`predict_precision_loss(expr)`** — returns predicted float64
  numerical error magnitude vs 50-digit mpmath ground truth, with a
  95% prediction interval and a convenience `predicted_digits_lost`
  field (decimal digits lost relative to perfect 16-digit float64).
- **`PrecisionLossEstimate`** dataclass with fields:
  `predicted_max_relerr`, `predicted_digits_lost`, `ci95`,
  `log10_relerr`, `log10_std`, `features`, `cv_r2`.
- **`precision_loss_model_metadata()`** — provenance for the shipped
  model (n=379, source, features, response, session, and an
  `honest_note` describing the modest CV R^2 and the deliberate
  decision NOT to ship a form-recommender — see Honest framing below).
- **`FLOAT64_EPS`** — exposed as a public constant
  (`2.220446049250313e-16`); the smallest meaningful detectable
  relerr in float64.

### Empirical performance

5-fold cross-validated R^2 on log10(mpmath_max_relerr), seed=42:

| metric                | value         |
|-----------------------|---------------|
| CV R^2 mean           | +0.271        |
| CV R^2 std            | 0.060         |
| residual sigma (log10)| 0.772         |
| Full-fit R^2          | +0.289        |
| n (training)          | 379           |

Coefficients (interpretation: log10 mpmath_max_relerr per unit of
feature):

    intercept       -15.4406
    eml_depth       +0.3367   (~0.34 decimal digits per depth unit)
    max_path_r      -0.1249
    log_count_ops   +0.9443
    log_tree_size   -1.3751

Features: `eml_depth`, `max_path_r`, `log10(count_ops+1)`,
`log10(tree_size+1)` — same pipeline as `estimate_time`.

### Honest framing (also baked into `precision_loss_model_metadata`)

This is a **modest** predictor by design. The underlying signal is
real (E-193 partial r = +0.357 controlling for tree size, q = 1.6e-11)
but moderate; the joint OLS R^2 captures ~27% of the log-variance.

- **Use it for** rank-ordering candidate expressions, surfacing
  high-risk subtrees in a SymPy linter, and quick sanity checks.
- **Do not use it for** absolute precision claims — the CI95 spans
  roughly factor 30 either way.
- **Not a form recommender.** E-193 Phase 3 measured 30% best-pick
  on 10 algebraically-equivalent rewrite tests, well below the 70%
  product threshold. A form recommender was deliberately NOT shipped;
  do not use this function to choose between
  `1/(1+exp(-x))` and `tanh(x/2)/2 + 1/2`.

### Floor handling

50 of 381 corpus rows had measured `mpmath_max_relerr = 0.0`
(float64 matched mpmath bit-for-bit at all 100 sampled points). These
are floored to `FLOAT64_EPS` (~2.22e-16) — the smallest reportable
relerr in float64 — for the regression. Without this floor, log10
sends those rows to `-inf` and the OLS collapses.

### Source

`monogate-research/exploration/E193_numerical_stability/corpus_with_stability.csv`
(403 rows, 379 retained after removing 12 no-mpmath + 12 nan-relerr rows).

Fit script: `monogate-research/exploration/E-193-precision-loss/fit_precision_loss.py`.

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
