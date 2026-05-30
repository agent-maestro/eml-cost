# eml-cost registry calibration audit — 2026-05-10

**Scope:** the 50+ entries in `PFAFFIAN_NOT_EML_R` (eml_cost/core.py).
**Trigger:** the polylog miscalibration surfaced by the
`Frontier_C_hypergeometric_chain` probe in monogate-research
(2026-05-10) raised the question: are there other entries with the
same calibration issue?
**Method:** for each entry, cross-check the registry chain value
against (a) the structural chain implied by the function's defining
ODE / integral representation, (b) the "explicit form" expanded out
into elementary primitives, (c) consistency with sister entries.
**Status:** AUDIT ONLY. No fixes shipped (per-fix discussion below
each finding).

## Headline

| Severity | Count | Entries |
|---|---|---|
| ✓ FIXED 2026-05-10 | 1 | `polylog` |
| ⚠ MISCALIBRATED (parameter-blind) | 2 | `polygamma`, `lerchphi` |
| ⚠ STALE (unreachable; sympy normalizes them away) | 2 | `digamma`, `trigamma` |
| ⚠ INCONSISTENT (registry vs explicit-form additivity) | 2 | `dirichlet_eta`, `riemann_xi` |
| ⚠ POSSIBLY OFF (parameter-dependent like polylog) | 1 | `harmonic` |
| ⚠ SEMANTIC (constants should be chain 0) | 1 | `stieltjes(n)` for n ≥ 1 |
| ⚠ DEBATABLE (empirical vs registry mismatch) | 4 | `mathieuc`, `mathieus`, `mathieucprime`, `mathieusprime` |
| ✓ OK | 40 | rest of registry |

Audit identifies **8 entries that warrant follow-up review** in
addition to the polylog fix that already shipped.

## Fix-1: `polylog` — DONE 2026-05-10 (eml-cost 0.20.1, branch
genome-corpus-augmentation)

Single registry value 3 → parameter-aware: `chain(polylog(n, x)) =
n` for integer `n ≥ 1`; falls back to 3 for symbolic / non-integer.

Surfaced by `monogate-research/exploration/Frontier_C_hypergeometric_chain_2026_05_10/`:
₃F₂(1,1,1; 2,2; x) = Li_2(x)/x reported chain 3 instead of the
structurally-correct chain 2. Closed via a shared `_effective_r`
helper applied at all three call sites (`pfaffian_r`,
`predict_chain_order_via_additivity`, `max_path_r`).

## Fix-2 (proposed): `polygamma(n, x)` — same pattern as polylog

**Current:** flat chain 3 regardless of `n`.

**Issue:** `polygamma(n, x) = (-1)^(n+1) · n! · ζ(n+1, x)` (Hurwitz
zeta). Each derivative of digamma adds chain. Structural prediction:

```
polygamma(0, x) = digamma(x):    chain 2  (Γ → ψ chain pair)
polygamma(1, x) = trigamma(x):   chain 3  (additional derivative)
polygamma(n, x) for n ≥ 2:       chain n + 2  (each n adds one)
```

**Currently observed (probe):**

```
polygamma(0, x)   chain 3   ✗  expected 2
polygamma(1, x)   chain 3   ✓  
polygamma(2, x)   chain 3   ✗  expected 4
polygamma(5, x)   chain 3   ✗  expected 7
polygamma(10, x)  chain 3   ✗  expected 12
```

**Additional issue:** the registry separately contains
`"digamma": 2` and `"trigamma": 3` with sister-conflict to
polygamma's 3.

**Proposed fix:** in `_effective_r`, special-case polygamma:

```python
if fname == "polygamma" and len(expr.args) >= 1:
    n = expr.args[0]
    if n.is_Integer and int(n) >= 0:
        return max(int(n) + 2, 2)  # digamma at n=0 is chain 2
```

**Caveat:** I have NOT verified this against differential-Galois
ground truth (Hurwitz zeta is well-studied; the chain formula
should be checked against, e.g., Rosen-Schwarzschild). Predict-then-
verify rather than ship blindly.

## Fix-3 (proposed): `lerchphi(z, s, a)` — same pattern

**Current:** flat chain 4 regardless of `s`.

**Issue:** Lerch transcendent `Φ(z, s, a) = ∑_{k=0}^∞ z^k / (k+a)^s`
generalizes polylog (`Φ(z, s, 1) = polylog(s, z) / z`) and Hurwitz
zeta (`Φ(1, s, a) = ζ(s, a)`). For integer `s ≥ 2` and `a = 1`,
this reduces to polylog(s, z)/z which has chain `s` (post-polylog-
fix).

**Currently observed:**

```
lerchphi(x, 2, 1)   chain 4   ✗  expected 2 (= polylog(2,x)/x)
lerchphi(x, 5, 1)   chain 4   ✗  expected 5
lerchphi(x, 10, 1)  chain 4   ✗  expected 10
```

**Proposed fix:** analogous to polylog. Special-case in
`_effective_r`. More involved than polylog because the s/a/n
relation matters:

```python
if fname == "lerchphi" and len(expr.args) >= 3:
    z, s, a = expr.args[:3]
    if s.is_Integer and int(s) >= 1 and a == 1:
        return int(s)  # reduces to polylog(s, z)/z
```

**Caveat:** for general `a`, lerchphi is the Hurwitz zeta family.
The chain there might be `s` plus a contribution from `a`. Punt
and only special-case the `a == 1` case.

## Fix-4 (proposed): drop stale `digamma` and `trigamma` registry
entries

**Issue:** `sympy.digamma(x)` constructs a `polygamma(0, x)` AST
node. The registry entry `"digamma": 2` is **unreachable** — eml-
cost's chain calculation never sees an AST node with
`func.__name__ == "digamma"`. Same for `trigamma`.

**Proposed fix:** delete both entries, document the redirect in a
comment. Saves confusion.

```python
# Remove from registry:
"digamma": 2,        # unreachable - sympy normalizes to polygamma(0)
"trigamma": 3,       # unreachable - sympy normalizes to polygamma(1)
```

## Fix-5 (proposed): consistency between registry value and
explicit-form additivity

**Affected entries:** `dirichlet_eta`, `riemann_xi`.

**Issue:** When a function is *registered* with a chain value, that
value is used directly. When the same function is *expanded* into
elementary primitives, the additivity rule applies. The two
should agree.

**`dirichlet_eta(s) = (1 - 2^(1-s)) · ζ(s)`:**

```
Currently: dirichlet_eta(s) → chain 4
Explicit:  (1 - 2^(1-s)) · ζ(s) → chain 5
   = chain((1-2^(1-s))) + chain(ζ(s))
   = (chain of non-integer Pow `2^(1-s)`)  +  4
   = 1 + 4 = 5
```

**`riemann_xi(s) = (s(s-1)/2) · π^(-s/2) · Γ(s/2) · ζ(s)`:**

```
Currently: riemann_xi(s) → chain 6
Explicit:  full expansion → chain 7
   = chain(Γ(s/2)) + chain(ζ(s)) + chain(π^(-s/2))
   = 2 + 4 + 1 = 7
```

The current registry value is internally consistent with
*Khovanskii-counting on the OWN ODE generators of riemann_xi /
dirichlet_eta* but inconsistent with the *additivity rule applied
to the explicit form*. Two camps:

  1. **Registry is right.** The chain order of `riemann_xi(s)` is
    a property of its OWN Pfaffian chain (Γ + ζ + auxiliary), not
    of any decomposition. The factor `π^(-s/2)` is "absorbed" into
    the Khovanskii system.

  2. **Explicit-form additivity is right.** Chain order should be a
    function of the AST, not the named symbol. Decomposing
    riemann_xi gives chain 7; the registry value should match.

The polylog fix took position (2). For consistency,
dirichlet_eta and riemann_xi should also take (2):

```
"dirichlet_eta": 5,   # was 4; sums chain(ζ)=4 + chain(2^(1-s))=1
"riemann_xi": 7,      # was 6; sums chain(Γ)=2 + chain(ζ)=4 + chain(π^{-s/2})=1
```

**Cost of the change:** any tests asserting the old chain values
will fail. Worth updating.

## Fix-6 (proposed): `harmonic(n, m)` parameter-aware

**Current:** flat chain 3.

**Issue:** `harmonic(n, m) = ∑_{k=1}^n 1/k^m`. As `n → ∞`,
`harmonic(∞, m) = ζ(m)`. So harmonic is structurally Hurwitz-zeta-
related, with chain depending on `m`.

**Proposed fix:** parameter-aware on `m`, analogous to polygamma:

```python
if fname == "harmonic" and len(expr.args) >= 2:
    n_arg, m = expr.args
    if m.is_Integer and int(m) >= 1:
        return max(int(m) + 1, 2)  # harmonic(n, m) ~ ζ(m)
```

**Caveat:** weaker confidence here than polylog/polygamma. Need to
verify against actual chain analysis of `H_n^{(m)}`. The asymptotic
`→ ζ(m)` argument is suggestive but not rigorous proof of chain.

## Fix-7 (proposed): `stieltjes(n)` for `n ≥ 1` is a constant — chain 0

**Current:** `stieltjes(n)` returns chain 4 for `n ≥ 1` (chain 0
for `n = 0` since `stieltjes(0) = γ` is treated as constant).

**Issue:** Stieltjes constants `γ_n` are real numbers (constants in
the Laurent expansion of `ζ(s)` at `s = 1`). They're not functions
of any variable. Chain order conventionally doesn't apply to
constants — eml-cost returns 0 for `e`, `π`, integer constants,
etc.

**Proposed fix:** return chain 0 for `stieltjes(n)`. Either drop
the registry entry entirely (so eml-cost falls through to its
"unknown atom" handling) or add a special case treating them as
chain-0 constants.

## Fix-8 (NOT proposed): Mathieu functions

**Current:** `mathieuc/mathieus/...prime` at chain 4.

**Empirical observation:** the predecessor `Frontier_C_heun_lame_mathieu`
probe assigned generic Mathieu solutions chain 3 (matching NL_floor).
The Mathieu equation is order 2 with non-Liouvillian generic G° ⊆
SL(2, ℂ), so by C-1ʹ chain ≥ NL_floor = 3. The registry value of 4
exceeds NL_floor.

**Why I don't recommend changing this:** the registry value 4 is a
*safe* over-approximation. C-1ʹ gives a lower bound; the registry
just gives a higher one. Changing 4 → 3 would tighten the chain
estimate but require careful verification that no test or downstream
consumer breaks. Lower priority than the polylog/polygamma/lerchphi
trio which are off by larger margins (3 → n for n up to 10).

If a future probe needs tighter Mathieu chains, adjust then.

## Summary table — proposed fix priority order

| Priority | Entry | Severity | Effort | Risk |
|---|---|---|---|---|
| 1 | `polygamma` | parameter-blind ✗ | small | low — analogous to polylog |
| 2 | `lerchphi(z, s, 1)` integer s | parameter-blind ✗ | small | low |
| 3 | `digamma` / `trigamma` stale entries | dead code | trivial | none |
| 4 | `dirichlet_eta` consistency | minor inconsistency | small | low |
| 5 | `riemann_xi` consistency | minor inconsistency | small | low |
| 6 | `harmonic(n, m)` integer m | possibly off | small | medium — needs verification |
| 7 | `stieltjes(n)` for n ≥ 1 | semantic | trivial | low |
| 8 | Mathieu (4 entries) | over-approximation | small | medium — tighter bound |

## Recommended approach

Ship as **eml-cost 0.21.0** (minor bump — multiple registry
calibration changes). One PR per fix-group:

  - **PR 1**: Fixes 1-3 (polygamma family + stale entry cleanup).
    Same pattern as the polylog fix. Add tests for n ∈ {0, 1, 2,
    5, 10}. Affects polygamma, drops digamma/trigamma.
  - **PR 2**: Fix 4 (lerchphi parameter-aware). Add tests for
    integer `s`. Affects lerchphi only.
  - **PR 3**: Fixes 5-6 (consistency: dirichlet_eta + riemann_xi).
    Update tests asserting old values. Affects 2 entries.
  - **PR 4**: Fixes 7-8 (harmonic + stieltjes + Mathieu). Lower
    priority; defer until any consumer cares.

Any user-visible behavior change should be documented in CHANGELOG
under "BREAKING (registry calibration)" — not a true API break,
but downstream callers comparing chain values to specific integers
may be affected.

## Cross-cutting improvement: replace the registry with a
parameter-aware classifier

The pattern emerging from this audit (polylog, polygamma, lerchphi,
harmonic) is that **functions parameterized by an integer order**
have parameter-dependent chain. The registry's flat int-per-fname
calibration is fundamentally a 1980s-style symbol table.

A more structurally-clean approach would be to register CLASSIFIER
FUNCTIONS rather than ints:

```python
PFAFFIAN_NOT_EML_R: dict[str, Callable[[sp.Basic], int]] = {
    "polylog": lambda expr: int(expr.args[0]) if expr.args[0].is_Integer
                            and int(expr.args[0]) >= 1 else 3,
    "polygamma": lambda expr: max(int(expr.args[0]) + 2, 2)
                              if expr.args[0].is_Integer
                              and int(expr.args[0]) >= 0 else 3,
    # ...
}
```

This is a deeper refactor (eml-cost 0.22+ probably). For now, the
shared `_effective_r(expr, fname)` helper introduced for polylog
is a stepping stone to the classifier-based design.

## Files

  - `src/eml_cost/core.py` — the registry + `_effective_r` helper
  - `tests/test_extended_registry.py` — current polylog test;
    proposed tests for polygamma + lerchphi to follow

## Sequel work

  1. Implement Fixes 1-3 in a focused PR (polygamma + stale cleanup).
  2. Re-run the monogate-research Frontier C corpus — any existing
    "₃F₂(...)" reductions involving polygamma/lerchphi should refresh.
  3. Cross-check Fix 6 (harmonic) against the actual differential-
    Galois chain calculation for generalized harmonic numbers.
  4. Plan eml-cost 0.22 classifier-based refactor as the next major
    change.
