"""Feynman-flavoured symbolic-regression benchmark.

Ten well-known closed-form expressions spanning chain orders 0 to 3.
Each problem is run four ways:

  - **A** — EML GP with **one-sided** chain-order regularizer
    (penalty above ``max_chain_order=3``).
  - **B** — EML GP **without** regularizer (lambda_chain = 0).
  - **C** — random-tree baseline (no evolution).
  - **D** — EML GP with **two-sided** chain-order regularizer
    (penalty on |chain - target|, target inferred via
    :func:`estimate_dynamics`).

The headline numbers:

  - A vs B: does the one-sided regularizer steer search?
  - D vs A: does targeting from data beat upper-bound-only?

Usage
-----

    >>> from eml_cost.regression import benchmark
    >>> table = benchmark.run_benchmark(seeds=(1, 2, 3))
    >>> print(benchmark.format_table(table))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import sympy as sp

from ..regularizer import RegularizerConfig
from .gp import GPConfig, GPResult, random_baseline, search


__all__ = [
    "FeynmanProblem",
    "FEYNMAN_PROBLEMS",
    "ProblemRun",
    "BenchmarkRow",
    "BenchmarkTable",
    "run_benchmark",
    "format_table",
]


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeynmanProblem:
    """A symbolic-regression target with a known closed form."""

    name: str
    formula: Callable[[np.ndarray], np.ndarray]
    sympy_form: sp.Expr
    true_chain_order: int
    sample_low: float = -1.0
    sample_high: float = 1.0
    n_samples: int = 200
    description: str = ""

    def sample(
        self,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = np.linspace(self.sample_low, self.sample_high, self.n_samples)
        # Add small jitter so the search can't memorise grid points.
        x = x + rng.normal(0, 1e-3, x.shape)
        x.sort()
        y = self.formula(x)
        return x, y


_x = sp.Symbol("x", real=True)


def _build_problems() -> list[FeynmanProblem]:
    return [
        FeynmanProblem(
            name="linear",
            formula=lambda x: 2.0 * x + 1.0,
            sympy_form=2 * _x + 1,
            true_chain_order=0,
            sample_low=-2.0, sample_high=2.0,
            description="Newton, F = m*a (1-D scaled)",
        ),
        FeynmanProblem(
            name="quadratic",
            formula=lambda x: 0.5 * x ** 2,
            sympy_form=sp.Rational(1, 2) * _x ** 2,
            true_chain_order=0,
            sample_low=-2.0, sample_high=2.0,
            description="Kinetic energy 0.5*m*v^2 (1-D)",
        ),
        FeynmanProblem(
            name="cubic",
            formula=lambda x: x ** 3 - 3.0 * x,
            sympy_form=_x ** 3 - 3 * _x,
            true_chain_order=0,
            sample_low=-2.0, sample_high=2.0,
            description="Chebyshev T3-shape; pure polynomial",
        ),
        FeynmanProblem(
            name="exp_decay",
            formula=lambda x: np.exp(-x),
            sympy_form=sp.exp(-_x),
            true_chain_order=1,
            sample_low=0.0, sample_high=4.0,
            description="Radioactive decay exp(-lambda*t)",
        ),
        FeynmanProblem(
            name="log_growth",
            formula=lambda x: np.log(1.0 + x),
            sympy_form=sp.log(1 + _x),
            true_chain_order=1,
            sample_low=0.0, sample_high=4.0,
            description="Logarithmic growth log(1+x)",
        ),
        FeynmanProblem(
            name="gaussian",
            formula=lambda x: np.exp(-x ** 2),
            sympy_form=sp.exp(-_x ** 2),
            true_chain_order=1,
            sample_low=-2.0, sample_high=2.0,
            description="Gaussian profile exp(-x^2)",
        ),
        FeynmanProblem(
            name="pendulum",
            formula=lambda x: 2.0 * math.pi * np.sqrt(np.maximum(x, 1e-9)),
            sympy_form=2 * sp.pi * sp.sqrt(_x),
            true_chain_order=1,
            sample_low=0.05, sample_high=4.0,
            description="Pendulum period 2*pi*sqrt(L/g)",
        ),
        FeynmanProblem(
            name="sine_wave",
            formula=lambda x: np.sin(2.0 * x),
            sympy_form=sp.sin(2 * _x),
            true_chain_order=2,
            sample_low=-3.0, sample_high=3.0,
            description="Simple harmonic A*sin(omega*t)",
        ),
        FeynmanProblem(
            name="cos_squared",
            formula=lambda x: np.cos(x ** 2),
            sympy_form=sp.cos(_x ** 2),
            true_chain_order=2,
            sample_low=-2.0, sample_high=2.0,
            description="Fresnel-like cos(x^2)",
        ),
        FeynmanProblem(
            name="damped_osc",
            formula=lambda x: np.sin(3.0 * x) * np.exp(-0.3 * x),
            sympy_form=sp.sin(3 * _x) * sp.exp(-sp.Rational(3, 10) * _x),
            true_chain_order=3,
            sample_low=0.0, sample_high=6.0,
            description="Damped oscillator sin(omega*t)*exp(-gamma*t)",
        ),
    ]


FEYNMAN_PROBLEMS: list[FeynmanProblem] = _build_problems()


# ---------------------------------------------------------------------------
# Per-run + per-problem result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProblemRun:
    """A single seeded run on a single problem under a single condition."""

    problem: str
    condition: str
    seed: int
    mse: float
    chain_order: int
    node_count: int
    generations_run: int
    converged: bool
    elapsed_seconds: float
    expression: str


@dataclass(frozen=True)
class BenchmarkRow:
    """Per-problem mean / hit-rate aggregates across seeds."""

    problem: str
    true_chain_order: int
    a_mean_mse: float
    a_chain_match_rate: float
    a_modal_chain: int
    b_mean_mse: float
    b_chain_match_rate: float
    b_modal_chain: int
    c_mean_mse: float
    c_chain_match_rate: float
    c_modal_chain: int
    d_mean_mse: float = math.inf
    d_chain_match_rate: float = 0.0
    d_modal_chain: int = 0
    d_mean_target: float = -1.0
    n_seeds: int = 0


@dataclass
class BenchmarkTable:
    rows: list[BenchmarkRow] = field(default_factory=list)
    runs: list[ProblemRun] = field(default_factory=list)

    @property
    def overall_a_chain_rate(self) -> float:
        return _hit_rate(self.runs, "A")

    @property
    def overall_b_chain_rate(self) -> float:
        return _hit_rate(self.runs, "B")

    @property
    def overall_c_chain_rate(self) -> float:
        return _hit_rate(self.runs, "C")

    @property
    def overall_d_chain_rate(self) -> float:
        return _hit_rate(self.runs, "D")


# ---------------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------------


def _make_config(
    seed: int,
    *,
    use_regularizer: bool,
    population_size: int,
    n_generations: int,
) -> GPConfig:
    reg = RegularizerConfig(
        lambda_chain=1.0,
        lambda_nodes=0.0,
        lambda_dynamics=0.0,
        lambda_stability=0.0,
        max_chain_order=3,
    ) if use_regularizer else None
    return GPConfig(
        population_size=population_size,
        n_generations=n_generations,
        seed=seed,
        regularizer=reg,
        use_data_dynamics=use_regularizer,
    )


def _modal(values: list[int]) -> int:
    if not values:
        return 0
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _hit_rate(runs: list[ProblemRun], condition: str) -> float:
    matched = 0
    total = 0
    by_name = {p.name: p for p in FEYNMAN_PROBLEMS}
    for r in runs:
        if r.condition != condition:
            continue
        total += 1
        if by_name[r.problem].true_chain_order == r.chain_order:
            matched += 1
    return matched / total if total else 0.0


def _make_config_two_sided(
    seed: int,
    target_chain_order: int,
    *,
    population_size: int,
    n_generations: int,
) -> GPConfig:
    return GPConfig(
        population_size=population_size,
        n_generations=n_generations,
        seed=seed,
        regularizer=RegularizerConfig(
            lambda_chain=1.0,
            target_chain_order=target_chain_order,
        ),
        use_data_dynamics=True,
    )


def _infer_target_chain(x, y) -> int:
    """Use estimate_dynamics(x, y) to derive a target chain order
    for two-sided mode. Falls back to 0 on failure."""
    try:
        from ..data_analyzer import estimate_dynamics
        d = estimate_dynamics(x, y)
        return int(d.estimated_chain_order)
    except Exception:  # noqa: BLE001
        return 0


def run_benchmark(
    problems: Optional[Sequence[FeynmanProblem]] = None,
    seeds: Sequence[int] = (1, 2, 3),
    *,
    population_size: int = 80,
    n_generations: int = 30,
    progress: bool = False,
    include_two_sided: bool = True,
) -> BenchmarkTable:
    """Run the four-condition benchmark and return aggregate stats.

    Per problem and per seed:

      - **A** uses ``RegularizerConfig(lambda_chain=1.0,
        max_chain_order=3)`` plus dynamics-aware initialisation.
      - **B** runs the same engine without any regularizer.
      - **C** samples random trees with a budget matched to A/B.
      - **D** uses ``RegularizerConfig(lambda_chain=1.0,
        target_chain_order=t)`` where ``t`` comes from
        :func:`estimate_dynamics` on ``(x, y)``.

    Pass ``include_two_sided=False`` to run only A/B/C (3 conditions).

    The seed is used identically across A, B, and D so the only
    differences come from regularization. C uses the same seed
    for sampling.
    """
    problems = list(problems) if problems is not None else FEYNMAN_PROBLEMS

    runs: list[ProblemRun] = []
    targets_by_problem: dict[str, list[int]] = {}

    for prob in problems:
        if progress:
            print(f"  [{prob.name}] true_chain={prob.true_chain_order}")
        for seed in seeds:
            x, y = prob.sample(seed)

            # A — one-sided regularizer
            cfg_a = _make_config(
                seed,
                use_regularizer=True,
                population_size=population_size,
                n_generations=n_generations,
            )
            res_a = search(x, y, var_names=["x"], config=cfg_a)
            runs.append(_to_run(prob.name, "A", seed, res_a))

            # B — no regularizer
            cfg_b = _make_config(
                seed,
                use_regularizer=False,
                population_size=population_size,
                n_generations=n_generations,
            )
            res_b = search(x, y, var_names=["x"], config=cfg_b)
            runs.append(_to_run(prob.name, "B", seed, res_b))

            # C — random baseline (compute-matched)
            res_c = random_baseline(
                x, y, var_names=["x"],
                n_samples=population_size * n_generations,
                max_depth=4,
                seed=seed,
            )
            runs.append(_to_run(prob.name, "C", seed, res_c))

            # D — two-sided regularizer with target from data
            if include_two_sided:
                target = _infer_target_chain(x, y)
                targets_by_problem.setdefault(
                    prob.name, []).append(target)
                cfg_d = _make_config_two_sided(
                    seed, target,
                    population_size=population_size,
                    n_generations=n_generations,
                )
                res_d = search(x, y, var_names=["x"], config=cfg_d)
                runs.append(_to_run(prob.name, "D", seed, res_d))

    rows = []
    for prob in problems:
        a_runs = [r for r in runs
                  if r.problem == prob.name and r.condition == "A"]
        b_runs = [r for r in runs
                  if r.problem == prob.name and r.condition == "B"]
        c_runs = [r for r in runs
                  if r.problem == prob.name and r.condition == "C"]
        d_runs = [r for r in runs
                  if r.problem == prob.name and r.condition == "D"]

        d_targets = targets_by_problem.get(prob.name, [])
        d_mean_target = (
            sum(d_targets) / len(d_targets) if d_targets else -1.0)

        rows.append(BenchmarkRow(
            problem=prob.name,
            true_chain_order=prob.true_chain_order,
            a_mean_mse=_safe_mean(r.mse for r in a_runs),
            a_chain_match_rate=_match_rate(a_runs, prob.true_chain_order),
            a_modal_chain=_modal([r.chain_order for r in a_runs]),
            b_mean_mse=_safe_mean(r.mse for r in b_runs),
            b_chain_match_rate=_match_rate(b_runs, prob.true_chain_order),
            b_modal_chain=_modal([r.chain_order for r in b_runs]),
            c_mean_mse=_safe_mean(r.mse for r in c_runs),
            c_chain_match_rate=_match_rate(c_runs, prob.true_chain_order),
            c_modal_chain=_modal([r.chain_order for r in c_runs]),
            d_mean_mse=_safe_mean(r.mse for r in d_runs),
            d_chain_match_rate=_match_rate(d_runs, prob.true_chain_order),
            d_modal_chain=(
                _modal([r.chain_order for r in d_runs]) if d_runs else 0),
            d_mean_target=d_mean_target,
            n_seeds=len(seeds),
        ))

    return BenchmarkTable(rows=rows, runs=runs)


def _to_run(problem: str, condition: str, seed: int,
            res: GPResult) -> ProblemRun:
    return ProblemRun(
        problem=problem,
        condition=condition,
        seed=seed,
        mse=res.mse,
        chain_order=res.chain_order,
        node_count=res.node_count,
        generations_run=res.generations_run,
        converged=res.converged,
        elapsed_seconds=res.elapsed_seconds,
        expression=str(res.expression),
    )


def _safe_mean(it) -> float:
    vals = [v for v in it if math.isfinite(v)]
    if not vals:
        return math.inf
    return float(sum(vals) / len(vals))


def _match_rate(runs: list[ProblemRun], target: int) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.chain_order == target) / len(runs)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def format_table(table: BenchmarkTable) -> str:
    """Render the per-problem results as a markdown table.

    Includes condition D when any D-runs are present in the table.
    """
    has_d = any(r.condition == "D" for r in table.runs)
    headers = [
        "Problem", "True chain",
        "A chain", "A mse",
        "B chain", "B mse",
        "C chain", "C mse",
    ]
    if has_d:
        headers += ["D target", "D chain", "D mse"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in table.rows:
        a_mark = "OK" if row.a_modal_chain == row.true_chain_order else "X"
        b_mark = "OK" if row.b_modal_chain == row.true_chain_order else "X"
        c_mark = "OK" if row.c_modal_chain == row.true_chain_order else "X"
        line = (
            f"| {row.problem} | {row.true_chain_order} | "
            f"{row.a_modal_chain} {a_mark} | {_fmt_mse(row.a_mean_mse)} | "
            f"{row.b_modal_chain} {b_mark} | {_fmt_mse(row.b_mean_mse)} | "
            f"{row.c_modal_chain} {c_mark} | {_fmt_mse(row.c_mean_mse)} |"
        )
        if has_d:
            d_mark = "OK" if row.d_modal_chain == row.true_chain_order else "X"
            target = (f"{row.d_mean_target:.1f}"
                      if row.d_mean_target >= 0 else "-")
            line += (
                f" {target} | {row.d_modal_chain} {d_mark} | "
                f"{_fmt_mse(row.d_mean_mse)} |")
        lines.append(line)

    lines.append("")
    summary = (
        f"**Overall chain-order hit rate:** "
        f"A={table.overall_a_chain_rate * 100:.1f}%  "
        f"B={table.overall_b_chain_rate * 100:.1f}%  "
        f"C={table.overall_c_chain_rate * 100:.1f}%"
    )
    if has_d:
        summary += f"  D={table.overall_d_chain_rate * 100:.1f}%"
    lines.append(summary)
    return "\n".join(lines)


def _fmt_mse(v: float) -> str:
    if not math.isfinite(v):
        return "inf"
    if v == 0.0:
        return "0"
    if v < 1e-3 or v > 1e3:
        return f"{v:.2e}"
    return f"{v:.4g}"
