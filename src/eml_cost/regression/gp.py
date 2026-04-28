"""Genetic-programming search engine for EML symbolic regression.

The engine evolves a population of :class:`EMLNode` trees against
a numerical target ``(X, y)``. Each candidate is scored by
:meth:`Fitness.evaluate`, which combines:

  - **MSE** of the candidate's predictions on the training set.
  - **Regularizer penalty** computed by :func:`eml_cost.regularize`
    on the canonicalised SymPy form (when ``GPConfig.regularizer``
    is supplied).

Fitness is **lower-is-better** (MSE + lambda * penalty). Selection
uses Koza-style tournaments. Genetic operators are subtree
crossover, subtree mutation, and constant-perturbation mutation.

Smart initialization
--------------------

When ``GPConfig.use_data_dynamics`` is ``True`` and the input is a
single-variable problem, the engine calls :func:`estimate_dynamics`
on the training data to bias the initial population:

  - oscillation modes detected → the initial trees are seeded with
    extra ``sin`` / ``cos`` operators.
  - decay modes detected → seeded with extra ``exp`` operators.

This is a shallow heuristic, not a guarantee. The population
remains diverse so the regularizer is doing the steering work.

Honest framing
--------------

This is a **reference implementation** — small population, fast
turnaround. It is not competitive with PySR or DEAP on raw
performance. The point of the engine is to provide a clean
substrate for measuring whether the chain-order regularizer
actually steers search; the benchmark in
:mod:`eml_cost.regression.benchmark` is the validation surface.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import sympy as sp

from ..regularizer import RegularizerConfig, RegularizerResult, regularize
from .nodes import (
    BINARY_OPS,
    EMLNode,
    TERMINALS_DEFAULT_CONSTS,
    UNARY_OPS,
    all_subtree_indices,
    get_subtree,
    random_tree,
    replace_subtree,
)


__all__ = [
    "GPConfig",
    "GPResult",
    "search",
]


@dataclass
class GPConfig:
    """Search-engine knobs.

    Attributes
    ----------
    population_size:
        How many candidates per generation.
    n_generations:
        Maximum generation count.
    tournament_size:
        Number of randomly drawn candidates per tournament.
    crossover_rate:
        Probability of producing each offspring via subtree crossover
        (vs cloning).
    mutation_rate:
        Probability of applying subtree mutation to each offspring.
    constant_mutation_rate:
        Probability of perturbing one constant in the offspring.
    init_max_depth:
        Maximum depth of randomly initialised trees.
    max_depth:
        Hard ceiling on depth after crossover (offspring exceeding
        the ceiling are reverted to the parent).
    early_stop_mse:
        Stop early when the population's best MSE falls below this.
    regularizer:
        Optional :class:`RegularizerConfig`. When provided, every
        candidate's fitness is augmented by ``regularize(...).total_penalty``.
    use_data_dynamics:
        If True (default) and the input is single-variable, seed
        the initial population using :func:`estimate_dynamics`.
    seed:
        RNG seed for reproducibility.
    """

    population_size: int = 80
    n_generations: int = 30
    tournament_size: int = 4
    crossover_rate: float = 0.7
    mutation_rate: float = 0.25
    constant_mutation_rate: float = 0.1
    init_max_depth: int = 4
    max_depth: int = 7
    early_stop_mse: float = 1e-8
    regularizer: Optional[RegularizerConfig] = None
    use_data_dynamics: bool = True
    seed: Optional[int] = None
    constants: tuple[float, ...] = TERMINALS_DEFAULT_CONSTS
    allowed_unary: tuple[str, ...] = UNARY_OPS


@dataclass
class GPResult:
    """Outcome of a search run."""

    expression: sp.Expr
    tree: EMLNode
    chain_order: int
    node_count: int
    mse: float
    penalty: float
    fitness: float
    generations_run: int
    best_mse_history: list[float] = field(default_factory=list)
    best_penalty_history: list[float] = field(default_factory=list)
    converged: bool = False
    elapsed_seconds: float = 0.0


def _mse(pred: np.ndarray, y: np.ndarray) -> float:
    if pred.shape != y.shape:
        return math.inf
    if not np.all(np.isfinite(pred)):
        return math.inf
    return float(np.mean((pred - y) ** 2))


def _evaluate_candidate(
    tree: EMLNode,
    env: dict[str, np.ndarray],
    y: np.ndarray,
    reg_config: Optional[RegularizerConfig],
) -> tuple[float, float, float, int, int]:
    """Return (fitness, mse, penalty, chain_order, node_count).

    ``fitness = mse + penalty`` (lower is better)."""
    try:
        pred = tree.evaluate(env)
    except Exception:  # noqa: BLE001
        return (math.inf, math.inf, 0.0, 0, tree.size())

    mse = _mse(pred, y)
    if not math.isfinite(mse):
        return (math.inf, math.inf, 0.0, 0, tree.size())

    chain_order = 0
    node_count = tree.size()
    penalty = 0.0
    if reg_config is not None:
        try:
            sympy_expr = tree.to_sympy()
            r: RegularizerResult = regularize(sympy_expr, reg_config)
            penalty = r.total_penalty
            chain_order = r.chain_order
            node_count = r.node_count
        except Exception:  # noqa: BLE001
            penalty = 0.0
    else:
        # Always compute chain_order for reporting, even when no
        # penalty is applied. Use a zero-weight config so the
        # number is available without affecting fitness.
        try:
            from ..core import predict_chain_order_via_additivity
            from ..canonicalize import canonicalize
            chain_order = int(
                predict_chain_order_via_additivity(canonicalize(tree.to_sympy())))
        except Exception:  # noqa: BLE001
            chain_order = 0

    return (mse + penalty, mse, penalty, chain_order, node_count)


def _tournament_select(
    population: list[tuple[float, EMLNode]],
    rng: random.Random,
    k: int,
) -> EMLNode:
    contenders = rng.sample(population, min(k, len(population)))
    contenders.sort(key=lambda e: e[0])
    return contenders[0][1].copy()


def _subtree_crossover(
    parent_a: EMLNode,
    parent_b: EMLNode,
    rng: random.Random,
    max_depth: int,
) -> EMLNode:
    paths_a = all_subtree_indices(parent_a)
    paths_b = all_subtree_indices(parent_b)
    path_a = rng.choice(paths_a)
    path_b = rng.choice(paths_b)
    donor = get_subtree(parent_b, path_b)
    candidate = replace_subtree(parent_a, path_a, donor)
    if candidate.depth() > max_depth:
        return parent_a.copy()
    return candidate


def _subtree_mutate(
    tree: EMLNode,
    rng: random.Random,
    var_names: list[str],
    config: GPConfig,
) -> EMLNode:
    paths = all_subtree_indices(tree)
    path = rng.choice(paths)
    new_sub = random_tree(
        rng, var_names,
        max_depth=max(2, config.init_max_depth - len(path)),
        method="grow",
        constants=config.constants,
        allowed_unary=config.allowed_unary,
    )
    candidate = replace_subtree(tree, path, new_sub)
    if candidate.depth() > config.max_depth:
        return tree.copy()
    return candidate


def _constant_mutate(tree: EMLNode, rng: random.Random) -> EMLNode:
    out = tree.copy()
    consts: list[EMLNode] = []
    stack = [out]
    while stack:
        node = stack.pop()
        if node.kind == "const":
            consts.append(node)
        stack.extend(node.children)
    if not consts:
        return out
    target = rng.choice(consts)
    target.value = float(target.value) + rng.gauss(0, 0.5)
    return out


def _seed_for_dynamics(
    var_names: list[str],
    expected_dyn: tuple[int, int],
    rng: random.Random,
    config: GPConfig,
) -> EMLNode:
    """Build a tree biased toward the expected (n_osc, n_decay)."""
    n_osc, n_dec = expected_dyn
    body: Optional[EMLNode] = None
    primary = var_names[0]

    for _ in range(n_osc):
        var_node = EMLNode.var(rng.choice(var_names))
        unary = rng.choice(("sin", "cos"))
        osc = EMLNode.unary(unary, EMLNode.binary(
            "*", EMLNode.const(rng.choice(config.constants)), var_node))
        body = osc if body is None else EMLNode.binary("+", body, osc)

    for _ in range(n_dec):
        var_node = EMLNode.var(rng.choice(var_names))
        decay = EMLNode.unary("exp", EMLNode.unary("neg", EMLNode.binary(
            "*", EMLNode.const(abs(rng.choice(config.constants)) + 0.1),
            var_node)))
        body = decay if body is None else EMLNode.binary("*", body, decay)

    if body is None:
        body = EMLNode.var(primary)
    return body


def search(
    X: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    var_names: Optional[Sequence[str]] = None,
    config: Optional[GPConfig] = None,
) -> GPResult:
    """Run genetic-programming symbolic regression on ``(X, y)``.

    Parameters
    ----------
    X:
        Either a 1-D / 2-D numpy array of shape ``(n_samples,)`` /
        ``(n_samples, n_features)``, or a dict mapping variable
        name to a 1-D array.
    y:
        Target values, shape ``(n_samples,)``.
    var_names:
        Variable names. Required when ``X`` is an array.
    config:
        Optional :class:`GPConfig`. ``None`` uses defaults.
    """
    if config is None:
        config = GPConfig()

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    # Normalise input.
    if isinstance(X, dict):
        env = {k: np.asarray(v, dtype=float) for k, v in X.items()}
        var_names = list(env.keys())
    else:
        arr = np.asarray(X, dtype=float)
        if var_names is None:
            raise ValueError("var_names is required when X is an array")
        if arr.ndim == 1:
            env = {var_names[0]: arr}
        else:
            if arr.shape[1] != len(var_names):
                raise ValueError(
                    "X has {} columns but {} var_names given".format(
                        arr.shape[1], len(var_names)))
            env = {n: arr[:, i] for i, n in enumerate(var_names)}

    var_names_list = list(var_names)
    y_arr = np.asarray(y, dtype=float)

    # Smart initialization via estimate_dynamics on single-var problems.
    expected_dyn: Optional[tuple[int, int]] = None
    if config.use_data_dynamics and len(var_names_list) == 1:
        try:
            from ..data_analyzer import estimate_dynamics
            primary = env[var_names_list[0]]
            d = estimate_dynamics(primary, y_arr)
            expected_dyn = (d.n_oscillations, d.n_decays)
        except Exception:  # noqa: BLE001
            expected_dyn = None

    # Build initial population: ~10% seeded from dynamics, rest random.
    population: list[EMLNode] = []
    n_seeded = 0
    if expected_dyn is not None and (
            expected_dyn[0] > 0 or expected_dyn[1] > 0):
        n_seeded = max(1, config.population_size // 10)
        for _ in range(n_seeded):
            population.append(_seed_for_dynamics(
                var_names_list, expected_dyn, rng, config))
    while len(population) < config.population_size:
        population.append(random_tree(
            rng, var_names_list,
            max_depth=config.init_max_depth,
            constants=config.constants,
            allowed_unary=config.allowed_unary,
        ))

    # Score and run evolution.
    start = time.time()
    scored: list[tuple[float, EMLNode]] = []
    best_mse_history: list[float] = []
    best_penalty_history: list[float] = []
    best_record: Optional[
        tuple[float, float, float, int, int, EMLNode]] = None
    generations_run = 0
    converged = False

    for tree in population:
        f, m, p, c, n = _evaluate_candidate(
            tree, env, y_arr, config.regularizer)
        scored.append((f, tree))
        if best_record is None or f < best_record[0]:
            best_record = (f, m, p, c, n, tree)

    for gen in range(config.n_generations):
        generations_run = gen + 1
        next_pop: list[EMLNode] = []
        # Elitism: carry over best.
        scored.sort(key=lambda e: e[0])
        next_pop.append(scored[0][1].copy())

        while len(next_pop) < config.population_size:
            parent_a = _tournament_select(
                scored, rng, config.tournament_size)
            if rng.random() < config.crossover_rate:
                parent_b = _tournament_select(
                    scored, rng, config.tournament_size)
                child = _subtree_crossover(
                    parent_a, parent_b, rng, config.max_depth)
            else:
                child = parent_a.copy()
            if rng.random() < config.mutation_rate:
                child = _subtree_mutate(child, rng, var_names_list, config)
            if rng.random() < config.constant_mutation_rate:
                child = _constant_mutate(child, rng)
            next_pop.append(child)

        # Re-score.
        scored = []
        for tree in next_pop:
            f, m, p, c, n = _evaluate_candidate(
                tree, env, y_arr, config.regularizer)
            scored.append((f, tree))
            if best_record is None or f < best_record[0]:
                best_record = (f, m, p, c, n, tree)

        scored.sort(key=lambda e: e[0])
        best_mse_history.append(best_record[1])
        best_penalty_history.append(best_record[2])

        if best_record[1] < config.early_stop_mse:
            converged = True
            break

    elapsed = time.time() - start

    if best_record is None:
        # No evaluable candidate at all — return a constant 0.
        zero = EMLNode.const(0.0)
        return GPResult(
            expression=sp.Float(0),
            tree=zero,
            chain_order=0,
            node_count=1,
            mse=math.inf,
            penalty=0.0,
            fitness=math.inf,
            generations_run=generations_run,
            best_mse_history=best_mse_history,
            best_penalty_history=best_penalty_history,
            converged=False,
            elapsed_seconds=elapsed,
        )

    f, m, p, c, n, tree = best_record
    try:
        sympy_expr = tree.to_sympy()
    except Exception:  # noqa: BLE001
        sympy_expr = sp.Float(0)

    return GPResult(
        expression=sympy_expr,
        tree=tree,
        chain_order=c,
        node_count=n,
        mse=m,
        penalty=p,
        fitness=f,
        generations_run=generations_run,
        best_mse_history=best_mse_history,
        best_penalty_history=best_penalty_history,
        converged=converged,
        elapsed_seconds=elapsed,
    )


def random_baseline(
    X: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    var_names: Optional[Sequence[str]] = None,
    *,
    n_samples: int = 2400,
    max_depth: int = 4,
    seed: Optional[int] = None,
    constants: tuple[float, ...] = TERMINALS_DEFAULT_CONSTS,
    allowed_unary: tuple[str, ...] = UNARY_OPS,
) -> GPResult:
    """Sample random trees and pick the one with lowest MSE.

    Compute budget is matched to a comparable GP run by sampling
    ``population_size * n_generations`` trees by default.
    """
    rng = random.Random(seed)
    if isinstance(X, dict):
        env = {k: np.asarray(v, dtype=float) for k, v in X.items()}
        var_names_list = list(env.keys())
    else:
        arr = np.asarray(X, dtype=float)
        if var_names is None:
            raise ValueError("var_names is required when X is an array")
        if arr.ndim == 1:
            env = {var_names[0]: arr}
        else:
            env = {n: arr[:, i] for i, n in enumerate(var_names)}
        var_names_list = list(var_names)

    y_arr = np.asarray(y, dtype=float)
    start = time.time()
    best: Optional[tuple[float, EMLNode, int, int]] = None

    for _ in range(n_samples):
        tree = random_tree(
            rng, var_names_list,
            max_depth=max_depth,
            constants=constants,
            allowed_unary=allowed_unary,
        )
        f, m, p, c, n = _evaluate_candidate(tree, env, y_arr, None)
        if best is None or m < best[0]:
            best = (m, tree, c, n)

    elapsed = time.time() - start
    if best is None:
        return GPResult(
            expression=sp.Float(0), tree=EMLNode.const(0.0),
            chain_order=0, node_count=1, mse=math.inf, penalty=0.0,
            fitness=math.inf, generations_run=0,
            elapsed_seconds=elapsed,
        )

    m, tree, c, n = best
    try:
        expr = tree.to_sympy()
    except Exception:  # noqa: BLE001
        expr = sp.Float(0)
    return GPResult(
        expression=expr, tree=tree, chain_order=c, node_count=n,
        mse=m, penalty=0.0, fitness=m, generations_run=0,
        converged=False, elapsed_seconds=elapsed,
    )
