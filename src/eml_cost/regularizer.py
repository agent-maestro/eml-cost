"""Chain-order regularizer for EML-native symbolic regression.

Penalises candidate expressions by:

  - **chain_penalty** — chain order beyond ``max_chain_order``
  - **node_penalty** — total node count (parsimony)
  - **dynamics_penalty** — mismatch between predicted oscillation /
    decay structure and the user-supplied ``expected_dynamics``
  - **stability_penalty** — divergence from the
    :func:`recommend_form` canonical-stable form for the four
    supported families (sigmoid / exp_decay / logistic / cosine)

The total penalty is a weighted sum of the four. Each component
is explainable: ``RegularizerResult.explanation`` returns a short
human-readable string with the per-component contribution.

The regularizer is **structural** — it does not fit data. It is
intended as the steering term in a symbolic-regression search
loop where the data-fit loss is computed separately. Use
:func:`eml_cost.estimate_dynamics` on the regression target to
populate ``expected_dynamics``.

Honest framing
--------------

Chain-order is **structural transcendental nesting**. It is
correlated with — but not identical to — numerical-approximation
cost (Spearman rho ~0.4 against Chebyshev degree on a 35-expression
probe; see ``monogate-research/exploration/zk-circuit-cost-2026-04-28``).
The regularizer biases search toward simpler EML-class forms;
it does not guarantee numerical stability or polynomial-degree
minimality.

Usage
-----

    >>> from eml_cost import regularize, RegularizerConfig
    >>> cfg = RegularizerConfig(
    ...     lambda_chain=1.0,
    ...     lambda_nodes=0.1,
    ...     max_chain_order=3,
    ... )
    >>> r = regularize("sin(exp(cos(x)))", cfg)
    >>> r.is_feasible      # chain_order > 3
    False
    >>> r.chain_penalty > 0
    True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import sympy as sp

from .analyze import analyze
from .canonicalize import canonicalize
from .core import predict_chain_order_via_additivity
from .recommend_form import recommend_form


__all__ = [
    "RegularizerConfig",
    "RegularizerResult",
    "regularize",
]


@dataclass(frozen=True)
class RegularizerConfig:
    """Tunable weights and constraints for :func:`regularize`.

    All four lambdas default to zero so the regularizer can be
    used as a pure feasibility check without any soft penalty.

    Attributes
    ----------
    lambda_chain:
        Weight on the chain-order excess (``max(0, chain_order -
        max_chain_order)``). Positive values steer search toward
        shallower transcendental nesting.
    lambda_nodes:
        Weight on raw node count (``count_ops`` over the canonical
        form). Encourages parsimony.
    lambda_dynamics:
        Weight on dynamics-mismatch penalty. Requires
        ``expected_dynamics`` to be populated; otherwise the
        component is zero.
    lambda_stability:
        Weight on a recommend_form distance term. When the
        recommender fires on the input and offers a canonical
        form, the penalty is the predicted ``digits_saved`` (so
        an input form that is *worse* than the canonical pays).
    max_chain_order:
        Soft constraint on chain order. Expressions with chain
        order strictly greater than this are flagged
        ``is_feasible=False``.
    expected_dynamics:
        Optional ``(n_oscillations, n_decays)`` tuple — the
        dynamics signature inferred from the regression target.
        Compared against the candidate's predicted dynamics
        counter; mismatch contributes to ``dynamics_penalty``.
    """

    lambda_chain: float = 0.0
    lambda_nodes: float = 0.0
    lambda_dynamics: float = 0.0
    lambda_stability: float = 0.0
    max_chain_order: int = 5
    expected_dynamics: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class RegularizerResult:
    """Per-component penalty breakdown for a single candidate.

    Attributes
    ----------
    chain_penalty:
        ``lambda_chain * max(0, chain_order - max_chain_order)``.
    node_penalty:
        ``lambda_nodes * node_count``.
    dynamics_penalty:
        ``lambda_dynamics * |predicted_dynamics - expected_dynamics|_1``
        when ``expected_dynamics`` is supplied; else 0.
    stability_penalty:
        ``lambda_stability * max(0, digits_saved)`` when
        :func:`recommend_form` finds a match; else 0.
    total_penalty:
        Sum of the four components.
    chain_order:
        Output of :func:`predict_chain_order_via_additivity`.
    node_count:
        ``sp.count_ops(canonical_expr)``.
    predicted_dynamics:
        ``(n_osc, n_decay)`` derived from the candidate's analyze
        corrections.
    is_feasible:
        ``chain_order <= max_chain_order``.
    explanation:
        One-line human summary of the penalty breakdown.
    """

    chain_penalty: float
    node_penalty: float
    dynamics_penalty: float
    stability_penalty: float
    total_penalty: float
    chain_order: int
    node_count: int
    predicted_dynamics: tuple[int, int]
    is_feasible: bool
    explanation: str = ""


def _predicted_dynamics(result) -> tuple[int, int]:
    """Map an :class:`AnalyzeResult` to ``(n_oscillations, n_decays)``.

    The dynamics counter from C-207 is

        r ~= 2 * n_osc + 1 * n_decay + r_tower_base

    so we can extract ``n_osc`` from ``c_osc`` (the path-restricted
    oscillation count) and ``n_decay`` from ``c_composite + 1``
    when there is at least one ``exp`` or ``log`` along the path
    that wasn't fused. We expose a coarse two-feature signature
    that matches the data-side estimator's output.
    """
    n_osc = result.corrections.c_osc
    # n_decay is the remaining transcendental contribution after
    # subtracting oscillation pairs and the structural backbone.
    n_decay = max(0, result.max_path_r - 2 * n_osc)
    return (n_osc, n_decay)


def regularize(
    expr: Union[str, sp.Basic],
    config: Optional[RegularizerConfig] = None,
) -> RegularizerResult:
    """Score a candidate expression under the chain-order regularizer.

    Parameters
    ----------
    expr:
        SymPy expression or parseable string.
    config:
        Optional :class:`RegularizerConfig`. ``None`` uses default
        weights (all zero) — useful for getting structural
        diagnostics without any soft penalty.

    Returns
    -------
    RegularizerResult

    Examples
    --------
    >>> from eml_cost import regularize, RegularizerConfig
    >>> r = regularize("sin(x)", RegularizerConfig(max_chain_order=3))
    >>> r.is_feasible
    True
    >>> r.chain_order
    2
    """
    if config is None:
        config = RegularizerConfig()

    canonical = canonicalize(expr)
    result = analyze(canonical)

    chain_order = predict_chain_order_via_additivity(canonical)
    node_count = int(sp.count_ops(canonical))
    pred_dyn = _predicted_dynamics(result)

    # Component 1: chain penalty (excess over the budget).
    excess = max(0, chain_order - config.max_chain_order)
    chain_penalty = float(config.lambda_chain * excess)

    # Component 2: parsimony / node count.
    node_penalty = float(config.lambda_nodes * node_count)

    # Component 3: dynamics mismatch (L1 distance on the (n_osc,
    # n_decay) signature).
    if config.expected_dynamics is not None:
        exp_osc, exp_dec = config.expected_dynamics
        mismatch = abs(pred_dyn[0] - exp_osc) + abs(pred_dyn[1] - exp_dec)
        dynamics_penalty = float(config.lambda_dynamics * mismatch)
    else:
        dynamics_penalty = 0.0

    # Component 4: stability — penalise when the input is *worse*
    # than the recommender's canonical form for one of the four
    # supported families. ``digits_saved`` > 0 means the canonical
    # form is more stable, so the input pays.
    stability_penalty = 0.0
    if config.lambda_stability:
        try:
            rec = recommend_form(canonical)
        except Exception:  # noqa: BLE001
            rec = None
        if rec is not None and rec.digits_saved > 0:
            stability_penalty = float(
                config.lambda_stability * rec.digits_saved)

    total = (chain_penalty + node_penalty
             + dynamics_penalty + stability_penalty)

    is_feasible = chain_order <= config.max_chain_order

    parts = []
    if chain_penalty:
        parts.append(
            f"chain={chain_penalty:.3f} (order={chain_order}, "
            f"max={config.max_chain_order})")
    if node_penalty:
        parts.append(f"nodes={node_penalty:.3f} (n={node_count})")
    if dynamics_penalty:
        parts.append(
            f"dynamics={dynamics_penalty:.3f} "
            f"(predicted={pred_dyn}, expected={config.expected_dynamics})")
    if stability_penalty:
        parts.append(f"stability={stability_penalty:.3f}")
    if not parts:
        parts.append("zero penalty")
    explanation = "total={:.3f}; ".format(total) + ", ".join(parts)

    return RegularizerResult(
        chain_penalty=chain_penalty,
        node_penalty=node_penalty,
        dynamics_penalty=dynamics_penalty,
        stability_penalty=stability_penalty,
        total_penalty=total,
        chain_order=chain_order,
        node_count=node_count,
        predicted_dynamics=pred_dyn,
        is_feasible=is_feasible,
        explanation=explanation,
    )
