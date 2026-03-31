from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger(__name__)


def softmax_allocate(
    scores: Dict[str, float],
    total_capital: Decimal,
    temperature: float = 2.0,
    min_alloc_pct: float = 5.0,
    max_alloc_pct: float = 40.0,
    bench_threshold: float = 30.0,
    min_active: int = 2,
) -> Dict[str, Decimal]:
    if not scores:
        return {}

    # Step 1: Filter benched agents (keep minimum active)
    sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    active = [(aid, s) for aid, s in sorted_agents if s >= bench_threshold]
    if len(active) < min_active:
        active = sorted_agents[:min_active]

    if not active:
        return {}

    agent_ids = [a[0] for a in active]
    agent_scores = [a[1] for a in active]

    # Step 2: Normalize to [0, 1]
    min_s = min(agent_scores)
    max_s = max(agent_scores)
    span = max_s - min_s if max_s != min_s else 1.0
    normalized = [(s - min_s) / span for s in agent_scores]

    # Step 3: Temperature-scaled softmax
    exp_values = [math.exp(n / temperature) for n in normalized]
    exp_sum = sum(exp_values)
    weights = [e / exp_sum for e in exp_values]

    # Step 4: Clamp to [min, max]
    min_w = min_alloc_pct / 100.0
    max_w = max_alloc_pct / 100.0
    clamped = [max(min_w, min(max_w, w)) for w in weights]

    # Step 5: Redistribute excess
    total_w = sum(clamped)
    if total_w > 0:
        clamped = [w / total_w for w in clamped]

    # Step 6: Convert to KRW allocations
    result: Dict[str, Decimal] = {}
    for aid, w in zip(agent_ids, clamped):
        result[aid] = (Decimal(str(w)) * total_capital).quantize(Decimal("1"))

    # Benched agents get 0
    for aid, s in sorted_agents:
        if aid not in result:
            result[aid] = Decimal("0")

    LOGGER.info("Allocation: %s", {k: f"{v:,.0f}" for k, v in result.items()})
    return result
