from env.models import StepRecord, GradeResult
from tasks.base import TaskConfig
import numpy as np


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_uncontrolled_cost(trajectory: list[StepRecord], config: TaskConfig) -> float:
    """
    Deterministic baseline: no battery, no curtailment.
    All load served from solar first, rest from grid.
    """
    total_cost = 0.0
    dt = 0.25
    for r in trajectory:
        solar = r.state.solar_kw
        base_load = r.state.base_load_kw
        flex_load = r.state.flexible_load_kw
        total_load = base_load + flex_load
        deficit = max(0.0, total_load - solar)
        total_cost += deficit * r.spot_price * dt
    return total_cost


def grade(trajectory: list[StepRecord], config: TaskConfig) -> GradeResult:
    dt = 0.25
    n = len(trajectory)

    # ── 1. Cost Score (0.30) ──────────────────────────────────────────
    baseline_cost = compute_uncontrolled_cost(trajectory, config)
    agent_import_cost = sum(r.import_kw * r.spot_price * dt for r in trajectory)
    agent_export_revenue = sum(r.export_kw * r.spot_price * 0.85 * dt for r in trajectory)
    agent_net_cost = agent_import_cost - agent_export_revenue

    if baseline_cost > 0:
        cost_score = _clamp((baseline_cost - agent_net_cost) / baseline_cost, 0.0, 1.0)
    else:
        cost_score = 1.0

    # ── 2. Blackout Score (0.25) ──────────────────────────────────────
    blackout_steps = sum(1 for r in trajectory if r.unmet_kw > 0.01)
    blackout_score = 1.0 - (blackout_steps / n)

    # ── 3. Self-Consumption Score (0.20) ──────────────────────────────
    total_solar = sum(r.state.solar_kw * dt for r in trajectory)
    total_exported = sum(r.export_kw * dt for r in trajectory)
    total_curtailed = sum(r.solar_curtailed_kw * dt for r in trajectory)
    if total_solar > 0:
        self_consumption_score = 1.0 - _clamp(
            (total_exported + total_curtailed) / total_solar, 0.0, 1.0
        )
    else:
        self_consumption_score = 1.0

    # ── 4. Deferred Load Clearance Score (0.15) ───────────────────────
    final_deferred = trajectory[-1].deferred_kwh
    max_possible_deferred = sum(r.state.flexible_load_kw * dt for r in trajectory)
    if max_possible_deferred > 0:
        clearance_score = 1.0 - _clamp(
            final_deferred / max_possible_deferred, 0.0, 1.0
        )
    else:
        clearance_score = 1.0

    # ── 5. Battery Health Score (0.10) ────────────────────────────────
    total_throughput = sum(abs(r.battery_kw_actual) * dt for r in trajectory)
    safe_throughput = config.battery_capacity_kwh * 1.0
    if total_throughput <= safe_throughput:
        health_score = 1.0
    else:
        health_score = _clamp(
            1.0 - (total_throughput - safe_throughput) / safe_throughput, 0.0, 1.0
        )

    # ── Constraint Violation Penalty (multiplicative) ─────────────────
    soc_violations = sum(1 for r in trajectory if r.soc < 0.05 or r.soc > 0.96)
    violation_penalty = max(0.5, 1.0 - soc_violations * 0.05)

    # ── Final Weighted Score ──────────────────────────────────────────
    raw = (
        0.30 * cost_score +
        0.25 * blackout_score +
        0.20 * self_consumption_score +
        0.15 * clearance_score +
        0.10 * health_score
    )
    final = round(_clamp(raw * violation_penalty, 0.0, 1.0), 4)

    return GradeResult(
        score=final,
        breakdown={
            "cost_efficiency": round(cost_score, 4),
            "blackout_avoidance": round(blackout_score, 4),
            "solar_utilization": round(self_consumption_score, 4),
            "load_clearance": round(clearance_score, 4),
            "battery_health": round(health_score, 4),
            "violation_penalty": round(violation_penalty, 4),
            "weighted_raw": round(raw, 4),
        }
    )