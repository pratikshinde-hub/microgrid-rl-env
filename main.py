import uuid
import logging
from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env.microgrid import MicrogridEnv
from env.models import MicrogridAction, MicrogridState, GradeResult
from env.grader import grade
from tasks import load_task, TASKS
from baseline.heuristic import ThresholdHeuristicBaseline

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("microgrid")

# ── App Init ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Microgrid RL Environment",
    description=(
        "OpenEnv-compliant Virtual Power Plant dispatch environment. "
        "An agent controls a battery + flexible loads to minimize cost "
        "and prevent blackouts across 3 difficulty tasks. "
        "All primary endpoints support both GET and POST."
    ),
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bounded Session Store ─────────────────────────────────────────────
# OrderedDict gives O(1) oldest-key eviction (FIFO).
# Max 100 sessions — prevents memory leak on HF Spaces.

MAX_SESSIONS = 100
env_store: OrderedDict[str, MicrogridEnv] = OrderedDict()


def _add_session(session_id: str, env: MicrogridEnv) -> None:
    """Insert session, evicting oldest if store is full."""
    if len(env_store) >= MAX_SESSIONS:
        oldest_key, _ = next(iter(env_store.items()))
        env_store.pop(oldest_key)
        log.info(f"Session store full. Evicted oldest session: {oldest_key}")
    env_store[session_id] = env


def _default_task_id() -> str:
    """Returns the first available task_id. Safe fallback for GET endpoints."""
    return next(iter(TASKS.keys()))


# ── Startup Validation ────────────────────────────────────────────────

@app.on_event("startup")
def validate_tasks_on_startup():
    log.info("Running startup task validation...")
    for task_id in TASKS:
        try:
            cfg = load_task(task_id)
            assert cfg.total_steps == 96,         f"total_steps must be 96, got {cfg.total_steps}"
            assert cfg.battery_capacity_kwh > 0,  f"battery_capacity_kwh must be > 0"
            assert 0.0 < cfg.flex_fraction < 1.0, f"flex_fraction must be in (0, 1)"
            assert cfg.max_charge_kw > 0,         f"max_charge_kw must be > 0"
            assert cfg.solar_capacity_kw > 0,     f"solar_capacity_kw must be > 0"
            log.info(f"  ✓ Task '{task_id}' ({cfg.difficulty}) validated.")
        except Exception as e:
            # Log the failure but do NOT raise — app continues running.
            log.error(f"  ✗ Task '{task_id}' config invalid: {e}")
    log.info(f"Startup validation complete. {len(TASKS)} tasks loaded.")


# ══════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42

class ResetResponse(BaseModel):
    session_id: str
    state: MicrogridState
    task_info: dict

class StepRequest(BaseModel):
    session_id: str
    action: MicrogridAction

class StepResponse(BaseModel):
    state: MicrogridState
    reward: float
    done: bool
    info: dict

class GraderRequest(BaseModel):
    session_id: str

class BaselineRequest(BaseModel):
    state: MicrogridState
    task_id: str

class BaselineResponse(BaseModel):
    action: MicrogridAction
    reasoning: str


# ══════════════════════════════════════════════════════════════════════
# META — Pure GET, no session required
# ══════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Meta"])
def root():
    """Entry point. Returns all available endpoints and HTTP methods."""
    return {
        "name": "Microgrid RL Environment",
        "version": "1.0.0",
        "openenv_compliant": True,
        "docs_ui": "/docs",
        "active_sessions": len(env_store),
        "endpoints": {
            "GET (no session required)": [
                "/",
                "/health",
                "/env/info",
                "/tasks",
                "/tasks/{task_id}",
                "/reset",
                "/baseline",
                "/baseline_run",
            ],
            "GET (session required)": [
                "/state/{session_id}",
                "/session/{session_id}/info",
                "/step?session_id=...",
                "/grader?session_id=...",
            ],
            "POST (primary agent interface)": [
                "/reset",
                "/step",
                "/grader",
                "/baseline",
            ],
        },
    }


@app.get("/health", tags=["Meta"])
def health():
    """
    Liveness + readiness check. Always returns 200 if server is running.
    First endpoint automated validators should call.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "tasks_loaded": list(TASKS.keys()),
        "total_steps_per_episode": 96,
        "dt_minutes": 15,
        "active_sessions": len(env_store),
        "max_sessions": MAX_SESSIONS,
    }


@app.get("/env/info", tags=["Meta"])
def env_info():
    """
    Full environment spec: state space, action space, reward structure, grader weights.
    Validators use this to confirm OpenEnv schema compliance without running an episode.
    """
    return {
        "state_space": {
            "solar_kw":             "float  — current PV output [0, solar_capacity_kw]",
            "solar_forecast_kw":    "float  — noisy 1-step-ahead solar forecast",
            "soc":                  "float  — battery state of charge [0.0, 1.0]",
            "battery_capacity_kwh": "float  — nominal capacity, fixed per episode",
            "base_load_kw":         "float  — non-curtailable demand this step",
            "flexible_load_kw":     "float  — curtailable demand this step",
            "deferred_kwh":         "float  — accumulated unserved flexible load",
            "grid_available":       "bool   — False during outage window",
            "spot_price":           "float  — $/kWh energy price this step",
            "price_forecast":       "float  — noisy 1-step-ahead price forecast",
            "step":                 "int    — current step index [0, 95]",
            "total_steps":          "int    — always 96",
        },
        "action_space": {
            "battery_kw": (
                "float — positive=charge, negative=discharge. "
                "Clamped to [-max_charge_kw, +max_charge_kw] at runtime. Never rejected."
            ),
            "curtail_fraction": (
                "float — [0.0, 1.0]. Fraction of flexible_load_kw deferred to future steps. "
                "Deferred load accumulates in deferred_kwh and must clear by step 96."
            ),
        },
        "reward": {
            "type": "dense",
            "frequency": "every step",
            "components": {
                "r_cost":     "energy import/export cost signal",
                "r_blackout": "-8.0 per step with unmet load",
                "r_solar":    "penalty for wasting solar via grid export",
                "r_defer":    "penalty proportional to deferred load added this step",
                "r_soc":      "boundary shaping for SoC outside [0.10, 0.92]",
            },
            "terminal": "additional penalty if deferred_kwh > 0.1 at episode end",
        },
        "grader": {
            "output_range": [0.0, 1.0],
            "deterministic": True,
            "components": {
                "cost_score":             "weight 0.30 — cost vs uncontrolled baseline",
                "blackout_score":         "weight 0.25 — fraction of steps without blackout",
                "self_consumption_score": "weight 0.20 — solar used locally vs wasted",
                "clearance_score":        "weight 0.15 — deferred load cleared by episode end",
                "health_score":           "weight 0.10 — battery throughput vs safe limit",
                "violation_penalty":      "multiplicative — applied for SoC constraint violations",
            },
        },
        "episode_length": 96,
        "dt_minutes": 15,
    }


# ══════════════════════════════════════════════════════════════════════
# TASKS — GET only, pure read, no session required
# ══════════════════════════════════════════════════════════════════════

@app.get("/tasks", tags=["Tasks"])
def list_tasks():
    """Lists all available tasks with difficulty and key parameters."""
    return {
        "tasks": [
            {
                "id":                   tid,
                "difficulty":           cfg.difficulty,
                "description":          cfg.description,
                "total_steps":          cfg.total_steps,
                "battery_capacity_kwh": cfg.battery_capacity_kwh,
                "solar_capacity_kw":    cfg.solar_capacity_kw,
                "flex_fraction":        cfg.flex_fraction,
                "has_grid_outage":      cfg.grid_outage_start is not None,
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.get("/tasks/{task_id}", tags=["Tasks"])
def get_task(task_id: str):
    """Full config for a single task. No session required."""
    try:
        config = load_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "task_id":    config.task_id,
        "difficulty": config.difficulty,
        "description": config.description,
        "parameters": {
            "solar_capacity_kw":    config.solar_capacity_kw,
            "battery_capacity_kwh": config.battery_capacity_kwh,
            "max_charge_kw":        config.max_charge_kw,
            "load_mean_kw":         config.load_mean_kw,
            "flex_fraction":        config.flex_fraction,
            "price_mean":           config.price_mean,
            "price_sigma":          config.price_sigma,
            "price_spike_prob":     config.price_spike_prob,
            "solar_sigma":          config.solar_sigma,
            "total_steps":          config.total_steps,
            "grid_outage_window": (
                f"steps {config.grid_outage_start}–{config.grid_outage_end}"
                if config.grid_outage_start is not None else "none"
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# SESSION INSPECTION — GET, requires active session_id
# ══════════════════════════════════════════════════════════════════════

@app.get("/state/{session_id}", response_model=MicrogridState, tags=["Episode"])
def get_state(session_id: str):
    """
    Returns current environment state for an active session.
    Read-only — does NOT advance the episode.
    """
    env = env_store.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call GET /reset or POST /reset first."
        )
    try:
        if env.done:
            return env._build_terminal_state()
        return env._build_state()
    except Exception as e:
        log.error(f"GET /state/{session_id} error: {e}")
        raise HTTPException(status_code=500, detail=f"State build failed: {str(e)}")


@app.get("/session/{session_id}/info", tags=["Episode"])
def session_info(session_id: str):
    """Lightweight session metadata — progress, SoC, deferred load, done status."""
    env = env_store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    return {
        "session_id":      session_id,
        "task_id":         env.config.task_id,
        "difficulty":      env.config.difficulty,
        "current_step":    env.current_step,
        "total_steps":     env.config.total_steps,
        "steps_remaining": env.config.total_steps - env.current_step,
        "done":            env.done,
        "soc":             round(env.soc, 4),
        "deferred_kwh":    round(env.deferred_kwh, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# GET COMPATIBILITY ENDPOINTS
# Exist for automated validators that cannot send POST bodies.
# Mirror POST endpoints exactly — same logic, same response models.
# FIX 2 applied: _add_session() used in both GET and POST /reset.
# ══════════════════════════════════════════════════════════════════════

@app.get("/reset", response_model=ResetResponse, tags=["GET Compatibility"])
def reset_get(
    task_id: Optional[str] = None,   # FIX 4: defaults to first available task
    seed: int = 42,
):
    """
    GET version of /reset for automated validator compatibility.
    Works with zero parameters — defaults to first available task, seed=42.
    FIX 4: task_id=None falls back to first available task, never crashes.
    FIX 2: uses _add_session() for bounded memory-safe session store.
    """
    # FIX 4: safe default when no task_id provided
    resolved_task_id = task_id if task_id else _default_task_id()

    try:
        config = load_task(resolved_task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        env = MicrogridEnv(config)
        state = env.reset(seed=seed)
    except Exception as e:
        log.error(f"GET /reset env init error: {e}")
        raise HTTPException(status_code=500, detail=f"Environment init failed: {str(e)}")

    session_id = f"{resolved_task_id}_{seed}_{uuid.uuid4().hex[:6]}"
    _add_session(session_id, env)   # FIX 2: bounded store

    return ResetResponse(
        session_id=session_id,
        state=state,
        task_info=config.summary(),
    )


@app.get("/step", response_model=StepResponse, tags=["GET Compatibility"])
def step_get(
    session_id: str,
    battery_kw: float = 0.0,           # FIX 6: safe no-op default
    curtail_fraction: float = 0.0,     # FIX 6: safe no-op default
):
    """
    GET version of /step for automated validator compatibility.
    Defaults to no-op action (battery_kw=0, curtail_fraction=0) — never crashes on retry.
    FIX 6: all defaults are physically safe values within valid action range.
    """
    env = env_store.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call GET /reset or POST /reset first."
        )
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode already complete. Call /reset to start a new episode."
        )

    try:
        action = MicrogridAction(
            battery_kw=battery_kw,
            curtail_fraction=curtail_fraction,
        )
        result = env.step(action)
    except Exception as e:
        log.error(f"GET /step error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

    return StepResponse(
        state=result.state,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/grader", response_model=GradeResult, tags=["GET Compatibility"])
def grader_get(session_id: str):
    """
    GET version of /grader for automated validator compatibility.
    Usage: GET /grader?session_id=...
    Episode must be complete (all 96 steps run).
    """
    env = env_store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not env.done:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode not complete. "
                f"{env.config.total_steps - env.current_step} steps remaining."
            )
        )

    try:
        trajectory = env.get_trajectory()
        return grade(trajectory, env.config)
    except Exception as e:
        log.error(f"GET /grader error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.get("/baseline", response_model=BaselineResponse, tags=["GET Compatibility"])
def baseline_get(
    task_id: Optional[str] = None,         # FIX 5: safe default
    solar_kw: float = 10.0,                # FIX 5: all params have safe defaults
    soc: float = 0.5,
    base_load_kw: float = 20.0,
    flexible_load_kw: float = 5.0,
    spot_price: float = 0.12,
    grid_available: bool = True,
    step: int = 0,
    deferred_kwh: float = 0.0,
    solar_forecast_kw: float = 10.0,
    price_forecast: float = 0.12,
):
    """
    GET version of /baseline for automated validator compatibility.
    FIX 5: ALL parameters have safe defaults — callable with zero query params.
    FIX 3: MicrogridState construction wrapped in try/except — returns 400 not 500.
    """
    resolved_task_id = task_id if task_id else _default_task_id()

    try:
        config = load_task(resolved_task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # FIX 3: wrap state construction — validation errors return 400, not 500
    try:
        state = MicrogridState(
            solar_kw=solar_kw,
            solar_forecast_kw=solar_forecast_kw,
            soc=max(0.0, min(1.0, soc)),       # clamp silently — validator safety
            battery_capacity_kwh=config.battery_capacity_kwh,
            base_load_kw=base_load_kw,
            flexible_load_kw=flexible_load_kw,
            deferred_kwh=max(0.0, deferred_kwh),
            grid_available=grid_available,
            spot_price=max(0.0, spot_price),
            price_forecast=max(0.0, price_forecast),
            step=max(0, min(step, config.total_steps - 1)),
            total_steps=config.total_steps,
        )
    except Exception as e:
        log.error(f"GET /baseline state construction failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state parameters: {str(e)}"
        )

    try:
        agent = ThresholdHeuristicBaseline()
        action, reasoning = agent.act_with_reason(state, config)
    except Exception as e:
        log.error(f"GET /baseline agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Baseline agent failed: {str(e)}")

    return BaselineResponse(action=action, reasoning=reasoning)


# ══════════════════════════════════════════════════════════════════════
# HIGH-VALUE BONUS ENDPOINT
# FIX 7: GET /baseline_run
# Runs a full episode with the heuristic baseline and returns grader score.
# Deterministic: same task_id + seed → same score every time.
# Does not store session in env_store (ephemeral — no memory leak).
# ══════════════════════════════════════════════════════════════════════

@app.get("/baseline_run", tags=["Evaluation"])
def baseline_run(
    task_id: Optional[str] = None,
    seed: int = 42,
):
    """
    Runs a complete episode using the heuristic baseline agent and returns the graded score.

    Deterministic: same task_id + seed always produces the same score.
    Ephemeral: does NOT persist a session — no memory impact.
    Useful for: validating grader, benchmarking baseline, confirming determinism.

    Example: GET /baseline_run?task_id=sunny_day&seed=42
    """
    resolved_task_id = task_id if task_id else _default_task_id()

    try:
        config = load_task(resolved_task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        env = MicrogridEnv(config)
        state = env.reset(seed=seed)
        agent = ThresholdHeuristicBaseline()

        step_count = 0
        while not env.done:
            action = agent.act(state, config)
            result = env.step(action)
            state = result.state
            step_count += 1
            # Safety guard — should never be needed, but prevents infinite loop
            if step_count > config.total_steps + 5:
                log.warning(f"baseline_run exceeded step limit for task={resolved_task_id}")
                break

        trajectory = env.get_trajectory()
        grade_result = grade(trajectory, config)

    except Exception as e:
        log.error(f"GET /baseline_run failed for task={resolved_task_id}, seed={seed}: {e}")
        raise HTTPException(status_code=500, detail=f"Baseline run failed: {str(e)}")

    return {
        "task_id":    resolved_task_id,
        "seed":       seed,
        "difficulty": config.difficulty,
        "steps_run":  step_count,
        "score":      grade_result.score,
        "breakdown":  grade_result.breakdown,
        "note":       "Deterministic heuristic baseline. Same seed → same score.",
    }


# ══════════════════════════════════════════════════════════════════════
# POST ENDPOINTS — Primary agent interface
# ══════════════════════════════════════════════════════════════════════

@app.post("/reset", response_model=ResetResponse, tags=["Episode"])
def reset(req: ResetRequest):
    """
    Creates a new episode. Returns session_id for all subsequent calls.
    Same task_id + seed always produces an identical episode (fully deterministic).
    FIX 2: uses _add_session() for bounded memory-safe session store.
    """
    try:
        config = load_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        env = MicrogridEnv(config)
        state = env.reset(seed=req.seed)
    except Exception as e:
        log.error(f"POST /reset env init error: {e}")
        raise HTTPException(status_code=500, detail=f"Environment init failed: {str(e)}")

    session_id = f"{req.task_id}_{req.seed}_{uuid.uuid4().hex[:6]}"
    _add_session(session_id, env)   # FIX 2: bounded store

    return ResetResponse(
        session_id=session_id,
        state=state,
        task_info=config.summary(),
    )


@app.post("/step", response_model=StepResponse, tags=["Episode"])
def step(req: StepRequest):
    """
    Advances the environment one 15-minute timestep.
    Actions are silently clamped to physical limits — never rejected with 422.
    Returns next state, step reward, done flag, and diagnostic info dict.
    """
    env = env_store.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /reset or GET /reset first."
        )
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode already complete. Call /reset to start a new episode."
        )

    try:
        result = env.step(req.action)
    except Exception as e:
        log.error(f"POST /step error for session {req.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

    return StepResponse(
        state=result.state,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.post("/grader", response_model=GradeResult, tags=["Evaluation"])
def grader(req: GraderRequest):
    """
    Scores a completed episode. Score ∈ [0.0, 1.0] with full component breakdown.
    Deterministic — same trajectory always returns identical score.
    Episode must be complete (done=True) before calling.
    """
    env = env_store.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not env.done:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode not complete. "
                f"{env.config.total_steps - env.current_step} steps remaining. "
                f"Run all {env.config.total_steps} steps before calling /grader."
            )
        )

    try:
        trajectory = env.get_trajectory()
        return grade(trajectory, env.config)
    except Exception as e:
        log.error(f"POST /grader error for session {req.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.post("/baseline", response_model=BaselineResponse, tags=["Evaluation"])
def baseline(req: BaselineRequest):
    """
    Returns heuristic baseline agent action for any given state.
    Stateless — no session required. Safe to call at any time.
    Includes human-readable reasoning showing which rule fired.
    """
    try:
        config = load_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        agent = ThresholdHeuristicBaseline()
        action, reasoning = agent.act_with_reason(req.state, config)
    except Exception as e:
        log.error(f"POST /baseline agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Baseline agent failed: {str(e)}")

    return BaselineResponse(action=action, reasoning=reasoning)