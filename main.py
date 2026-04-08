import uuid
import logging
from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env.microgrid import MicrogridEnv
from env.models import MicrogridAction, GradeResult
from env.grader import grade
from tasks import load_task, TASKS
from baseline.heuristic import ThresholdHeuristicBaseline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("microgrid")

app = FastAPI(
    title="Microgrid RL Environment",
    description="OpenEnv compliant microgrid environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS = 100
env_store: OrderedDict[str, MicrogridEnv] = OrderedDict()


def _add_session(session_id: str, env: MicrogridEnv):
    if len(env_store) >= MAX_SESSIONS:
        oldest = next(iter(env_store))
        env_store.pop(oldest)
    env_store[session_id] = env


def _default_task_id():
    return next(iter(TASKS.keys()))


# ─────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42


class ResetResponse(BaseModel):
    session_id: str
    state: dict
    task_info: dict


class StepRequest(BaseModel):
    session_id: str
    action: MicrogridAction


class StepResponse(BaseModel):
    state: dict
    reward: float
    done: bool
    info: dict


class GraderRequest(BaseModel):
    session_id: str


class BaselineRequest(BaseModel):
    state: dict
    task_id: str


class BaselineResponse(BaseModel):
    action: MicrogridAction
    reasoning: str


# ─────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "tasks": list(TASKS.keys()),
        "active_sessions": len(env_store)
    }


# ─────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):

    try:
        config = load_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    env = MicrogridEnv(config)
    state = env.reset(seed=req.seed)

    session_id = f"{req.task_id}_{uuid.uuid4().hex[:6]}"

    _add_session(session_id, env)

    return ResetResponse(
        session_id=session_id,
        state=state,
        task_info=config.summary(),
    )


@app.get("/reset", response_model=ResetResponse)
def reset_get(task_id: Optional[str] = None, seed: int = 42):

    task_id = task_id or _default_task_id()
    config = load_task(task_id)

    env = MicrogridEnv(config)
    state = env.reset(seed=seed)

    session_id = f"{task_id}_{uuid.uuid4().hex[:6]}"

    _add_session(session_id, env)

    return ResetResponse(
        session_id=session_id,
        state=state,
        task_info=config.summary(),
    )


# ─────────────────────────────────────────────────────────
# Step
# ─────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):

    env = env_store.get(req.session_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if env.done:
        raise HTTPException(status_code=400, detail="Episode finished")

    result = env.step(req.action)

    return StepResponse(
        state=result["state"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"],
    )


@app.get("/step", response_model=StepResponse)
def step_get(
    session_id: str,
    battery_kw: float = 0.0,
    curtail_fraction: float = 0.0
):

    env = env_store.get(session_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    action = MicrogridAction(
        battery_kw=battery_kw,
        curtail_fraction=curtail_fraction
    )

    result = env.step(action)

    return StepResponse(
        state=result["state"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"],
    )


# ─────────────────────────────────────────────────────────
# Grader
# ─────────────────────────────────────────────────────────

@app.post("/grader", response_model=GradeResult)
def grader(req: GraderRequest):

    env = env_store.get(req.session_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not env.done:
        raise HTTPException(status_code=400, detail="Episode not complete")

    trajectory = env.get_trajectory()

    return grade(trajectory, env.config)


@app.get("/grader", response_model=GradeResult)
def grader_get(session_id: str):

    env = env_store.get(session_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    trajectory = env.get_trajectory()

    return grade(trajectory, env.config)


# ─────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest):

    config = load_task(req.task_id)

    agent = ThresholdHeuristicBaseline()

    action, reasoning = agent.act_with_reason(req.state, config)

    return BaselineResponse(
        action=action,
        reasoning=reasoning
    )


@app.get("/baseline", response_model=BaselineResponse)
def baseline_get(task_id: Optional[str] = None):

    task_id = task_id or _default_task_id()

    config = load_task(task_id)

    agent = ThresholdHeuristicBaseline()

    dummy_state = {
        "solar_kw": 10,
        "solar_forecast_kw": 10,
        "soc": 0.5,
        "battery_capacity_kwh": config.battery_capacity_kwh,
        "base_load_kw": 20,
        "flexible_load_kw": 5,
        "deferred_kwh": 0,
        "grid_available": True,
        "spot_price": 0.1,
        "price_forecast": 0.1,
        "step": 0,
        "total_steps": config.total_steps
    }

    action, reasoning = agent.act_with_reason(dummy_state, config)

    return BaselineResponse(
        action=action,
        reasoning=reasoning
    )
