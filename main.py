import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from env.microgrid import MicrogridEnv
from env.models import MicrogridAction, MicrogridState, GradeResult
from env.grader import grade
from tasks import load_task, TASKS
from baseline.heuristic import ThresholdHeuristicBaseline

app = FastAPI(
    title="Microgrid RL Environment",
    description="OpenEnv-compliant Virtual Power Plant dispatch environment.",
    version="1.0.0"
)

# In-memory session store
env_store: dict[str, MicrogridEnv] = {}


# ── Request / Response Models ────────────────────────────────────────

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


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Microgrid RL Environment",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state/{session_id}",
                      "/tasks", "/grader", "/baseline", "/docs"]
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": cfg.difficulty,
                "description": cfg.description,
                "total_steps": cfg.total_steps,
                "battery_capacity_kwh": cfg.battery_capacity_kwh,
                "solar_capacity_kw": cfg.solar_capacity_kw,
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    try:
        config = load_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    env = MicrogridEnv(config)
    state = env.reset(seed=req.seed)

    session_id = f"{req.task_id}_{req.seed}_{uuid.uuid4().hex[:6]}"
    env_store[session_id] = env

    return ResetResponse(
        session_id=session_id,
        state=state,
        task_info=config.summary()
    )
    
@app.get("/reset", response_model=ResetResponse)
def reset_get(task_id: Optional[str] = None, seed: int = 42):
    """
    GET version of reset.
    - If task_id is not provided → default to a valid task
    """

    # Default task if none provided
    if task_id is None:
        task_id = list(TASKS.keys())[0]   # first available task

    try:
        config = load_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    env = MicrogridEnv(config)
    state = env.reset(seed=seed)

    session_id = f"{task_id}_{seed}_{uuid.uuid4().hex[:6]}"
    env_store[session_id] = env

    return ResetResponse(
        session_id=session_id,
        state=state,
        task_info=config.summary()
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = env_store.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode complete. Call /reset to start a new one.")

    result = env.step(req.action)
    return StepResponse(
        state=result.state,
        reward=result.reward,
        done=result.done,
        info=result.info
    )


@app.get("/state/{session_id}", response_model=MicrogridState)
def get_state(session_id: str):
    env = env_store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state if not env.done else env._build_terminal_state()


@app.post("/grader", response_model=GradeResult)
def grader(req: GraderRequest):
    env = env_store.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not env.done:
        raise HTTPException(status_code=400, detail="Episode not complete. Run all 96 steps first.")

    trajectory = env.get_trajectory()
    return grade(trajectory, env.config)


@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest):
    try:
        config = load_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent = ThresholdHeuristicBaseline()
    action, reasoning = agent.act_with_reason(req.state, config)
    return BaselineResponse(action=action, reasoning=reasoning)