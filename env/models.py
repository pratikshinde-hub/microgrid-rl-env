from pydantic import BaseModel
from typing import Dict, Any


class MicrogridState(BaseModel):
    solar_kw: float
    solar_forecast_kw: float
    soc: float
    battery_capacity_kwh: float
    base_load_kw: float
    flexible_load_kw: float
    deferred_kwh: float
    grid_available: bool
    spot_price: float
    price_forecast: float
    step: int
    total_steps: int


class MicrogridAction(BaseModel):
    battery_kw: float       # positive = charge, negative = discharge
    curtail_fraction: float # 0.0 → 1.0


class StepRecord(BaseModel):
    step: int
    state: MicrogridState
    action: MicrogridAction
    reward: float
    import_kw: float
    export_kw: float
    unmet_kw: float
    solar_curtailed_kw: float
    soc: float
    deferred_kwh: float
    spot_price: float
    battery_kw_actual: float


class StepResult(BaseModel):
    state: MicrogridState
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeResult(BaseModel):
    score: float
    breakdown: Dict[str, Any]
