import numpy as np
from env.dynamics import (
    generate_solar, generate_load, generate_prices,
    generate_grid_availability, update_soc
)
from env.models import (
    MicrogridState, MicrogridAction, StepRecord, StepResult
)
from tasks.base import TaskConfig


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class MicrogridEnv:

    def __init__(self, config: TaskConfig):
        self.config = config
        self.rng = None
        self.solar_seq = None
        self.load_seq = None
        self.price_seq = None
        self.grid_seq = None
        self.soc = 0.5
        self.deferred_kwh = 0.0
        self.current_step = 0
        self.done = False
        self.trajectory: list[StepRecord] = []

    def reset(self, seed: int = 42) -> MicrogridState:
        self.rng = np.random.RandomState(seed)
        self.solar_seq = generate_solar(self.rng, self.config)
        self.load_seq = generate_load(self.rng, self.config)
        self.price_seq = generate_prices(self.rng, self.config)
        self.grid_seq = generate_grid_availability(self.config)
        self.soc = 0.5
        self.deferred_kwh = 0.0
        self.current_step = 0
        self.done = False
        self.trajectory = []
        return self._build_state()

    def _build_state(self) -> MicrogridState:
        t = self.current_step
        T = self.config.total_steps
        solar = float(self.solar_seq[t])
        solar_forecast = float(self.solar_seq[t + 1]) if t + 1 < T else 0.0
        price = float(self.price_seq[t])
        price_forecast = float(self.price_seq[t + 1]) if t + 1 < T else price
        base_load = float(self.load_seq[t])
        flex_load = base_load * self.config.flex_fraction

        return MicrogridState(
            solar_kw=round(solar, 3),
            solar_forecast_kw=round(solar_forecast, 3),
            soc=round(self.soc, 4),
            battery_capacity_kwh=self.config.battery_capacity_kwh,
            base_load_kw=round(base_load, 3),
            flexible_load_kw=round(flex_load, 3),
            deferred_kwh=round(self.deferred_kwh, 4),
            grid_available=bool(self.grid_seq[t]),
            spot_price=round(price, 4),
            price_forecast=round(price_forecast, 4),
            step=t,
            total_steps=T,
        )

    def step(self, action: MicrogridAction) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        t = self.current_step
        cfg = self.config
        dt = 0.25  # 15-min steps

        # ── Clamp action ─────────────────────────────────────────────
        battery_kw = _clamp(action.battery_kw, -cfg.max_charge_kw, cfg.max_charge_kw)
        curtail_frac = _clamp(action.curtail_fraction, 0.0, 1.0)
        
        is_clipped = (battery_kw != action.battery_kw) or (curtail_frac != action.curtail_fraction)

        # ── Current step values ───────────────────────────────────────
        solar_kw = float(self.solar_seq[t])
        base_load = float(self.load_seq[t])
        flex_load = base_load * cfg.flex_fraction
        price = float(self.price_seq[t])
        grid_on = bool(self.grid_seq[t])

        # ── Battery update ────────────────────────────────────────────
        new_soc, actual_battery_kw = update_soc(self.soc, battery_kw, cfg, dt)

        # ── Deferred load ─────────────────────────────────────────────
        deferred_added = flex_load * curtail_frac * dt
        load_served_now = base_load + flex_load * (1.0 - curtail_frac)
        self.deferred_kwh += deferred_added

        # Attempt to clear old deferred load from surplus if available
        # (simple: environment auto-serves deferred when there's surplus)
        net_gen = solar_kw + max(0, -actual_battery_kw)  # solar + discharge
        surplus = net_gen - load_served_now

        if surplus > 0 and self.deferred_kwh > 0:
            cleared = min(surplus * dt, self.deferred_kwh)
            self.deferred_kwh -= cleared
            surplus -= cleared / dt

        # ── Power balance ─────────────────────────────────────────────
        residual = solar_kw + (-actual_battery_kw) - load_served_now

        if grid_on:
            import_kw = max(0.0, -residual)
            export_kw = max(0.0, residual)
            unmet_kw = 0.0
            solar_curtailed_kw = 0.0
        else:
            import_kw = 0.0
            export_kw = 0.0
            unmet_kw = max(0.0, -residual)
            solar_curtailed_kw = max(0.0, residual)

        # ── Reward ────────────────────────────────────────────────────
        reward = self._compute_reward(
            actual_battery_kw, curtail_frac, import_kw, export_kw,
            unmet_kw, price, new_soc, deferred_added, solar_kw
        )

        # ── Update state ──────────────────────────────────────────────
        self.soc = new_soc
        self.current_step += 1
        done = self.current_step >= cfg.total_steps

        if done:
            # Terminal penalty for uncleared deferred load
            if self.deferred_kwh > 0.1:
                reward -= 2.0 * self.deferred_kwh

        self.done = done
        next_state = self._build_state() if not done else self._build_terminal_state()

        # ── Record trajectory ─────────────────────────────────────────
        prev_state_snapshot = MicrogridState(
            solar_kw=round(solar_kw, 3),
            solar_forecast_kw=next_state.solar_forecast_kw,
            soc=round(self.soc, 4),
            battery_capacity_kwh=cfg.battery_capacity_kwh,
            base_load_kw=round(base_load, 3),
            flexible_load_kw=round(flex_load, 3),
            deferred_kwh=round(self.deferred_kwh, 4),
            grid_available=grid_on,
            spot_price=round(price, 4),
            price_forecast=next_state.price_forecast,
            step=t,
            total_steps=cfg.total_steps,
        )

        record = StepRecord(
            step=t,
            state=prev_state_snapshot,
            action=MicrogridAction(battery_kw=actual_battery_kw, curtail_fraction=curtail_frac),
            reward=round(reward, 4),
            import_kw=round(import_kw, 3),
            export_kw=round(export_kw, 3),
            unmet_kw=round(unmet_kw, 3),
            solar_curtailed_kw=round(solar_curtailed_kw, 3),
            soc=round(new_soc, 4),
            deferred_kwh=round(self.deferred_kwh, 4),
            spot_price=round(price, 4),
            battery_kw_actual=round(actual_battery_kw, 3),
        )
        self.trajectory.append(record)

        return StepResult(
            state=next_state,
            reward=round(reward, 4),
            done=done,
            info={
                "import_kw": round(import_kw, 3),
                "export_kw": round(export_kw, 3),
                "unmet_kw": round(unmet_kw, 3),
                "deferred_kwh": round(self.deferred_kwh, 4),
                "soc": round(new_soc, 4),
                "battery_kw_actual": round(actual_battery_kw, 3),
                "clipped": is_clipped,
            }
        )

    def _compute_reward(self, battery_kw, curtail_frac, import_kw, export_kw,
                        unmet_kw, price, soc, deferred_added, solar_kw):
        dt = 0.25

        # Cost signal
        energy_cost = (import_kw - export_kw * 0.85) * price * dt
        r_cost = -energy_cost * 2.0

        # Blackout penalty
        r_blackout = -8.0 if unmet_kw > 0.01 else 0.0

        # Solar waste penalty
        solar_waste = export_kw / max(solar_kw, 0.5)
        r_solar = -0.3 * _clamp(solar_waste, 0.0, 1.0)

        # Deferred load penalty
        r_defer = -0.5 * deferred_added

        # SoC boundary shaping
        if soc < 0.10:
            r_soc = -1.5 * (0.10 - soc) / 0.10
        elif soc > 0.92:
            r_soc = -0.5 * (soc - 0.92) / 0.08
        else:
            r_soc = 0.0

        return r_cost + r_blackout + r_solar + r_defer + r_soc

    def _build_terminal_state(self) -> MicrogridState:
        """Return a zeroed-out state when episode is over."""
        return MicrogridState(
            solar_kw=0.0, solar_forecast_kw=0.0,
            soc=round(self.soc, 4),
            battery_capacity_kwh=self.config.battery_capacity_kwh,
            base_load_kw=0.0, flexible_load_kw=0.0,
            deferred_kwh=round(self.deferred_kwh, 4),
            grid_available=True, spot_price=0.0, price_forecast=0.0,
            step=self.config.total_steps,
            total_steps=self.config.total_steps,
        )

    def get_trajectory(self):
        return self.trajectory