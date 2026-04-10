from tasks.base import TaskConfig


def SUNNY_DAY() -> TaskConfig:
    return TaskConfig(
        task_id="sunny_day",
        description="Clear day, stable grid, low noise. Learn basic solar arbitrage.",
        difficulty="easy",
        solar_capacity_kw=30.0,
        battery_capacity_kwh=50.0,
        max_charge_kw=25.0,
        load_mean_kw=20.0,
        flex_fraction=0.25,
        price_mean=0.12,
        price_sigma=0.01,
        price_spike_prob=0.0,
        price_spike_min=0.0,
        price_spike_max=0.0,
        solar_sigma=0.05,
        grid_outage_start=None,
        grid_outage_end=None,
    )
