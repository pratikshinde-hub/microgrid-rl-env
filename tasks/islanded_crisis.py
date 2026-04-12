from tasks.base import TaskConfig

ISLANDED_CRISIS = TaskConfig(
    task_id="islanded_crisis",
    description="Grid outage steps 20–52. Survive on solar+battery. High flex load must clear.",
    difficulty="hard",
    solar_capacity_kw=15.0,
    battery_capacity_kwh=20.0,
    max_charge_kw=10.0,
    load_mean_kw=18.0,
    flex_fraction=0.60,
    price_mean=0.20,
    price_sigma=0.06,
    price_spike_prob=0.12,
    price_spike_min=0.50,
    price_spike_max=1.20,
    solar_sigma=0.25,
    grid_outage_start=20,
    grid_outage_end=52,
)
