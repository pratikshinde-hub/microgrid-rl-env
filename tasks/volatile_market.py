from tasks.base import TaskConfig


def VOLATILE_MARKET() -> TaskConfig:
    return TaskConfig(
        task_id="volatile_market",
        description="Intermittent solar, price spikes, higher flexible load. Plan under uncertainty.",
        difficulty="medium",
        solar_capacity_kw=20.0,
        battery_capacity_kwh=30.0,
        max_charge_kw=15.0,
        load_mean_kw=22.0,
        flex_fraction=0.40,
        price_mean=0.15,
        price_sigma=0.04,
        price_spike_prob=0.08,
        price_spike_min=0.40,
        price_spike_max=0.70,
        solar_sigma=0.15,
        grid_outage_start=None,
        grid_outage_end=None,
    )
