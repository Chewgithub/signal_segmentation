"""
Script to generate synthetic multi-variable timeseries CSV data with known changepoints.
Each variable has distinct regime shifts at different points in time.
"""

import numpy as np
import pandas as pd

# Fix random seed for reproducibility
rng = np.random.default_rng(42)

# ── Time axis ──────────────────────────────────────────────────────────────────
n_samples = 500
timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")

# ── Helper: piecewise signal ───────────────────────────────────────────────────
def piecewise(segments, noise_std, rng):
    """
    Build a 1-D signal from a list of (length, mean) segments,
    then add Gaussian noise.
    """
    signal = np.concatenate([np.full(length, mean) for length, mean in segments])
    return signal + rng.normal(0, noise_std, size=len(signal))


# ── Variable 1 – Temperature (°C) ─────────────────────────────────────────────
# Changepoints at t = 120, 280, 400
temperature = piecewise(
    [(120, 20.0), (160, 25.5), (120, 18.0), (100, 30.0)],
    noise_std=1.5,
    rng=rng,
)

# ── Variable 2 – Pressure (hPa) ───────────────────────────────────────────────
# Changepoints at t = 100, 250, 370
pressure = piecewise(
    [(100, 1013.0), (150, 1005.0), (120, 1020.0), (130, 1008.0)],
    noise_std=3.0,
    rng=rng,
)

# ── Variable 3 – Vibration (mm/s) ─────────────────────────────────────────────
# Changepoints at t = 150, 300
# Also includes a sinusoidal component to mimic machinery oscillation
t = np.arange(n_samples)
vibration = (
    piecewise([(150, 0.5), (150, 2.0), (200, 0.8)], noise_std=0.3, rng=rng)
    + 0.2 * np.sin(2 * np.pi * t / 20)   # periodic oscillation
)

# ── Assemble DataFrame ─────────────────────────────────────────────────────────
df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "temperature_C": np.round(temperature, 3),
        "pressure_hPa": np.round(pressure, 3),
        "vibration_mm_s": np.round(vibration, 4),
    }
)

output_path = "timeseries_data.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to '{output_path}'")
print(df.head())
