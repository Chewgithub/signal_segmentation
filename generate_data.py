"""
Generate synthetic multi-sensor timeseries CSV with known changepoints.
Seven variables covering different physical quantities and noise levels.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n_samples  = 600
timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")
t          = np.arange(n_samples)


def piecewise(segments: list[tuple[int, float]], noise_std: float) -> np.ndarray:
    """Concatenate constant segments then add Gaussian noise."""
    signal = np.concatenate([np.full(length, mean) for length, mean in segments])
    return signal + rng.normal(0, noise_std, size=len(signal))


# ── Variable 1 – Temperature (°C) ─────────────────────────────────────────────
# Changepoints at: 120, 280, 420
temperature = piecewise([(120, 20.0), (160, 26.0), (140, 17.5), (180, 31.0)],
                        noise_std=1.5)

# ── Variable 2 – Pressure (hPa) ───────────────────────────────────────────────
# Changepoints at: 100, 260, 400
pressure = piecewise([(100, 1013.0), (160, 1004.0), (140, 1021.0), (200, 1007.0)],
                     noise_std=3.5)

# ── Variable 3 – Vibration (mm/s) ─────────────────────────────────────────────
# Changepoints at: 150, 320  (+periodic oscillation to simulate machinery)
vibration = (piecewise([(150, 0.4), (170, 2.2), (280, 0.9)], noise_std=0.3)
             + 0.25 * np.sin(2 * np.pi * t / 18))

# ── Variable 4 – Humidity (%) ─────────────────────────────────────────────────
# Changepoints at: 90, 210, 370, 490
humidity = piecewise([(90, 58.0), (120, 76.0), (160, 52.0), (120, 83.0), (110, 64.0)],
                     noise_std=2.5)

# ── Variable 5 – Electrical Current (A) ──────────────────────────────────────
# Changepoints at: 140, 290, 430
current = piecewise([(140, 2.1), (150, 5.8), (140, 1.6), (170, 4.3)],
                    noise_std=0.35)

# ── Variable 6 – Motor RPM ────────────────────────────────────────────────────
# Changepoints at: 110, 240, 360, 470
rpm = piecewise([(110, 1500), (130, 2300), (120, 1750), (110, 2900), (130, 2100)],
                noise_std=90)

# ── Variable 7 – Ambient Sound (dB) ──────────────────────────────────────────
# Changepoints at: 180, 350
sound_db = piecewise([(180, 44.0), (170, 63.0), (250, 51.0)], noise_std=3.8)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "timestamp":      timestamps,
    "temperature_C":  np.round(temperature, 3),
    "pressure_hPa":   np.round(pressure,    3),
    "vibration_mm_s": np.round(vibration,   4),
    "humidity_pct":   np.round(humidity,    3),
    "current_A":      np.round(current,     4),
    "rpm":            np.round(rpm,         1),
    "sound_dB":       np.round(sound_db,    3),
})

output_path = "timeseries_data.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows × {len(df.columns)-1} variables to '{output_path}'")
print(df.head())
