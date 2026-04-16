"""
Generate realistic, messy multi-sensor timeseries data.

Design choices that make it feel real:
  - Sigmoid (smooth) transitions between regimes instead of hard steps
  - Heteroscedastic noise: each regime has its own noise level
  - AR(1) autocorrelation: consecutive samples are correlated
  - Slow within-segment drift (linear + low-freq sine)
  - Random outlier spikes (sensor glitches, transient events)
  - Inter-variable correlations (e.g. humidity tracks inverse of temperature)
  - Diurnal (24 h) cycle embedded in temperature
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

N          = 1000                                          # total samples
timestamps = pd.date_range("2024-03-01", periods=N, freq="1h")
t          = np.arange(N, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level building blocks
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x, center, width=20):
    """Smooth 0→1 transition; `width` controls how many samples the ramp takes."""
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def blend_regimes(t, regimes, transition_width=20):
    """
    Build a smooth baseline from a list of (length, mean) regimes.
    Adjacent regimes are blended with a sigmoid rather than a hard step.
    """
    n        = len(t)
    baseline = np.zeros(n)
    # cumulative breakpoints
    breaks   = np.cumsum([r[0] for r in regimes])
    means    = [r[1] for r in regimes]

    # start from the first mean, then add each subsequent jump smoothly
    baseline[:] = means[0]
    for i, bp in enumerate(breaks[:-1]):
        jump      = means[i + 1] - means[i]
        baseline += jump * sigmoid(t, bp, transition_width)
    return baseline


def ar1_noise(n, std, phi=0.6):
    """
    AR(1) correlated noise: x[t] = phi*x[t-1] + eps.
    `phi` in (0,1) controls autocorrelation strength.
    Scaled so the long-run std ≈ `std`.
    """
    eps  = rng.normal(0, std * np.sqrt(1 - phi**2), n)
    x    = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def hetero_noise(t, regimes, phi=0.5):
    """
    Stitch together AR(1) noise segments with per-regime standard deviations.
    Each regime entry is (length, mean, noise_std).
    """
    noise  = np.zeros(len(t))
    cursor = 0
    for length, _, std in regimes:
        end          = min(cursor + length, len(t))
        seg_len      = end - cursor
        noise[cursor:end] = ar1_noise(seg_len, std, phi)
        cursor       = end
    return noise


def add_drift(t, regimes, drift_scale=0.4):
    """
    Add a slow sinusoidal drift within each segment to simulate gradual
    environmental changes (temperature creep, sensor drift, etc.).
    """
    drift  = np.zeros(len(t))
    cursor = 0
    for length, mean, _ in regimes:
        end     = min(cursor + length, len(t))
        seg_t   = np.linspace(0, np.pi, end - cursor)
        # amplitude proportional to mean magnitude
        amp     = drift_scale * abs(mean) * 0.05
        drift[cursor:end] = amp * np.sin(seg_t + rng.uniform(0, np.pi))
        cursor  = end
    return drift


def add_spikes(n, prob=0.008, scale=4.0, signal_std=1.0):
    """Random outlier spikes — short bursts mimicking sensor glitches."""
    spikes = np.zeros(n)
    mask   = rng.random(n) < prob
    spikes[mask] = rng.choice([-1, 1], size=mask.sum()) * scale * signal_std
    return spikes


def build_signal(t, regimes, transition_width=20, drift_scale=0.4,
                 ar_phi=0.55, spike_prob=0.008):
    """Full pipeline: smooth baseline + drift + heteroscedastic AR(1) noise + spikes."""
    baseline = blend_regimes(t, regimes, transition_width)
    drift    = add_drift(t, regimes, drift_scale)
    noise    = hetero_noise(t, regimes, ar_phi)
    avg_std  = np.mean([r[2] for r in regimes])
    spikes   = add_spikes(len(t), spike_prob, scale=4.0, signal_std=avg_std)
    return baseline + drift + noise + spikes


# ─────────────────────────────────────────────────────────────────────────────
# Variable definitions  (length, mean, noise_std)
# ─────────────────────────────────────────────────────────────────────────────

# Temperature — slow diurnal cycle + 3 weather regime shifts
# Transitions are wide (30 samples ≈ 30 h) to mimic gradual weather changes
temp_regimes = [(220, 12.0, 1.8), (280, 19.5, 2.5), (230, 8.5, 1.2), (270, 24.0, 3.0)]
temperature  = build_signal(t, temp_regimes, transition_width=30, ar_phi=0.75)
# Add diurnal (24 h) cycle — amplitude varies per regime
diurnal      = 3.5 * np.sin(2 * np.pi * t / 24 + 0.8) + 0.6 * np.sin(2 * np.pi * t / 12)
temperature += diurnal

# Pressure — very slow-moving, wide transitions (frontal passages take hours)
pres_regimes = [(200, 1015.0, 2.5), (260, 1006.5, 4.0), (300, 1022.0, 2.0), (240, 1009.0, 3.5)]
pressure     = build_signal(t, pres_regimes, transition_width=45, ar_phi=0.85)

# Vibration — abrupt machine-state changes, higher spikes rate
vib_regimes  = [(180, 0.35, 0.25), (200, 1.80, 0.55), (150, 0.60, 0.20),
                (200, 2.50, 0.70), (270, 0.45, 0.18)]
vibration    = build_signal(t, vib_regimes, transition_width=8, ar_phi=0.45,
                            spike_prob=0.02)
# Add machinery oscillation (two harmonics)
vibration   += 0.15 * np.sin(2 * np.pi * t / 17) + 0.07 * np.sin(2 * np.pi * t / 7)

# Humidity — loosely anti-correlated with temperature, slow transitions
hum_regimes  = [(220, 72.0, 3.5), (280, 55.0, 4.5), (230, 81.0, 2.8), (270, 48.0, 5.0)]
humidity     = build_signal(t, hum_regimes, transition_width=35, ar_phi=0.80)
# Add weak coupling with temperature (inverse)
humidity    -= 0.3 * (temperature - temperature.mean())
humidity     = np.clip(humidity, 20, 100)    # physical bounds

# Electrical current — relatively abrupt load changes, occasional dropout spikes
curr_regimes = [(250, 2.2, 0.30), (200, 5.9, 0.60), (180, 1.4, 0.22),
                (180, 4.7, 0.50), (190, 3.1, 0.35)]
current      = build_signal(t, curr_regimes, transition_width=12, ar_phi=0.50,
                            spike_prob=0.015)
current      = np.clip(current, 0, None)     # current can't be negative

# RPM — gradual spin-up/down, noisy at high speeds
rpm_regimes  = [(200, 1450, 95), (220, 2350, 180), (170, 1750, 110),
                (190, 3050, 250), (220, 2150, 160)]
rpm          = build_signal(t, rpm_regimes, transition_width=25, ar_phi=0.65)
rpm          = np.clip(rpm, 500, None)

# Sound — correlated with RPM (louder when faster), medium transitions
snd_regimes  = [(250, 42.0, 4.0), (250, 61.0, 5.5), (250, 50.0, 4.5), (250, 68.0, 6.0)]
sound_db     = build_signal(t, snd_regimes, transition_width=20, ar_phi=0.60)
# Add weak coupling with RPM (higher RPM → slightly louder)
sound_db    += 0.004 * (rpm - rpm.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Assemble & save
# ─────────────────────────────────────────────────────────────────────────────

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

df.to_csv("timeseries_data.csv", index=False)
print(f"Saved {len(df)} rows × {len(df.columns)-1} variables to 'timeseries_data.csv'")
print(df.describe().round(2).to_string())
