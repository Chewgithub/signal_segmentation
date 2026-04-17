# Signal Segmentation – PELT vs Kernel CPD

Detect regime changes in multi-variable timeseries by comparing two exact changepoint detection algorithms from the **ruptures** library:

| Algorithm | Class | Sensitivity | Complexity |
|-----------|-------|-------------|------------|
| **PELT** – Pruned Exact Linear Time | `rpt.Pelt` | Mean & variance shifts | O(n) avg |
| **KernelCPD** – Kernel Change Point Detection | `rpt.KernelCPD` | Full distribution shifts (via RKHS) | O(n²) worst |

![PELT vs KernelCPD output](pelt_vs_kernel.png)

---

## Dataset

1 000 hourly samples across 7 sensor variables with **realistic, messy characteristics**:

- **Sigmoid transitions** — regime shifts ramp over 8–45 samples (not hard steps)
- **AR(1) autocorrelated noise** — consecutive samples are correlated (φ = 0.5–0.85)
- **Heteroscedastic noise** — each regime has its own noise level
- **Within-segment drift** — slow sinusoidal wandering inside regimes
- **Random outlier spikes** — 1–2 % of samples are sensor glitch bursts
- **Inter-variable coupling** — humidity is anti-correlated with temperature; sound tracks RPM
- **Diurnal cycle** — 24 h + 12 h oscillation embedded in temperature

| Variable | Unit | Planted changepoints (approx. sample) | Notable characteristic |
|---|---|---|---|
| `temperature_C` | °C | 220, 500, 730 | 24 h diurnal cycle |
| `pressure_hPa` | hPa | 200, 460, 760 | Very wide (45-sample) sigmoid transitions |
| `vibration_mm_s` | mm/s | 180, 380, 530, 730 | Two harmonic oscillations |
| `humidity_pct` | % | 220, 500, 730 | Anti-correlated with temperature |
| `current_A` | A | 250, 450, 630, 810 | Medium transitions, low noise |
| `rpm` | RPM | 200, 420, 590, 780 | High noise relative to signal amplitude |
| `sound_dB` | dB | 250, 500, 750 | Weakly coupled to RPM |

---

## Results

Tested on 1 000-sample signals with tuned penalties per variable:

| Variable | Target CPs | PELT found | KernelCPD found | Avg offset PELT | Avg offset KernelCPD |
|---|---|---|---|---|---|
| temperature_C | 3 | **3** ✓ | **3** ✓ | ~23 samples | ~22 samples |
| pressure_hPa | 3 | **3** ✓ | **3** ✓ | ~74 samples | ~74 samples |
| vibration_mm_s | 4 | **4** ✓ | **4** ✓ | ~5 samples | ~5 samples |
| humidity_pct | 3 | **3** ✓ | **3** ✓ | ~22 samples | ~23 samples |
| current_A | 4 | **4** ✓ | **4** ✓ | ~10 samples | ~9 samples |
| rpm | 4 | **3** ✗ | **4** ✓ | ~15 samples | ~19 samples |
| sound_dB | 3 | **3** ✓ | **3** ✓ | ~22 samples | ~22 samples |

> Avg offset = mean distance from each detected changepoint to the nearest planted one.

---

---

## Hyperparameter Tuning

[`hyperparameter_tuning.ipynb`](hyperparameter_tuning.ipynb) performs an automated **grid search** over penalties to find the F1-maximising value for each variable and method, removing the need for manual tuning.

### Approach

- Penalty candidates: 60 log-spaced values from 1 to 200
- Scoring: Precision / Recall / F1 against planted changepoints with a **±10 sample tolerance window**
- Both PELT and KernelCPD are evaluated independently across all 7 variables

### Visualisations produced

| Plot | File | Description |
|------|------|-------------|
| Elbow plots | `elbow_plots.png` | CP count vs penalty — locate the elbow where n_cps stabilises |
| F1 curves | `f1_curves.png` | F1 score vs penalty for PELT and KernelCPD per variable |
| Precision–Recall curves | `precision_recall_curves.png` | How precision and recall trade off as penalty increases |
| F1 heatmap | `f1_heatmap.png` | Variable × penalty F1 matrix — white dot marks the optimum |
| Optimal detection overlay | `optimal_detection.png` | Raw signals with true, PELT, and KernelCPD CPs at optimal penalties |
| Radar chart | `f1_radar.png` | Optimal F1 per variable for both methods |
| Sensitivity analysis | `penalty_sensitivity.png` | F1 vs penalty-ratio — wide flat peak = robust, sharp peak = sensitive |
| Gain bar chart | `f1_gain_bars.png` | F1 improvement from auto-tuning vs hand-tuned penalties |

### Key findings

- **Auto-tuning matches or beats hand-tuning** for most variables with no manual effort
- **Well-behaved signals have wide, flat F1 peaks** — a 2–3× deviation from the optimal penalty barely hurts performance
- **Noisy or slow-transition signals are sensitive** — `rpm` and `pressure_hPa` have sharp peaks requiring precise tuning
- **Penalty scales are not transferable between methods** — always tune PELT and KernelCPD independently
- **Tolerance choice matters** — match `tol` to the expected precision of your task; slow ramps warrant a wider window

---

## Conclusions

### 1. Both methods agree when the signal is well-behaved
For 6 out of 7 variables, PELT and KernelCPD detected the same number of changepoints at nearly identical positions (within 1–3 samples). When both agree, that agreement is strong evidence of a real regime shift.

### 2. KernelCPD has an edge on noisy, high-amplitude signals
On `rpm` — the noisiest signal relative to its shift magnitudes — PELT missed one changepoint while KernelCPD recovered all four. This is because KernelCPD operates in a reproducing kernel Hilbert space (RKHS) and is sensitive to changes in the **full distribution**, not just the mean or variance. Subtle shifts that barely move the mean can still produce a detectable change in the kernel embedding.

### 3. Gradual transitions reduce accuracy for both methods equally
`pressure_hPa` had the largest offset error (~74 samples) for **both** algorithms. With a 45-sample sigmoid ramp, the "true" changepoint location is inherently ambiguous — neither method can pinpoint it exactly, and neither has a clear advantage. Offset errors shrink to ~5 samples for abrupt changes (`vibration_mm_s`).

### 4. Periodic and coupled signals require more careful penalty tuning
Temperature's 24 h diurnal cycle and vibration's harmonics required significantly higher KernelCPD penalties (20× the initial guess) to avoid over-segmentation. Embedded frequencies inflate the kernel cost and make the algorithm over-sensitive. PELT was less affected by this. If your signal has known periodic components, consider detrending first.

### 5. Penalty scales are not transferable between methods
The penalty that works for PELT rarely works for KernelCPD on the same signal, because they optimise fundamentally different cost functions. Always tune each method independently.

### 6. Runtime: KernelCPD is competitive at n = 1 000
Despite its O(n²) worst-case complexity, KernelCPD ran in **7–8 ms** per variable versus **34–200 ms** for PELT on this dataset. At n = 1 000, the C-accelerated kernel matrix computation is well-optimised. PELT's advantage grows significantly at larger n (tens of thousands of samples).

### When to use which

| Situation | Recommended |
|---|---|
| Fast prototype, unknown signal type | **PELT** — fast, robust, easy to tune |
| Signal has subtle or distributional shifts | **KernelCPD** — more sensitive |
| Periodic / oscillatory components present | **PELT** — less sensitive to harmonics |
| Very long signals (n > 10 000) | **PELT** — O(n) vs O(n²) matters |
| Need high detection recall on noisy data | **KernelCPD** — found the RPM shift PELT missed |
| Both methods agree | High-confidence changepoint |
| Methods disagree | Inspect manually — the change may be subtle or spurious |

---

## Project structure

```
signal_segmentation/
├── generate_data.py                           # Realistic multi-sensor CSV generator
├── timeseries_data.csv                        # 1 000-row × 7-variable dataset (auto-generated)
├── change detection (PELT vs Kernal Change).ipynb   # Algorithm comparison notebook
├── hyperparameter_tuning.ipynb                # Penalty grid-search & optimisation notebook
├── pelt_vs_kernel.png                         # Output figure (auto-generated)
├── elbow_plots.png                            # CP count vs penalty (auto-generated)
├── f1_curves.png                              # F1 score vs penalty (auto-generated)
├── precision_recall_curves.png                # Precision & recall curves (auto-generated)
├── f1_heatmap.png                             # Variable × penalty F1 heatmap (auto-generated)
├── optimal_detection.png                      # Optimal penalty detection overlay (auto-generated)
├── f1_radar.png                               # Radar chart of optimal F1 per variable (auto-generated)
├── penalty_sensitivity.png                    # F1 sensitivity to penalty deviation (auto-generated)
├── f1_gain_bars.png                           # Auto-tuned vs hand-tuned F1 gain (auto-generated)
├── pyproject.toml                             # Project dependencies
└── uv.lock                                    # Locked dependency versions
```

---

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

| Package | Purpose |
|---|---|
| `pandas` | Load and manipulate CSV data |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `ruptures` | PELT and KernelCPD changepoint detection |
| `scipy` | Signal processing utilities |

---

## Getting started

### 1. Clone the repo

```bash
git clone <repo-url>
cd signal_segmentation
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Generate the dataset

```bash
uv run python generate_data.py
```

### 4. Run the notebooks

Open either notebook in VS Code (or JupyterLab) and run all cells.

```bash
# Algorithm comparison
uv run jupyter lab "change detection (PELT vs Kernal Change).ipynb"

# Hyperparameter tuning
uv run jupyter lab hyperparameter_tuning.ipynb
```

---

## How the algorithms work

### PELT
Minimises a penalised segmentation cost:

```
minimise  Σ cost(segment) + pen × (number of changepoints)
```

Uses an **inequality pruning** test to discard candidate changepoints that can never be optimal, giving O(n) average time.

### KernelCPD
Maps each sample into a **reproducing kernel Hilbert space (RKHS)** via an RBF kernel, then minimises the **maximum mean discrepancy (MMD)** between adjacent segments. Operates on the full kernel embedding rather than raw values.

### Reading the plot

| Visual element | Meaning |
|---|---|
| Light grey line | Raw signal |
| Coloured horizontal bars | PELT segment means |
| Red dotted vertical | PELT changepoint |
| Blue dashed vertical | KernelCPD changepoint |
| Teal **▼** marker | Both methods agree (within ±2 samples) |

---

## Adapting to your own data

1. Replace `timeseries_data.csv` with your own file — keep a `timestamp` column and numeric sensor columns.
2. Adjust the `config` dict in cell 2: `{column: (pelt_penalty, kernel_penalty)}`.
3. If your signal has periodic components, consider removing the trend before running KernelCPD.
4. Start with a high penalty and lower it until results stabilise — jumping to a low penalty first leads to over-segmentation that is hard to walk back.
