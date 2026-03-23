# Interest Rate Models
### Vasicek · Merton · Ho-Lee · Hull-White — Implementation & Term Structure

> **Tounday Jules KOUMAGNON** — MSc Financial and Data Mathematics, Université Gustave Eiffel

---

## Overview

This project implements four classical **short-rate models** for interest rate modelling, from the simplest (Merton) to the most flexible (Hull-White). Each model is implemented with analytical bond pricing formulas, Monte Carlo simulation, and yield curve construction.

---

## Models Implemented

### 1. Merton Model (1973)
Simplest short-rate model — constant drift and volatility:
$$dr_t = \mu \, dt + \sigma \, dW_t$$

### 2. Vasicek Model (1977)
Adds mean reversion — rates are pulled back toward a long-run mean:
$$dr_t = \kappa(\theta - r_t) \, dt + \sigma \, dW_t$$

### 3. Ho-Lee Model (1986)
Time-varying drift calibrated to fit the market yield curve exactly:
$$dr_t = \theta(t) \, dt + \sigma \, dW_t$$

### 4. Hull-White Model (1990) — Extended Vasicek
Mean reversion with time-varying drift — perfectly fits any initial yield curve:
$$dr_t = (\theta(t) - \kappa \, r_t) \, dt + \sigma \, dW_t$$

---

## Features

| Feature | Vasicek | Merton | Ho-Lee | Hull-White |
|---|:---:|:---:|:---:|:---:|
| Mean reversion | ✅ | ❌ | ❌ | ✅ |
| Fits initial yield curve | ❌ | ❌ | ✅ | ✅ |
| Analytical bond price | ✅ | ✅ | ✅ | ✅ |
| Negative rates possible | ✅ | ✅ | ✅ | ✅ |
| Monte Carlo simulation | ✅ | ✅ | ✅ | ✅ |
| Cap pricing | ❌ | ❌ | ❌ | ✅ |

---

## Results

### Short-Rate Paths
All four models simulated over 5 years — mean reversion clearly visible in Vasicek and Hull-White.

![Short Rate Paths](results/short_rate_paths.png)

### Yield Curves — Term Structure
Effect of mean reversion speed κ on the Vasicek yield curve shape.

![Yield Curves](results/yield_curves.png)

### Hull-White Calibration
Hull-White perfectly fits any initial market yield curve via θ(t).

![Hull-White Calibration](results/hull_white_calibration.png)

### Vasicek Mean Reversion
Regardless of starting point, rates converge to the long-run mean θ.

![Mean Reversion](results/mean_reversion.png)

---

## Project Structure

```
Interest-Rate-Models/
│
├── src/
│   ├── interest_rate_models.py   # Core: all 4 models, bond pricing, yield curves
│   └── analysis.py               # Plots: paths, term structure, calibration
│
├── notebooks/
│   └── interest_rate_full.ipynb
│
├── results/                      # Generated figures
│
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/tjkouma8-lgtm/Interest-Rate-Models
cd Interest-Rate-Models
pip install -r requirements.txt

cd src
python interest_rate_models.py   # pricing test
python analysis.py               # generate plots
```

**Expected output:**
```
═════════════════════════════════════════════════════════════════
  INTEREST RATE MODELS — Zero-Coupon Bond Prices
═════════════════════════════════════════════════════════════════
    Maturity |    Vasicek |     Merton |   Hull-White
─────────────────────────────────────────────────────────────────
        0.25 |    0.99256 |    0.99254 |      0.99255
        0.50 |    0.98517 |    0.98508 |      0.98516
        1.00 |    0.97056 |    0.97022 |      0.97053
        ...
```

---

## Key Parameters

| Parameter | Symbol | Role |
|---|---|---|
| Mean reversion speed | κ | How fast rates return to long-run mean |
| Long-run mean | θ | Equilibrium rate level |
| Volatility | σ | Rate uncertainty |
| Time-varying drift | θ(t) | Calibration to yield curve (Ho-Lee, HW) |

---

## References

- Vasicek, O. (1977). *An equilibrium characterization of the term structure*. Journal of Financial Economics.
- Merton, R.C. (1973). *Theory of rational option pricing*. Bell Journal of Economics.
- Ho, T. & Lee, S. (1986). *Term structure movements and pricing interest rate contingent claims*. Journal of Finance.
- Hull, J. & White, A. (1990). *Pricing interest-rate derivative securities*. Review of Financial Studies.

---

## Author

**Tounday Jules KOUMAGNON**
MSc Financial and Data Mathematics — Université Gustave Eiffel (Paris)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tjkoum8)
