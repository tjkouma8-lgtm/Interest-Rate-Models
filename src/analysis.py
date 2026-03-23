"""
Interest Rate Models — Analysis & Visualisation
================================================
Generates:
- Short-rate paths comparison (Vasicek, Merton, Ho-Lee, Hull-White)
- Yield curves / term structures
- Hull-White calibration on market yield curve
- Interest rate cap pricing
"""

import numpy as np
import matplotlib.pyplot as plt
from interest_rate_models import VasicekModel, MertonRateModel, HoLeeModel, HullWhiteModel

plt.style.use('seaborn-v0_8-whitegrid')
BLUE   = '#1A3C5E'
RED    = '#E74C3C'
GREEN  = '#27AE60'
ORANGE = '#F39C12'
PURPLE = '#8E44AD'

COLORS = {
    'Vasicek':     BLUE,
    'Merton':      RED,
    'Ho-Lee':      GREEN,
    'Hull-White':  ORANGE
}


def plot_short_rate_paths():
    """Compare short-rate paths across all four models."""
    T, n_steps, n_paths_plot = 5.0, 252*5, 6

    vasicek = VasicekModel(kappa=0.3, theta=0.05, sigma=0.02, r0=0.03)
    merton  = MertonRateModel(mu=0.003, sigma=0.015, r0=0.03)
    ho_lee  = HoLeeModel(sigma=0.015, r0=0.03)
    hw      = HullWhiteModel(kappa=0.3, sigma=0.02, r0=0.03)

    models  = [vasicek, merton, ho_lee, hw]
    names   = ['Vasicek', 'Merton', 'Ho-Lee', 'Hull-White']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Short-Rate Models — Simulated Paths',
                 fontsize=14, fontweight='bold', color=BLUE)

    t_grid = np.linspace(0, T, n_steps + 1)

    for ax, model, name in zip(axes.flat, models, names):
        r_paths = model.simulate(T, n_steps, n_paths_plot, seed=42)
        cmap = plt.cm.Blues(np.linspace(0.4, 0.9, n_paths_plot))

        for i in range(n_paths_plot):
            ax.plot(t_grid, r_paths[i] * 100, lw=1.3, alpha=0.8, color=cmap[i])

        # Mean reversion level
        if hasattr(model, 'theta') and isinstance(getattr(model, 'theta'), float):
            ax.axhline(model.theta * 100, color=RED, linestyle='--', lw=1.5,
                       label=f'Long-run mean = {model.theta*100:.1f}%')
        ax.axhline(model.r0 * 100, color='gray', linestyle=':', lw=1, alpha=0.6,
                   label=f'r0 = {model.r0*100:.1f}%')

        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_ylabel('Short Rate (%)', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold', color=COLORS[name])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('../results/short_rate_paths.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/short_rate_paths.png")


def plot_yield_curves():
    """Plot yield curves / term structures for each model."""
    maturities = np.linspace(0.1, 10, 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Term Structure of Interest Rates',
                 fontsize=14, fontweight='bold', color=BLUE)

    # Left: Vasicek with different kappa (mean reversion speed)
    ax = axes[0]
    kappas = [0.1, 0.3, 0.8, 2.0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(kappas)))
    for kappa, col in zip(kappas, colors):
        v = VasicekModel(kappa=kappa, theta=0.05, sigma=0.02, r0=0.02)
        yields = v.yield_curve(maturities) * 100
        ax.plot(maturities, yields, lw=2, color=col, label=f'κ={kappa}')
    ax.axhline(5.0, color=RED, linestyle='--', lw=1.5, alpha=0.6, label='θ=5% (long-run)')
    ax.set_xlabel('Maturity (years)', fontsize=11)
    ax.set_ylabel('Zero Yield (%)', fontsize=11)
    ax.set_title('Vasicek — Effect of Mean Reversion Speed κ', fontsize=11)
    ax.legend(fontsize=9)

    # Right: All models comparison
    ax2 = axes[1]
    vasicek = VasicekModel(kappa=0.3, theta=0.05, sigma=0.02, r0=0.03)
    merton  = MertonRateModel(mu=0.003, sigma=0.015, r0=0.03)
    hw      = HullWhiteModel(kappa=0.3, sigma=0.02, r0=0.03)

    ax2.plot(maturities, vasicek.yield_curve(maturities)*100,
             color=BLUE, lw=2.5, label='Vasicek')
    ax2.plot(maturities, merton.yield_curve(maturities)*100,
             color=RED, lw=2, linestyle='--', label='Merton')
    ax2.plot(maturities, [hw.bond_price_analytical(0, T) for T in maturities],
             color=ORANGE, lw=2, linestyle=':', label='Hull-White (P(0,T))')

    ax2.set_xlabel('Maturity (years)', fontsize=11)
    ax2.set_ylabel('Zero Yield (%) / Bond Price', fontsize=11)
    ax2.set_title('Model Comparison — Yield Curves', fontsize=11)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('../results/yield_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/yield_curves.png")


def plot_hull_white_calibration():
    """Calibrate Hull-White to a synthetic market yield curve."""
    # Synthetic market yield curve (upward sloping)
    mkt_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    mkt_yields = np.array([0.030, 0.032, 0.035, 0.040, 0.043, 0.048, 0.051, 0.054])

    hw = HullWhiteModel(kappa=0.3, sigma=0.015, r0=mkt_yields[0],
                         market_yields=mkt_yields, maturities=mkt_maturities)

    maturities_fine = np.linspace(0.1, 10, 80)
    hw_yields = hw.yield_curve(maturities_fine)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hull-White — Calibration to Market Yield Curve',
                 fontsize=13, fontweight='bold', color=BLUE)

    # Left: yield curve fit
    ax = axes[0]
    ax.scatter(mkt_maturities, mkt_yields*100, s=80, color=RED, zorder=5,
               label='Market yields')
    ax.plot(maturities_fine, hw_yields*100, color=BLUE, lw=2.5,
            label='Hull-White fit')
    ax.set_xlabel('Maturity (years)', fontsize=11)
    ax.set_ylabel('Zero Yield (%)', fontsize=11)
    ax.set_title('Perfect Fit to Initial Yield Curve', fontsize=11)
    ax.legend(fontsize=10)

    # Right: short-rate paths calibrated to market
    ax2 = axes[1]
    T, n_steps, n_paths_plot = 5.0, 252*5, 8
    r_paths = hw.simulate(T, n_steps, n_paths_plot, seed=42)
    t_grid = np.linspace(0, T, n_steps + 1)
    cmap = plt.cm.Blues(np.linspace(0.4, 0.9, n_paths_plot))

    for i in range(n_paths_plot):
        ax2.plot(t_grid, r_paths[i]*100, lw=1.3, alpha=0.8, color=cmap[i])
    ax2.axhline(mkt_yields[0]*100, color='gray', linestyle=':', lw=1.2,
                label=f'r0={mkt_yields[0]*100:.1f}%')
    ax2.set_xlabel('Time (years)', fontsize=11)
    ax2.set_ylabel('Short Rate (%)', fontsize=11)
    ax2.set_title('Hull-White Paths (Calibrated to Market)', fontsize=11)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('../results/hull_white_calibration.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/hull_white_calibration.png")


def plot_mean_reversion():
    """Visualise Vasicek mean reversion property."""
    n_paths_plot = 200
    T, n_steps = 10.0, 252*10

    # High r0 (above mean) and low r0 (below mean)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Vasicek — Mean Reversion Property',
                 fontsize=13, fontweight='bold', color=BLUE)

    for ax, r0, title in zip(axes,
                              [0.10, 0.01],
                              ['r0=10% > θ=5% (reverting down)',
                               'r0=1% < θ=5% (reverting up)']):
        v = VasicekModel(kappa=0.5, theta=0.05, sigma=0.02, r0=r0)
        r_paths = v.simulate(T, n_steps, n_paths_plot, seed=42)
        t_grid = np.linspace(0, T, n_steps + 1)

        # Distribution of paths
        mean_path = np.mean(r_paths, axis=0) * 100
        std_path  = np.std(r_paths, axis=0) * 100

        ax.plot(t_grid, mean_path, color=BLUE, lw=2.5, label='Mean path')
        ax.fill_between(t_grid,
                         mean_path - 2*std_path,
                         mean_path + 2*std_path,
                         alpha=0.2, color=BLUE, label='±2σ band')
        ax.axhline(5.0, color=RED, linestyle='--', lw=2, label='Long-run mean θ=5%')
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Short Rate (%)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('../results/mean_reversion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/mean_reversion.png")


if __name__ == "__main__":
    import os
    os.makedirs('../results', exist_ok=True)

    print("Generating plots...\n")
    plot_short_rate_paths()
    plot_yield_curves()
    plot_hull_white_calibration()
    plot_mean_reversion()
    print("\nAll plots saved in results/")
