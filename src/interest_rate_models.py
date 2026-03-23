"""
Interest Rate Models — Core Implementation
===========================================
Implements:
- Vasicek model
- Merton model (Ho-Lee without drift)
- Ho-Lee model
- Hull-White (extended Vasicek) model
- Zero-coupon bond pricing
- Yield curve / term structure simulation
- Interest rate cap pricing
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. VASICEK MODEL (1977)
# ─────────────────────────────────────────────

class VasicekModel:
    """
    Vasicek short-rate model:
        dr_t = kappa*(theta - r_t)*dt + sigma*dW_t

    Parameters
    ----------
    kappa : float - Mean reversion speed
    theta : float - Long-term mean rate
    sigma : float - Volatility
    r0    : float - Initial short rate
    """

    def __init__(self, kappa, theta, sigma, r0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0    = r0

    def simulate(self, T, n_steps, n_paths, seed=42):
        """Simulate short-rate paths via Euler-Maruyama."""
        np.random.seed(seed)
        dt = T / n_steps
        r = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        for i in range(n_steps):
            dW = np.random.standard_normal(n_paths) * np.sqrt(dt)
            r[:, i+1] = (r[:, i]
                         + self.kappa * (self.theta - r[:, i]) * dt
                         + self.sigma * dW)
        return r

    def bond_price_analytical(self, t, T):
        """
        Analytical zero-coupon bond price P(t,T).
        P(t,T) = A(t,T) * exp(-B(t,T) * r_t)
        """
        tau = T - t
        B = (1 - np.exp(-self.kappa * tau)) / self.kappa
        A = np.exp(
            (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - tau)
            - self.sigma**2 * B**2 / (4 * self.kappa)
        )
        return A * np.exp(-B * self.r0)

    def yield_curve(self, maturities):
        """Compute zero yields R(0,T) = -log(P(0,T))/T."""
        yields = []
        for T in maturities:
            P = self.bond_price_analytical(0, T)
            yields.append(-np.log(P) / T)
        return np.array(yields)

    def bond_price_mc(self, T, n_steps=252, n_paths=50000, seed=42):
        """Monte Carlo bond price via E[exp(-int r_t dt)]."""
        r = self.simulate(T, n_steps, n_paths, seed)
        dt = T / n_steps
        integral = np.sum(r[:, :-1], axis=1) * dt
        return np.mean(np.exp(-integral))


# ─────────────────────────────────────────────
# 2. MERTON MODEL (1973)
# ─────────────────────────────────────────────

class MertonRateModel:
    """
    Merton short-rate model (simplest):
        dr_t = mu*dt + sigma*dW_t

    Parameters
    ----------
    mu    : float - Constant drift
    sigma : float - Volatility
    r0    : float - Initial short rate
    """

    def __init__(self, mu, sigma, r0):
        self.mu    = mu
        self.sigma = sigma
        self.r0    = r0

    def simulate(self, T, n_steps, n_paths, seed=42):
        """Simulate short-rate paths."""
        np.random.seed(seed)
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        dW = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
        increments = self.mu * dt + self.sigma * dW
        r = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0
        r[:, 1:] = self.r0 + self.mu * t_grid[1:] + self.sigma * np.cumsum(dW, axis=1)
        return r

    def bond_price_analytical(self, T):
        """
        Analytical ZCB price under Merton:
        P(0,T) = exp(-r0*T - 0.5*mu*T^2 + (1/6)*sigma^2*T^3)
        """
        return np.exp(
            -self.r0 * T
            - 0.5 * self.mu * T**2
            + (1/6) * self.sigma**2 * T**3
        )

    def yield_curve(self, maturities):
        """Zero yields from analytical bond prices."""
        return np.array([-np.log(self.bond_price_analytical(T)) / T
                         for T in maturities])


# ─────────────────────────────────────────────
# 3. HO-LEE MODEL (1986)
# ─────────────────────────────────────────────

class HoLeeModel:
    """
    Ho-Lee short-rate model:
        dr_t = theta(t)*dt + sigma*dW_t

    theta(t) is calibrated to fit the initial yield curve exactly.

    Parameters
    ----------
    sigma         : float  - Volatility
    r0            : float  - Initial short rate
    market_yields : array  - Market zero yields for calibration
    maturities    : array  - Corresponding maturities
    """

    def __init__(self, sigma, r0, market_yields=None, maturities=None):
        self.sigma = sigma
        self.r0    = r0
        self.market_yields = market_yields
        self.maturities    = maturities

        if market_yields is not None and maturities is not None:
            self._calibrate_theta()

    def _calibrate_theta(self):
        """
        Calibrate theta(t) to fit initial forward rates:
        theta(t) = df(0,t)/dt + sigma^2 * t
        where f(0,t) = -d(log P(0,t))/dt
        """
        T = np.array(self.maturities)
        R = np.array(self.market_yields)

        # Market ZCB prices and forward rates
        P = np.exp(-R * T)
        log_P = -R * T

        # Numerical derivative of log P
        dlogP_dT = np.gradient(log_P, T)
        f_0t = -dlogP_dT  # instantaneous forward rates

        d2logP_dT2 = np.gradient(f_0t, T)
        self.theta_t = d2logP_dT2 + self.sigma**2 * T
        self.theta_interp = lambda t: np.interp(t, T, self.theta_t)

    def simulate(self, T, n_steps, n_paths, seed=42):
        """Simulate Ho-Lee paths with time-varying drift."""
        np.random.seed(seed)
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)

        r = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        for i in range(n_steps):
            t_i = t_grid[i]
            theta_i = self.theta_interp(t_i) if self.market_yields is not None else 0.02
            dW = np.random.standard_normal(n_paths) * np.sqrt(dt)
            r[:, i+1] = r[:, i] + theta_i * dt + self.sigma * dW
        return r

    def bond_price_analytical(self, T):
        """
        Analytical ZCB price:
        P(0,T) = exp(A(T) - B(T)*r0)
        B(T) = T
        A(T) = int_0^T theta(s)*(T-s)ds - 0.5*sigma^2*T^3/3
        """
        if self.market_yields is None:
            return np.exp(-self.r0 * T - 0.5 * self.sigma**2 * T**3 / 3)

        # Numerical integration of theta(s)*(T-s)
        s = np.linspace(0, T, 500)
        integrand = self.theta_interp(s) * (T - s)
        A = np.trapz(integrand, s) - 0.5 * self.sigma**2 * T**3 / 3
        return np.exp(A - T * self.r0)


# ─────────────────────────────────────────────
# 4. HULL-WHITE MODEL (1990)
# ─────────────────────────────────────────────

class HullWhiteModel:
    """
    Hull-White (extended Vasicek) model:
        dr_t = (theta(t) - kappa*r_t)*dt + sigma*dW_t

    theta(t) calibrated to fit initial yield curve exactly.

    Parameters
    ----------
    kappa         : float - Mean reversion speed
    sigma         : float - Volatility
    r0            : float - Initial short rate
    market_yields : array - Market zero yields
    maturities    : array - Corresponding maturities
    """

    def __init__(self, kappa, sigma, r0, market_yields=None, maturities=None):
        self.kappa = kappa
        self.sigma = sigma
        self.r0    = r0
        self.market_yields = market_yields
        self.maturities    = maturities

        if market_yields is not None and maturities is not None:
            self._build_forward_curve()

    def _build_forward_curve(self):
        """Build instantaneous forward curve f(0,T) from market yields."""
        T = np.array(self.maturities)
        R = np.array(self.market_yields)
        log_P = -R * T
        self.f_0t = -np.gradient(log_P, T)
        self.df_0t = np.gradient(self.f_0t, T)
        self.T_grid = T

    def _forward_rate(self, t):
        """Interpolate instantaneous forward rate f(0,t)."""
        if self.market_yields is None:
            return self.r0
        return np.interp(t, self.T_grid, self.f_0t)

    def _df_forward(self, t):
        """Derivative of forward rate."""
        if self.market_yields is None:
            return 0.0
        return np.interp(t, self.T_grid, self.df_0t)

    def theta(self, t):
        """
        theta(t) = df(0,t)/dt + kappa*f(0,t) + sigma^2/(2*kappa)*(1-exp(-2*kappa*t))
        """
        return (self._df_forward(t)
                + self.kappa * self._forward_rate(t)
                + self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * t)))

    def simulate(self, T, n_steps, n_paths, seed=42):
        """Simulate Hull-White paths."""
        np.random.seed(seed)
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)

        r = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        for i in range(n_steps):
            t_i = t_grid[i]
            theta_i = self.theta(t_i)
            dW = np.random.standard_normal(n_paths) * np.sqrt(dt)
            r[:, i+1] = (r[:, i]
                          + (theta_i - self.kappa * r[:, i]) * dt
                          + self.sigma * dW)
        return r

    def B(self, t, T):
        """B(t,T) = (1 - exp(-kappa*(T-t))) / kappa"""
        return (1 - np.exp(-self.kappa * (T - t))) / self.kappa

    def bond_price_analytical(self, t, T):
        """
        Analytical ZCB price:
        P(t,T) = P(0,T)/P(0,t) * exp(B(t,T)*f(0,t) - sigma^2/(4*kappa)*B(t,T)^2*(1-exp(-2*kappa*t)))
        """
        if self.market_yields is None:
            tau = T - t
            B_val = self.B(t, T)
            A_val = np.exp(
                (self.r0 - self.sigma**2 / (2 * self.kappa**2)) * (B_val - tau)
                - self.sigma**2 * B_val**2 / (4 * self.kappa)
            )
            return A_val * np.exp(-B_val * self.r0)

        P_0T = np.exp(-np.interp(T, self.T_grid, self.market_yields) * T)
        P_0t = np.exp(-np.interp(t, self.T_grid, self.market_yields) * t) if t > 0 else 1.0
        B_val = self.B(t, T)
        f_0t  = self._forward_rate(t)

        return (P_0T / P_0t * np.exp(
            B_val * f_0t
            - self.sigma**2 / (4 * self.kappa) * B_val**2 * (1 - np.exp(-2 * self.kappa * t))
        ))

    def yield_curve(self, maturities):
        """Model zero yields."""
        yields = []
        for T in maturities:
            P = self.bond_price_analytical(0, T)
            yields.append(-np.log(max(P, 1e-10)) / T)
        return np.array(yields)

    def cap_price(self, K, T_expiry, T_pay, notional=1.0):
        """
        Price an interest rate caplet under Hull-White.
        Uses Black's formula with Hull-White vol adjustment.
        """
        B_val = self.B(T_expiry, T_pay)
        P_0_Tpay = self.bond_price_analytical(0, T_pay)
        P_0_Texp = self.bond_price_analytical(0, T_expiry)

        sigma_p = self.sigma * B_val * np.sqrt((1 - np.exp(-2 * self.kappa * T_expiry)) / (2 * self.kappa))

        h = np.log(P_0_Tpay / (P_0_Texp * (1 + K * (T_pay - T_expiry)))) / sigma_p + sigma_p / 2

        caplet = notional * (
            P_0_Tpay * norm.cdf(h) -
            P_0_Texp * (1 + K * (T_pay - T_expiry)) * norm.cdf(h - sigma_p)
        )
        return max(caplet, 0.0)


# ─────────────────────────────────────────────
# 5. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])

    print("=" * 65)
    print("  INTEREST RATE MODELS — Zero-Coupon Bond Prices")
    print("=" * 65)
    print(f"  {'Maturity':>10} | {'Vasicek':>10} | {'Merton':>10} | {'Hull-White':>12}")
    print("-" * 65)

    vasicek = VasicekModel(kappa=0.3, theta=0.05, sigma=0.02, r0=0.03)
    merton  = MertonRateModel(mu=0.005, sigma=0.02, r0=0.03)
    hw      = HullWhiteModel(kappa=0.3, sigma=0.02, r0=0.03)

    for T in maturities:
        P_v  = vasicek.bond_price_analytical(0, T)
        P_m  = merton.bond_price_analytical(T)
        P_hw = hw.bond_price_analytical(0, T)
        print(f"  {T:>10.2f} | {P_v:>10.5f} | {P_m:>10.5f} | {P_hw:>12.5f}")

    print("=" * 65)

    print("\n  Vasicek Yield Curve:")
    yields_v = vasicek.yield_curve(maturities)
    for T, y in zip(maturities, yields_v):
        print(f"  T={T:.2f}  R={y*100:.3f}%")
