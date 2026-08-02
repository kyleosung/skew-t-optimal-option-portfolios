"""
Normal (Gaussian) option portfolio for Sharpe ratio optimization.

This module rederives all portfolio quantities (D, Gamma, Sigma, p, R, B, xi, theta)
directly from Black-Scholes greeks following the approach in:

    Pan, Kaize. Portfolio Risk Management. Master's Thesis,
    McMaster University, 2023.

The Sharpe ratio optimal weights are then computed using the formula from:

    "In-sample and out-of-sample Sharpe ratios of multi-factor asset
    pricing models":

        w* = (sigma / theta) Sigma^{-1} mu,  where theta = sqrt(mu' Sigma^{-1} mu)

This file is intentionally self-contained - it does NOT import from
option_portfolio.py - to guarantee zero leakage between the
t-distribution optimization code and the normal-case baseline.
"""

import numpy as np
from scipy.stats import norm


def _bs_call_price(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes European call price (zero dividends).

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    sigma : float
        Annualized volatility.
    r : float
        Annual risk-free rate.

    Returns
    -------
    float
        Call option price.
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_call_greeks(
    S: float, K: float, T: float, sigma: float, r: float
) -> tuple[float, float, float]:
    """Black-Scholes call delta, gamma, theta (zero dividends).

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    sigma : float
        Annualized volatility.
    r : float
        Annual risk-free rate.

    Returns
    -------
    tuple[float, float, float]
        (delta, gamma, theta) where theta is per-year.
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    theta = float(
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    return delta, gamma, theta


def compute_normal_sharpe_weights(
    stock_prices: list[float],
    volatilities: list[float],
    correlation_matrix: np.ndarray,
    annual_returns: np.ndarray,
    annual_risk_free_rate: float = 0.05,
    time_to_maturity: float = 1.0,
    time_period_for_risk: float = 1 / 52,
) -> np.ndarray:
    """Compute Sharpe-optimal weights under normality for ATM call options.

    All intermediate quantities are built from scratch using direct
    Black-Scholes computations, following the matrix construction in
    Pan (2023) sections 3 and 4.  The result is the tangency-portfolio
    direction w ~ Sigma_return^{-1} mu_excess, normalized so that sum(w) = 1.

    Parameters
    ----------
    stock_prices : list[float]
        Spot prices of the N underlying stocks.
    volatilities : list[float]
        Annualized volatilities of the N stocks.
    correlation_matrix : np.ndarray
        N x N correlation matrix (normal-case correlations).
    annual_returns : np.ndarray
        Annualized expected returns for each stock (length N).
    annual_risk_free_rate : float
        Annual risk-free interest rate.
    time_to_maturity : float
        Time to maturity for the options (years).
    time_period_for_risk : float
        Risk-measurement horizon Delta_t (years).

    Returns
    -------
    np.ndarray
        Sharpe-optimal weights of shape (N,), summing to 1.
    """
    N = len(stock_prices)
    S = np.array(stock_prices, dtype=float)
    sig = np.array(volatilities, dtype=float)
    Rho = np.asarray(correlation_matrix, dtype=float)
    r = annual_risk_free_rate
    T_mat = time_to_maturity
    dt = time_period_for_risk

    # - Scale (covariance) matrix Sigma (Pan2023 eq. for COV) -
    # COV[i,j] = dt * S[i] * S[j] * sig[i] * sig[j] * Rho[i,j]
    COV = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            COV[i, j] = dt * S[i] * S[j] * sig[i] * sig[j] * Rho[i, j]

    # - Option values, greeks (all ATM calls, K = S) -
    V = np.zeros(N)
    DELTA = np.zeros(N)
    GAMMA_diag = np.zeros(N)
    THETA_vec = np.zeros(N)

    for i in range(N):
        V[i] = _bs_call_price(S[i], S[i], T_mat, sig[i], r)
        delta_i, gamma_i, theta_i = _bs_call_greeks(S[i], S[i], T_mat, sig[i], r)
        DELTA[i] = delta_i
        GAMMA_diag[i] = gamma_i
        THETA_vec[i] = theta_i

    # - Build per-option Gamma matrices Gamma^[m] (N x N, diagonal) -
    # For M = N ATM calls on distinct stocks, Gamma^[m] has gamma only at [m,m].
    def get_gamma_m(m: int) -> np.ndarray:
        G = np.zeros((N, N))
        G[m, m] = GAMMA_diag[m]
        return G

    # - D matrix (N x M) - delta matrix, Pan2023's M = diag(DELTA) -
    # D[n, m] = d(V_m)/d(S_n). For one option per stock: D = diag(DELTA).
    D = np.diag(DELTA)  # N x N (= N x M since M = N)

    # - p vector: p_m = trace(Gamma^[m] @ Sigma) -
    p = np.zeros((N, 1))
    for m in range(N):
        p[m, 0] = np.trace(get_gamma_m(m) @ COV)

    # - R matrix: R[i,j] = trace(Gamma^[i] @ Sigma @ Gamma^[j] @ Sigma) -
    R_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            R_mat[i, j] = np.trace(get_gamma_m(i) @ COV @ get_gamma_m(j) @ COV)

    # - Drift vector mu = S * returns * Delta_t (absolute price change) -
    mu = (S * np.asarray(annual_returns, dtype=float) * dt).reshape(-1, 1)

    # - B matrix: B[m,:] = mu^T @ Gamma^[m] -
    B_mat = np.zeros((N, N))
    for m in range(N):
        B_mat[m, :] = (mu.T @ get_gamma_m(m)).ravel()

    # - xi vector: xi_m = (1/2) mu^T Gamma^[m] mu -
    xi = np.zeros((N, 1))
    for m in range(N):
        xi[m, 0] = 0.5 * (mu.T @ get_gamma_m(m) @ mu).item()

    # - theta vector (option thetas, not Sharpe theta) -
    thetas = THETA_vec.reshape(-1, 1)

    # - Normal-case zeta_normal (nu -> inf limit) -
    # zeta_normal = Delta_t*theta + D^T*mu + (1/2) p + xi
    zeta_normal = dt * thetas + D.T @ mu + 0.5 * p + xi

    # - Normal-case U_normal (nu -> inf limit) -
    # U_normal = 2*(D^T + B) Sigma (D^T + B)^T + R
    DtB = D.T + B_mat  # M x N
    U_normal = 2.0 * DtB @ COV @ DtB.T + R_mat

    # - Map to the paper's return-space notation -
    # mu_excess_i = zeta_i / v_i - r_f*Delta_t
    # Sigma_return = diag(1/v) * (U_normal / 2) * diag(1/v)
    rf_dt = r * dt
    inv_V = np.diag(1.0 / V)

    mu_excess = (zeta_normal.ravel() / V) - rf_dt
    Sigma_return = inv_V @ (U_normal / 2.0) @ inv_V

    # - Paper formula: w* ~ Sigma_return^{-1} mu_excess -
    Sigma_return_inv = np.linalg.pinv(Sigma_return)
    w_direction = Sigma_return_inv @ mu_excess

    w_sum = np.sum(w_direction)
    if abs(w_sum) < 1e-15:
        raise ValueError(
            "Normal Sharpe ratio weights sum to zero; the optimal "
            "portfolio direction is indeterminate."
        )
    weights = w_direction / w_sum

    return weights
