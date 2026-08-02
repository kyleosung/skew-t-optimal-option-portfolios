"""
Constants for five option datasets: https://doi.org/10.1080/14697680902814225.

Parameters are from Hu and Kercheval (2010), Tables 2-4.

For the multivariate Student t the fitted scale matrix \\Sigma
satisfies Cov = \\nu/(\\nu-2) \\times \\Sigma  (for \\nu > 2).
"""

import numpy as np

# Asset order used throughout
ASSET_NAMES = ["Disney", "Exxon", "Pfizer", "Altria", "Intel"]
ASSET_TICKERS = ["DIS", "XOM", "PFE", "MO", "INTC"]

# Trading days per year used for annualization
TRADING_DAYS_PER_YEAR = 252
SQRT_252 = np.sqrt(TRADING_DAYS_PER_YEAR)

# Spot prices as of the reference date
STOCK_PRICES = np.array([28.02, 60.01, 25.24, 65.53, 23.29], dtype=float)

# ---------------------------------------------------------------------
# Shared daily / annual volatilities
# ---------------------------------------------------------------------
# Daily filtered volatilities from Hu and Kercheval
DAILY_VOL = np.array(
    [
        0.0107,  # Disney
        0.0128,  # Exxon
        0.0130,  # Pfizer
        0.0113,  # Altria
        0.0156,  # Intel
    ],
    dtype=float,
)

# Annualized version (sigma_annual = sigma_daily * sqrt(252))
ANNUAL_VOL = DAILY_VOL * SQRT_252

PAN_ANNUAL_VOL = np.array(
    [
        0.1699,  # Disney
        0.2032,  # Exxon
        0.2064,  # Pfizer
        0.1794,  # Altria
        0.2476,  # Intel
    ]
)

# ---------------------------------------------------------------------
# Normal fit (Tables 2-3)
# ---------------------------------------------------------------------

NORMAL_MU_HAT = np.array(
    [
        0.040,  # Disney
        0.073,  # Exxon
        -0.015,  # Pfizer
        0.039,  # Altria
        0.027,  # Intel
    ],
    dtype=float,
)

NORMAL_CORR = np.array(
    [
        [1.000, 0.367, 0.337, 0.189, 0.420],
        [0.367, 1.000, 0.359, 0.197, 0.303],
        [0.337, 0.359, 1.000, 0.215, 0.297],
        [0.189, 0.197, 0.215, 1.000, 0.168],
        [0.420, 0.303, 0.297, 0.168, 1.000],
    ],
    dtype=float,
)

# Daily-scale restored quantities
_D_daily = np.diag(DAILY_VOL)
NORMAL_MU_DAILY = NORMAL_MU_HAT * DAILY_VOL
NORMAL_COV_DAILY = _D_daily @ NORMAL_CORR @ _D_daily

# Annualized counterparts
NORMAL_MU_ANNUAL = NORMAL_MU_DAILY * TRADING_DAYS_PER_YEAR
NORMAL_COV_ANNUAL = NORMAL_COV_DAILY * TRADING_DAYS_PER_YEAR

# ---------------------------------------------------------------------
# Student t fit (Table 4)
# ---------------------------------------------------------------------

t_NU = 5.87

t_MU_HAT = np.array(
    [
        0.015,  # Disney
        0.077,  # Exxon
        -0.018,  # Pfizer
        0.069,  # Altria
        0.030,  # Intel
    ],
    dtype=float,
)

t_CORR = np.array(
    [
        [1.000, 0.363, 0.378, 0.265, 0.460],
        [0.363, 1.000, 0.373, 0.271, 0.324],
        [0.378, 0.373, 1.000, 0.259, 0.349],
        [0.265, 0.271, 0.259, 1.000, 0.225],
        [0.460, 0.324, 0.349, 0.225, 1.000],
    ],
    dtype=float,
)

# Daily-scale restored quantities
t_MU_DAILY = t_MU_HAT * DAILY_VOL
t_SCALE_DAILY = _D_daily @ t_CORR @ _D_daily
t_COV_DAILY = (t_NU / (t_NU - 2.0)) * t_SCALE_DAILY

# Annualized counterparts
t_MU_ANNUAL = t_MU_DAILY * TRADING_DAYS_PER_YEAR
t_SCALE_ANNUAL = t_SCALE_DAILY * TRADING_DAYS_PER_YEAR
t_COV_ANNUAL = t_COV_DAILY * TRADING_DAYS_PER_YEAR

# ---------------------------------------------------------------------
# For convenience?
# ---------------------------------------------------------------------

PAPER_FITS = {
    "asset_names": ASSET_NAMES,
    "asset_tickers": ASSET_TICKERS,
    "trading_days": TRADING_DAYS_PER_YEAR,
    "daily_vol": DAILY_VOL,
    "annual_vol": ANNUAL_VOL,
    "normal": {
        "mu_hat": NORMAL_MU_HAT,
        "corr": NORMAL_CORR,
        "mu_daily": NORMAL_MU_DAILY,
        "cov_daily": NORMAL_COV_DAILY,
        "mu_annual": NORMAL_MU_ANNUAL,
        "cov_annual": NORMAL_COV_ANNUAL,
    },
    "t": {
        "nu": t_NU,
        "mu_hat": t_MU_HAT,
        "corr": t_CORR,
        "mu_daily": t_MU_DAILY,
        "scale_daily": t_SCALE_DAILY,
        "cov_daily": t_COV_DAILY,
        "mu_annual": t_MU_ANNUAL,
        "scale_annual": t_SCALE_ANNUAL,
        "cov_annual": t_COV_ANNUAL,
    },
}

# ---------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------
stock_names = ASSET_TICKERS
stock_prices = STOCK_PRICES.tolist()
volatilities = ANNUAL_VOL.tolist()
volatility_daily_T = DAILY_VOL.tolist()

correlation_matrix_NORMAL = NORMAL_CORR
correlation_matrix_T = t_CORR
T_DOF_FIT = t_NU

# Dict-based mu_hat lookups (kept for scripts that index by ticker)
EXPECTED_LOG_RETURNS_NORMAL = dict(zip(stock_names, NORMAL_MU_HAT))
EXPECTED_LOG_RETURNS_T = dict(zip(stock_names, t_MU_HAT))

# Corrected annual quantities (mu_hat \\times daily_vol \\times 252)
ANNUAL_EXPECTED_LOG_RETURNS_T_arr = t_MU_ANNUAL
ANNUAL_EXPECTED_RETURNS_T_arr = np.exp(t_MU_ANNUAL) - 1
EXPECTED_RETURNS_T_arr = ANNUAL_EXPECTED_RETURNS_T_arr
