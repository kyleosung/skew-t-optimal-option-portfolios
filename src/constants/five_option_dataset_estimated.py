"""
Estimated parameters for five-stock dataset (t and skew-t distributions).

All parameters are loaded at import time from JSON files produced by the
estimation scripts, so re-running those scripts automatically propagates
new values without any manual editing of this file.

NOTE: The GARCH filtering step rescales log returns to percentage scale
(x100) before fitting to avoid the ``arch`` DataScaleWarning and improve
optimizer convergence. The standardized residuals are invariant to this
linear rescaling, but the different optimization landscape may produce
slightly different GARCH parameters compared to unscaled fitting.

Parameters
----------
``EST_t_*``
    Student t parameters estimated via R (``sn::mst.mple`` with
    ``alpha`` fixed to zero), fit to GARCH-filtered daily log returns
    (750 observations, 2002-07-02 to 2005-08-04).  Loaded from
    ``experiment_t_vs_skew_t_estimation/estimation_results/t_fit.json``.

``AC_SKEW_t_*``
    Azzalini-Capitanio (AC) skew-t parameters estimated via R
    (``sn::mst.mple``), fit to the same GARCH-filtered residuals.
    Loaded from
    ``experiment_t_vs_skew_t_estimation/estimation_results/ac_fit.json``.

These are separate from the Hu and Kercheval (2010) constants in
``five_option_dataset.py``.  Small differences from HK are expected
because the data source is Yahoo Finance (with auto_adjust) rather
than CRSP.

To regenerate parameters, run:
    PYTHONPATH=$(pwd) python experiment_t_vs_skew_t_estimation/estimate_skew_t.py
    PYTHONPATH=$(pwd) python experiment_t_vs_skew_t_estimation/estimate_standard_t.py
"""

import json
from pathlib import Path

import numpy as np

from src.constants.five_option_dataset import DAILY_VOL, TRADING_DAYS_PER_YEAR

_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / (
    "experiment_t_vs_skew_t_estimation/estimation_results"
)

# Daily diagonal scale matrix (shared by all sections below)
_D_daily = np.diag(DAILY_VOL)

# =====================================================================
# Student t estimated parameters (R/SN ``sn::mst.mple``, alpha=0)
# Source: experiment_t_vs_skew_t_estimation/estimation_results/t_fit.json
# =====================================================================

with open(_RESULTS_DIR / "t_fit.json", "r") as _f:
    _t_fit = json.load(_f)

EST_t_NU = float(_t_fit["nu"])

# Location vector mu_hat on the GARCH standardized-residual scale
EST_t_MU_HAT = np.array(_t_fit["mu"], dtype=float)

# Correlation matrix
EST_t_CORR = np.array(_t_fit["Corr"], dtype=float)

# Log-likelihood at the MLE solution
EST_t_LOGLIK = float(_t_fit["logL"])

# Daily-scale restored quantities
EST_t_MU_DAILY = EST_t_MU_HAT * DAILY_VOL
EST_t_SCALE_DAILY = _D_daily @ EST_t_CORR @ _D_daily
EST_t_COV_DAILY = (EST_t_NU / (EST_t_NU - 2.0)) * EST_t_SCALE_DAILY

# Annualized counterparts
EST_t_MU_ANNUAL = EST_t_MU_DAILY * TRADING_DAYS_PER_YEAR
EST_t_SCALE_ANNUAL = EST_t_SCALE_DAILY * TRADING_DAYS_PER_YEAR
EST_t_COV_ANNUAL = EST_t_COV_DAILY * TRADING_DAYS_PER_YEAR

# =====================================================================
# AC skew-t parameters from R/SN package (``sn::mst.mple``)
# Source: experiment_t_vs_skew_t_estimation/estimation_results/skew_t_ac_fit.json
#
# The Azzalini-Capitanio (AC) skew-t is parameterised in direct (DP)
# form as  Y ~ ST(xi, Omega, alpha, nu):
#   xi    - location vector (mu in project notation)
#   Omega - scale matrix  (Sigma in project notation / skew_t_ac_fit.json)
#   alpha - slant vector  (omega in project notation / skew_t_ac_fit.json)
#   nu    - degrees of freedom
#
# All DP parameters are on the GARCH standardized-residual scale
# (unit individual GARCH variance).  The CP block gives the centred
# parameterisation (actual mean, covariance, standardized skewness).
#
# Parameters are loaded directly from skew_t_ac_fit.json at import time.
# =====================================================================

with open(_RESULTS_DIR / "skew_t_ac_fit.json", "r") as _f:
    _ac_fit = json.load(_f)

_dp = _ac_fit["dp"]
_cp = _ac_fit["cp"]

AC_SKEW_t_NU = float(_dp["nu"])

# Location vector xi on the GARCH standardized-residual scale
AC_SKEW_t_MU_HAT = np.array(_dp["mu"], dtype=float)

# Scale matrix Omega on the GARCH standardized-residual scale
AC_SKEW_t_SIGMA_DP = np.array(_dp["Sigma"], dtype=float)

# Slant vector alpha (called omega here to avoid confusion with tail-risk alpha)
AC_SKEW_t_OMEGA = np.array(_dp["omega"], dtype=float)

# Correlation matrix extracted from the DP scale matrix Omega
# Corr[i,j] = Omega[i,j] / sqrt(Omega[i,i] * Omega[j,j])
_AC_sigma_std = np.sqrt(np.diag(AC_SKEW_t_SIGMA_DP))
AC_SKEW_t_CORR = AC_SKEW_t_SIGMA_DP / np.outer(_AC_sigma_std, _AC_sigma_std)
np.fill_diagonal(AC_SKEW_t_CORR, 1.0)

# Daily-scale restored quantities (GARCH vol * correlation structure)
# XI_* constants use the AC skew-t location (xi / direct parameterisation),
# NOT the distribution mean.  Use MEAN_* constants for actual E[Y].
AC_SKEW_t_XI_DAILY = AC_SKEW_t_MU_HAT * DAILY_VOL
AC_SKEW_t_SCALE_DAILY = _D_daily @ AC_SKEW_t_CORR @ _D_daily
AC_SKEW_t_COV_DAILY = (AC_SKEW_t_NU / (AC_SKEW_t_NU - 2.0)) * AC_SKEW_t_SCALE_DAILY

# Annualized location-based counterparts
AC_SKEW_t_XI_ANNUAL = AC_SKEW_t_XI_DAILY * TRADING_DAYS_PER_YEAR
AC_SKEW_t_SCALE_ANNUAL = AC_SKEW_t_SCALE_DAILY * TRADING_DAYS_PER_YEAR
AC_SKEW_t_COV_ANNUAL = AC_SKEW_t_COV_DAILY * TRADING_DAYS_PER_YEAR

# ------------------------------------------------------------------
# Centred parameterisation (CP) - actual moments of the distribution
# ------------------------------------------------------------------

# CP mean vector (actual E[Y] of the AC skew-t)
AC_SKEW_t_CP_MEAN = np.array(_cp["mean"], dtype=float)

# CP variance-covariance matrix (actual Cov[Y] of the AC skew-t)
AC_SKEW_t_CP_VAR_COV = np.array(_cp["var.cov"], dtype=float)

# CP standardized skewness (gamma1) and multivariate kurtosis (gamma2M)
AC_SKEW_t_CP_GAMMA1 = np.array(_cp["gamma1"], dtype=float)
AC_SKEW_t_CP_GAMMA2M = float(_cp["gamma2M"])

# Log-likelihood at the R/SN MLE solution
AC_SKEW_t_LOGLIK = float(_ac_fit["logL"])

# ------------------------------------------------------------------
# Mean-based daily / annualized expected returns
# These are derived from the CP mean (actual E[Y]), not the location xi,
# and should be used wherever expected returns are required.
# ------------------------------------------------------------------
AC_SKEW_t_MEAN_DAILY = AC_SKEW_t_CP_MEAN * DAILY_VOL
AC_SKEW_t_MEAN_ANNUAL = AC_SKEW_t_MEAN_DAILY * TRADING_DAYS_PER_YEAR

# ------------------------------------------------------------------
# Convenience arrays aligned with ASSET_TICKERS order for experiments
# ------------------------------------------------------------------
AC_SKEW_t_XI_ANNUAL_arr = AC_SKEW_t_XI_ANNUAL


if __name__ == "__main__":
    from src.constants.five_option_dataset import ASSET_TICKERS

    np.set_printoptions(precision=6, suppress=True)

    sep = "=" * 60

    print(sep)
    print("Student t  (EST_t_*)  - loaded from t_fit.json")
    print(sep)
    print(f"  nu          : {EST_t_NU}")
    print(f"  log-lik     : {EST_t_LOGLIK}")
    print("  mu_hat      :")
    for ticker, v in zip(ASSET_TICKERS, EST_t_MU_HAT):
        print(f"    {ticker}: {v:.6f}")
    print("  correlation matrix:")
    for row in EST_t_CORR:
        print("   ", row)

    print()
    print(sep)
    print("AC skew-t  (AC_SKEW_t_*)  - loaded from skew_t_ac_fit.json")
    print(sep)
    print(f"  nu          : {AC_SKEW_t_NU}")
    print(f"  log-lik     : {AC_SKEW_t_LOGLIK}")
    print("  xi (mu_hat) :")
    for ticker, v in zip(ASSET_TICKERS, AC_SKEW_t_MU_HAT):
        print(f"    {ticker}: {v:.6f}")
    print("  omega (slant) :")
    for ticker, v in zip(ASSET_TICKERS, AC_SKEW_t_OMEGA):
        print(f"    {ticker}: {v:.6f}")
    print("  Sigma_DP (scale matrix) :")
    for row in AC_SKEW_t_SIGMA_DP:
        print("   ", row)
    print("  correlation matrix (from Sigma_DP) :")
    for row in AC_SKEW_t_CORR:
        print("   ", row)
    print("  CP mean :")
    for ticker, v in zip(ASSET_TICKERS, AC_SKEW_t_CP_MEAN):
        print(f"    {ticker}: {v:.6f}")
    print("  CP gamma1 (skewness) :")
    for ticker, v in zip(ASSET_TICKERS, AC_SKEW_t_CP_GAMMA1):
        print(f"    {ticker}: {v:.6f}")
    print(f"  CP gamma2M (kurtosis) : {AC_SKEW_t_CP_GAMMA2M:.6f}")
