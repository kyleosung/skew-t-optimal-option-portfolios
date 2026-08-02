"""
Estimated parameters for five-stock dataset (v2: multi-GARCH filtered).

Loads parameter sets for ALL three GARCH innovation distributions
(normal, t, skewt) into dictionaries keyed by the distribution name.
This allows downstream code to select the desired parameter set by key:

    from src.constants.five_option_dataset_estimated_v2 import (
        AC_SKEW_T_PARAMS,
        EST_T_PARAMS,
    )

    # Select the normal-filtered parameters (best MLE)
    params = AC_SKEW_T_PARAMS["normal"]
    nu = params["nu"]
    corr = params["corr"]

Each dictionary entry contains the same fields as the flat constants in
``five_option_dataset_estimated.py``, but nested under the GARCH dist key.

NOTE: Log returns are rescaled to percentage scale (x100) before GARCH
fitting to avoid the ``arch`` DataScaleWarning and improve optimizer
convergence. The standardized residuals are theoretically invariant to
this linear rescaling, but in practice the different optimization landscape
can produce slightly different GARCH parameter estimates (and hence slightly
different filtered residuals) compared to unscaled fitting. This is why
parameter values may differ from earlier runs without rescaling.

MLE comparison (Gaussian GARCH wins):
    normal:  skew-t logL = -4885.46
    t:       skew-t logL = -4908.91
    skewt:   skew-t logL = -4908.50

File naming convention in comparison_results/:
    - ``skew_t_ac_fit_normal.json`` (normal GARCH, from compare_garch_innovations.py)
    - ``skew_t_ac_fit_t.json``      (t-GARCH)
    - ``skew_t_ac_fit_skewt.json``  (skewt-GARCH)
    - ``t_fit_normal.json``         (normal GARCH, from compare_garch_innovations.py)
    - ``t_fit_t.json``              (t-GARCH)
    - ``t_fit_skewt.json``          (skewt-GARCH)

NOTE: ``t_fit.json`` and ``skew_t_ac_fit.json`` (no suffix) hold the original
pre-comparison estimates used by ``five_option_dataset_estimated.py`` (v1)
and live in ``estimation_results/``.
Do not overwrite those files from ``compare_garch_innovations.py``.

To regenerate parameters for all distributions, run:
    PYTHONPATH=$(pwd) python estimation/compare_garch_innovations.py
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.constants.five_option_dataset import DAILY_VOL, TRADING_DAYS_PER_YEAR

_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / (
    "estimation/comparison_results"
)

# Daily diagonal scale matrix (shared by all sections below)
_D_daily = np.diag(DAILY_VOL)

# The three supported GARCH innovation distributions
GARCH_DISTRIBUTIONS = ("normal", "t", "skewt")


def _suffix_for_dist(dist: str) -> str:
    """Return the file suffix for a given GARCH innovation distribution."""
    return f"_{dist}"


def _load_skew_t_params(dist: str) -> Dict[str, Any]:
    """
    Load AC skew-t parameters for the given GARCH dist.

    Parameters
    ----------
    dist : str
        One of "normal", "t", "skewt".

    Returns
    -------
    dict
        Parameter dictionary with keys: nu, mu_hat, sigma_dp, omega, corr,
        xi_daily, scale_daily, cov_daily, xi_annual, scale_annual, cov_annual,
        cp_mean, cp_var_cov, cp_gamma1, cp_gamma2m, loglik,
        mean_daily, mean_annual.
        Returns empty dict if the JSON file is not found.
    """
    suffix = _suffix_for_dist(dist)
    path = _RESULTS_DIR / f"skew_t_ac_fit{suffix}.json"

    if not path.exists():
        return {}

    with open(path, "r") as f:
        ac_fit = json.load(f)

    dp = ac_fit["dp"]
    cp = ac_fit["cp"]

    nu = float(dp["nu"])
    mu_hat = np.array(dp["mu"], dtype=float)
    sigma_dp = np.array(dp["Sigma"], dtype=float)
    omega = np.array(dp["omega"], dtype=float)

    # Correlation matrix extracted from the DP scale matrix
    sigma_std = np.sqrt(np.diag(sigma_dp))
    corr = sigma_dp / np.outer(sigma_std, sigma_std)
    np.fill_diagonal(corr, 1.0)

    # Daily-scale restored quantities
    xi_daily = mu_hat * DAILY_VOL
    scale_daily = _D_daily @ corr @ _D_daily
    cov_daily = (nu / (nu - 2.0)) * scale_daily

    # Annualized
    xi_annual = xi_daily * TRADING_DAYS_PER_YEAR
    scale_annual = scale_daily * TRADING_DAYS_PER_YEAR
    cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

    # Centred parameterisation (CP) - actual moments
    cp_mean = np.array(cp["mean"], dtype=float)
    cp_var_cov = np.array(cp["var.cov"], dtype=float)
    cp_gamma1 = np.array(cp["gamma1"], dtype=float)
    cp_gamma2m = float(cp["gamma2M"])
    loglik = float(ac_fit["logL"])

    # Mean-based daily / annualized expected returns
    mean_daily = cp_mean * DAILY_VOL
    mean_annual = mean_daily * TRADING_DAYS_PER_YEAR

    return {
        "nu": nu,
        "mu_hat": mu_hat,
        "sigma_dp": sigma_dp,
        "omega": omega,
        "corr": corr,
        "xi_daily": xi_daily,
        "scale_daily": scale_daily,
        "cov_daily": cov_daily,
        "xi_annual": xi_annual,
        "scale_annual": scale_annual,
        "cov_annual": cov_annual,
        "cp_mean": cp_mean,
        "cp_var_cov": cp_var_cov,
        "cp_gamma1": cp_gamma1,
        "cp_gamma2m": cp_gamma2m,
        "loglik": loglik,
        "mean_daily": mean_daily,
        "mean_annual": mean_annual,
    }


def _load_t_params(dist: str) -> Dict[str, Any]:
    """
    Load symmetric Student-t parameters for the given GARCH dist.

    Parameters
    ----------
    dist : str
        One of "normal", "t", "skewt".

    Returns
    -------
    dict
        Parameter dictionary with keys: nu, mu_hat, corr, loglik,
        mu_daily, scale_daily, cov_daily, mu_annual, scale_annual, cov_annual.
        Returns empty dict if the JSON file is not found.
    """
    suffix = _suffix_for_dist(dist)
    path = _RESULTS_DIR / f"t_fit{suffix}.json"

    if not path.exists():
        return {}

    with open(path, "r") as f:
        t_fit = json.load(f)

    nu = float(t_fit["nu"])
    mu_hat = np.array(t_fit["mu"], dtype=float)
    corr = np.array(t_fit["Corr"], dtype=float)
    loglik = float(t_fit["logL"])

    mu_daily = mu_hat * DAILY_VOL
    scale_daily = _D_daily @ corr @ _D_daily
    cov_daily = (nu / (nu - 2.0)) * scale_daily

    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    scale_annual = scale_daily * TRADING_DAYS_PER_YEAR
    cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

    return {
        "nu": nu,
        "mu_hat": mu_hat,
        "corr": corr,
        "loglik": loglik,
        "mu_daily": mu_daily,
        "scale_daily": scale_daily,
        "cov_daily": cov_daily,
        "mu_annual": mu_annual,
        "scale_annual": scale_annual,
        "cov_annual": cov_annual,
    }


# =====================================================================
# Load all parameter sets into dictionaries keyed by GARCH dist name
# =====================================================================

AC_SKEW_T_PARAMS: Dict[str, Dict[str, Any]] = {}
"""AC skew-t parameters keyed by GARCH innovation distribution."""

EST_T_PARAMS: Dict[str, Dict[str, Any]] = {}
"""Symmetric Student-t parameters keyed by GARCH innovation distribution."""

for _dist in GARCH_DISTRIBUTIONS:
    _skewt = _load_skew_t_params(_dist)
    if _skewt:
        AC_SKEW_T_PARAMS[_dist] = _skewt

    _t = _load_t_params(_dist)
    if _t:
        EST_T_PARAMS[_dist] = _t

# =====================================================================
# Backward-compatible flat constants.~
# Default to the best-by-MLE dist ("normal") when available; otherwise fall back
# to the first available dist in priority order.
_DEFAULT_DIST = next(
    (d for d in ("normal", "t", "skewt") if d in AC_SKEW_T_PARAMS), None
)


if _DEFAULT_DIST in AC_SKEW_T_PARAMS:
    _ac = AC_SKEW_T_PARAMS[_DEFAULT_DIST]
    AC_SKEW_t_NU_v2 = _ac["nu"]
    AC_SKEW_t_MU_HAT_v2 = _ac["mu_hat"]
    AC_SKEW_t_SIGMA_DP_v2 = _ac["sigma_dp"]
    AC_SKEW_t_OMEGA_v2 = _ac["omega"]
    AC_SKEW_t_CORR_v2 = _ac["corr"]
    AC_SKEW_t_XI_DAILY_v2 = _ac["xi_daily"]
    AC_SKEW_t_SCALE_DAILY_v2 = _ac["scale_daily"]
    AC_SKEW_t_COV_DAILY_v2 = _ac["cov_daily"]
    AC_SKEW_t_XI_ANNUAL_v2 = _ac["xi_annual"]
    AC_SKEW_t_SCALE_ANNUAL_v2 = _ac["scale_annual"]
    AC_SKEW_t_COV_ANNUAL_v2 = _ac["cov_annual"]
    AC_SKEW_t_CP_MEAN_v2 = _ac["cp_mean"]
    AC_SKEW_t_CP_VAR_COV_v2 = _ac["cp_var_cov"]
    AC_SKEW_t_CP_GAMMA1_v2 = _ac["cp_gamma1"]
    AC_SKEW_t_CP_GAMMA2M_v2 = _ac["cp_gamma2m"]
    AC_SKEW_t_LOGLIK_v2 = _ac["loglik"]
    AC_SKEW_t_MEAN_DAILY_v2 = _ac["mean_daily"]
    AC_SKEW_t_MEAN_ANNUAL_v2 = _ac["mean_annual"]

if _DEFAULT_DIST in EST_T_PARAMS:
    _t_p = EST_T_PARAMS[_DEFAULT_DIST]
    EST_t_NU_v2 = _t_p["nu"]
    EST_t_MU_HAT_v2 = _t_p["mu_hat"]
    EST_t_CORR_v2 = _t_p["corr"]
    EST_t_LOGLIK_v2 = _t_p["loglik"]
    EST_t_MU_DAILY_v2 = _t_p["mu_daily"]
    EST_t_SCALE_DAILY_v2 = _t_p["scale_daily"]
    EST_t_COV_DAILY_v2 = _t_p["cov_daily"]
    EST_t_MU_ANNUAL_v2 = _t_p["mu_annual"]
    EST_t_SCALE_ANNUAL_v2 = _t_p["scale_annual"]
    EST_t_COV_ANNUAL_v2 = _t_p["cov_annual"]


if __name__ == "__main__":
    from src.constants.five_option_dataset import ASSET_TICKERS

    np.set_printoptions(precision=6, suppress=True)

    sep = "=" * 60

    print(sep)
    print("Available GARCH distributions in AC_SKEW_T_PARAMS:")
    print(f"  {list(AC_SKEW_T_PARAMS.keys())}")
    print("Available GARCH distributions in EST_T_PARAMS:")
    print(f"  {list(EST_T_PARAMS.keys())}")
    print(sep)

    for dist, params in AC_SKEW_T_PARAMS.items():
        print(f"\n{sep}")
        print(f"AC skew-t (GARCH dist='{dist}')")
        print(sep)
        print(f"  nu      : {params['nu']}")
        print(f"  log-lik : {params['loglik']}")
        print("  xi (mu_hat) :")
        for ticker, v in zip(ASSET_TICKERS, params["mu_hat"]):
            print(f"    {ticker}: {v:.6f}")
        print("  omega (slant) :")
        for ticker, v in zip(ASSET_TICKERS, params["omega"]):
            print(f"    {ticker}: {v:.6f}")
        print("  correlation matrix:")
        for row in params["corr"]:
            print("   ", row)

    for dist, params in EST_T_PARAMS.items():
        print(f"\n{sep}")
        print(f"Student t (GARCH dist='{dist}')")
        print(sep)
        print(f"  nu      : {params['nu']}")
        print(f"  log-lik : {params['loglik']}")
        print("  mu_hat :")
        for ticker, v in zip(ASSET_TICKERS, params["mu_hat"]):
            print(f"    {ticker}: {v:.6f}")
        print("  correlation matrix:")
        for row in params["corr"]:
            print("   ", row)
