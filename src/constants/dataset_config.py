"""
Unified parameter configuration for all experiments.

Provides a ``DatasetConfig`` dataclass that bundles every distributional
parameter required to run an experiment under a specific model assumption.
Pre-built configs are exported for the common cases:

- ``HK_NORMAL_CONFIG``   - Hu & Kercheval (2010) Normal fit
- ``HK_T_CONFIG``        - Hu & Kercheval (2010) Student-t fit
- ``GAUSSIAN_GARCH_SKEWT_CONFIG`` - AC skew-t estimated with Gaussian GARCH
  (best MLE, logL = -4885.46)
- ``GAUSSIAN_GARCH_T_CONFIG`` - symmetric t estimated with Gaussian GARCH
- ``T_GARCH_SKEWT_CONFIG``    - AC skew-t estimated with t-GARCH
- ``T_GARCH_T_CONFIG``        - symmetric t estimated with t-GARCH
- ``SKEWT_GARCH_SKEWT_CONFIG``- AC skew-t estimated with skewt-GARCH
- ``SKEWT_GARCH_T_CONFIG``    - symmetric t estimated with skewt-GARCH
- ``OLD_SKEWT_CONFIG`` - AC skew-t v1 (unscaled, original estimation)
- ``OLD_T_CONFIG``     - Student-t v1 (unscaled, original estimation)

Usage
-----
    from src.constants.dataset_config import GAUSSIAN_GARCH_SKEWT_CONFIG as cfg

    nu = cfg.nu
    corr = cfg.corr
    omega = cfg.omega  # None for symmetric distributions

Switching between parameter sets is a one-line import change.

Toggle between old and new skew-t fits::

    # Use new (v2) scaled skew-t fit (default)
    from src.constants.dataset_config import GAUSSIAN_GARCH_SKEWT_CONFIG as cfg
    assert cfg.use_old_skew_t == False

    # Use old (v1) unscaled skew-t fit
    from src.constants.dataset_config import OLD_SKEWT_CONFIG as cfg
    assert cfg.use_old_skew_t == True
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.constants.five_option_dataset import (
    ANNUAL_VOL,
    ASSET_NAMES,
    ASSET_TICKERS,
    DAILY_VOL,
    STOCK_PRICES,
    TRADING_DAYS_PER_YEAR,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable container for all model parameters needed by experiments.

    Parameters
    ----------
    name : str
        Human-readable label for this configuration (e.g.
        ``"Gaussian GARCH / AC skew-t"``).
    distribution : str
        Distributional model type: ``"normal"``, ``"t"``, or ``"skewt"``.
    garch_dist : str | None
        GARCH innovation distribution used during filtering.
        ``None`` for literature-sourced parameters (Hu & Kercheval).
    nu : float
        Degrees of freedom.  Set to a large value (e.g. 1000) for Normal.
    mu_hat : np.ndarray
        Location vector on the GARCH standardized-residual scale, shape (5,).
    corr : np.ndarray
        Correlation matrix, shape (5, 5).
    omega : np.ndarray | None
        AC skew-t slant vector (``None`` for symmetric distributions).
    sigma_dp : np.ndarray | None
        AC skew-t DP scale matrix (``None`` for symmetric distributions).
    loglik : float | None
        Log-likelihood at the MLE solution (``None`` if not available).
    cp_mean : np.ndarray | None
        Centred-parameterisation mean vector (AC skew-t only).
    cp_gamma1 : np.ndarray | None
        Centred-parameterisation skewness (AC skew-t only).
    use_old_skew_t : bool
        If True, uses v1 (unscaled) skew-t parameters. If False, uses v2 (scaled).
        Only relevant for skew-t distributions. Default is False (v2).

    Derived (computed at construction):
    mu_daily, mu_annual, scale_daily, scale_annual, cov_daily, cov_annual,
    mean_daily, mean_annual (for skew-t, uses cp_mean; for others, uses mu_hat).
    """

    name: str
    distribution: str
    garch_dist: Optional[str]
    nu: float
    mu_hat: np.ndarray
    corr: np.ndarray
    omega: Optional[np.ndarray] = None
    sigma_dp: Optional[np.ndarray] = None
    loglik: Optional[float] = None
    cp_mean: Optional[np.ndarray] = None
    cp_gamma1: Optional[np.ndarray] = None
    use_old_skew_t: bool = False

    # --- Derived quantities (set via __post_init__) ---
    mu_daily: np.ndarray = None  # type: ignore[assignment]
    mu_annual: np.ndarray = None  # type: ignore[assignment]
    scale_daily: np.ndarray = None  # type: ignore[assignment]
    scale_annual: np.ndarray = None  # type: ignore[assignment]
    cov_daily: np.ndarray = None  # type: ignore[assignment]
    cov_annual: np.ndarray = None  # type: ignore[assignment]
    mean_daily: np.ndarray = None  # type: ignore[assignment]
    mean_annual: np.ndarray = None  # type: ignore[assignment]

    # --- Shared constants (same for all configs) ---
    asset_names: tuple = tuple(ASSET_NAMES)
    asset_tickers: tuple = tuple(ASSET_TICKERS)
    stock_prices: np.ndarray = None  # type: ignore[assignment]
    daily_vol: np.ndarray = None  # type: ignore[assignment]
    annual_vol: np.ndarray = None  # type: ignore[assignment]
    trading_days: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        """Compute derived quantities from the core parameters."""
        # Shared constants
        object.__setattr__(self, "stock_prices", STOCK_PRICES)
        object.__setattr__(self, "daily_vol", DAILY_VOL)
        object.__setattr__(self, "annual_vol", ANNUAL_VOL)

        d_daily = np.diag(DAILY_VOL)

        # Daily mu (location * vol)
        mu_daily = self.mu_hat * DAILY_VOL
        object.__setattr__(self, "mu_daily", mu_daily)

        # Annual mu
        mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
        object.__setattr__(self, "mu_annual", mu_annual)

        # Scale and covariance matrices
        scale_daily = d_daily @ self.corr @ d_daily
        object.__setattr__(self, "scale_daily", scale_daily)

        scale_annual = scale_daily * TRADING_DAYS_PER_YEAR
        object.__setattr__(self, "scale_annual", scale_annual)

        cov_daily = (self.nu / (self.nu - 2.0)) * scale_daily
        object.__setattr__(self, "cov_daily", cov_daily)

        cov_annual = cov_daily * TRADING_DAYS_PER_YEAR
        object.__setattr__(self, "cov_annual", cov_annual)

        # Mean daily/annual: use cp_mean for skew-t, mu_hat otherwise
        if self.cp_mean is not None:
            mean_daily = self.cp_mean * DAILY_VOL
        else:
            mean_daily = mu_daily
        object.__setattr__(self, "mean_daily", mean_daily)

        mean_annual = mean_daily * TRADING_DAYS_PER_YEAR
        object.__setattr__(self, "mean_annual", mean_annual)

    @property
    def is_skew_t(self) -> bool:
        """Whether this config represents a skew-t distribution."""
        return self.distribution == "skewt"

    @property
    def is_symmetric(self) -> bool:
        """Whether this config represents a symmetric distribution (normal or t)."""
        return self.distribution in ("normal", "t")


# =====================================================================
# Pre-built configurations
# =====================================================================


def _load_old_skewt_params_v1() -> Optional[dict]:
    """Load the original v1 (unscaled) AC skew-t parameters.

    Returns
    -------
    dict or None
        Parameter dictionary with keys: nu, mu_hat, sigma_dp, omega, corr,
        cp_mean, cp_gamma1, loglik, or None if JSON file not found.
    """
    from pathlib import Path

    _RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / (
        "experiment_t_vs_skew_t_estimation/estimation_results"
    )

    # Try to find the file with or without the _t suffix
    possible_paths = [
        _RESULTS_DIR / "skew_t_ac_fit.json",
        _RESULTS_DIR / "skew_t_ac_fit_t.json",
    ]

    path = None
    for p in possible_paths:
        if p.exists():
            path = p
            break

    if path is None:
        return None

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

    # Centred parameterisation (CP) - actual moments
    cp_mean = np.array(cp["mean"], dtype=float)
    cp_gamma1 = np.array(cp["gamma1"], dtype=float)
    loglik = float(ac_fit["logL"])

    return {
        "nu": nu,
        "mu_hat": mu_hat,
        "sigma_dp": sigma_dp,
        "omega": omega,
        "corr": corr,
        "cp_mean": cp_mean,
        "cp_gamma1": cp_gamma1,
        "loglik": loglik,
    }


def _load_old_t_params_v1() -> Optional[dict]:
    """Load the original v1 (unscaled) symmetric Student-t parameters.

    Returns
    -------
    dict or None
        Parameter dictionary with keys: nu, mu_hat, corr, loglik,
        or None if JSON file not found.
    """
    from pathlib import Path

    _RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / (
        "experiment_t_vs_skew_t_estimation/estimation_results"
    )
    path = _RESULTS_DIR / "t_fit.json"

    if not path.exists():
        return None

    import json

    with open(path, "r") as f:
        t_fit = json.load(f)

    nu = float(t_fit["nu"])
    mu_hat = np.array(t_fit["mu"], dtype=float)
    corr = np.array(t_fit["Corr"], dtype=float)
    loglik = float(t_fit["logL"])

    return {
        "nu": nu,
        "mu_hat": mu_hat,
        "corr": corr,
        "loglik": loglik,
    }


def _build_hk_normal_config() -> DatasetConfig:
    """Hu & Kercheval (2010) Normal fit parameters."""
    from src.constants.five_option_dataset import NORMAL_CORR, NORMAL_MU_HAT

    return DatasetConfig(
        name="HK Normal",
        distribution="normal",
        garch_dist=None,
        nu=1e6,  # effectively infinite for Normal
        mu_hat=NORMAL_MU_HAT,
        corr=NORMAL_CORR,
    )


def _build_hk_t_config() -> DatasetConfig:
    """Hu & Kercheval (2010) Student-t fit parameters."""
    from src.constants.five_option_dataset import t_CORR, t_MU_HAT, t_NU

    return DatasetConfig(
        name="HK Student-t",
        distribution="t",
        garch_dist=None,
        nu=t_NU,
        mu_hat=t_MU_HAT,
        corr=t_CORR,
    )


def _build_estimated_skewt_config(garch_dist: str) -> Optional[DatasetConfig]:
    """Build AC skew-t config from comparison_results JSON files."""
    from src.constants.five_option_dataset_estimated_v2 import AC_SKEW_T_PARAMS

    params = AC_SKEW_T_PARAMS.get(garch_dist)
    if params is None:
        return None

    return DatasetConfig(
        name=f"{garch_dist.capitalize()} GARCH / AC skew-t",
        distribution="skewt",
        garch_dist=garch_dist,
        nu=params["nu"],
        mu_hat=params["mu_hat"],
        corr=params["corr"],
        omega=params["omega"],
        sigma_dp=params["sigma_dp"],
        loglik=params["loglik"],
        cp_mean=params["cp_mean"],
        cp_gamma1=params["cp_gamma1"],
    )


def _build_estimated_t_config(garch_dist: str) -> DatasetConfig:
    """Build symmetric Student-t config from comparison_results JSON files."""
    from src.constants.five_option_dataset_estimated_v2 import EST_T_PARAMS

    params = EST_T_PARAMS.get(garch_dist)
    if params is None:
        raise ValueError(f"No estimated t parameters found for garch_dist={garch_dist}")

    return DatasetConfig(
        name=f"{garch_dist.capitalize()} GARCH / Student-t",
        distribution="t",
        garch_dist=garch_dist,
        nu=params["nu"],
        mu_hat=params["mu_hat"],
        corr=params["corr"],
        loglik=params["loglik"],
    )


def _build_estimated_skewt_config_old_v1() -> Optional[DatasetConfig]:
    """Build AC skew-t config from old v1 (unscaled) JSON file."""
    params = _load_old_skewt_params_v1()
    if params is None:
        return None

    return DatasetConfig(
        name="AC skew-t (unscaled, v1)",
        distribution="skewt",
        garch_dist=None,
        nu=params["nu"],
        mu_hat=params["mu_hat"],
        corr=params["corr"],
        omega=params["omega"],
        sigma_dp=params["sigma_dp"],
        loglik=params["loglik"],
        cp_mean=params["cp_mean"],
        cp_gamma1=params["cp_gamma1"],
        use_old_skew_t=True,
    )


def _build_estimated_t_config_old_v1() -> Optional[DatasetConfig]:
    """Build symmetric Student-t config from old v1 (unscaled) JSON file."""
    params = _load_old_t_params_v1()
    if params is None:
        return None

    return DatasetConfig(
        name="Student-t (unscaled, v1)",
        distribution="t",
        garch_dist=None,
        nu=params["nu"],
        mu_hat=params["mu_hat"],
        corr=params["corr"],
        loglik=params["loglik"],
    )


# --- Hu & Kercheval (2010) literature parameters ---
HK_NORMAL_CONFIG = _build_hk_normal_config()
HK_T_CONFIG = _build_hk_t_config()

# --- Estimated parameters (Gaussian GARCH = best MLE) ---
GAUSSIAN_GARCH_SKEWT_CONFIG = _build_estimated_skewt_config("normal")
GAUSSIAN_GARCH_T_CONFIG = _build_estimated_t_config("normal")

# --- Estimated parameters (t-GARCH) ---
T_GARCH_SKEWT_CONFIG = _build_estimated_skewt_config("t")
T_GARCH_T_CONFIG = _build_estimated_t_config("t")

# --- Estimated parameters (skewt-GARCH) ---
SKEWT_GARCH_SKEWT_CONFIG = _build_estimated_skewt_config("skewt")
SKEWT_GARCH_T_CONFIG = _build_estimated_t_config("skewt")

# --- Old v1 (unscaled) estimated parameters ---
OLD_SKEWT_CONFIG = _build_estimated_skewt_config_old_v1()
OLD_T_CONFIG = _build_estimated_t_config_old_v1()

# --- Convenience: default config is the best-MLE skew-t ---
DEFAULT_SKEWT_CONFIG = GAUSSIAN_GARCH_SKEWT_CONFIG
DEFAULT_T_CONFIG = GAUSSIAN_GARCH_T_CONFIG

# --- All configs in a dict for iteration ---
ALL_CONFIGS = {
    "hk_normal": HK_NORMAL_CONFIG,
    "hk_t": HK_T_CONFIG,
    "gaussian_garch_skewt": GAUSSIAN_GARCH_SKEWT_CONFIG,
    "gaussian_garch_t": GAUSSIAN_GARCH_T_CONFIG,
    "t_garch_skewt": T_GARCH_SKEWT_CONFIG,
    "t_garch_t": T_GARCH_T_CONFIG,
    "skewt_garch_skewt": SKEWT_GARCH_SKEWT_CONFIG,
    "skewt_garch_t": SKEWT_GARCH_T_CONFIG,
    "old_skewt_v1": OLD_SKEWT_CONFIG,
    "old_t_v1": OLD_T_CONFIG,
}

# Filter out None values (in case JSON files are missing)
ALL_CONFIGS = {k: v for k, v in ALL_CONFIGS.items() if v is not None}

# All skew-t configs for out-of-sample looping
SKEWT_CONFIGS = {
    k: v for k, v in ALL_CONFIGS.items() if v is not None and v.distribution == "skewt"
}

# All symmetric-t configs
T_CONFIGS = {
    k: v
    for k, v in ALL_CONFIGS.items()
    if v is not None and v.distribution == "t" and v.garch_dist is not None
}
