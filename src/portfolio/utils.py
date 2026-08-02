"""
Utilities for portfolio management.
"""

import numpy as np


def number_of_shares_to_weights(
    x: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Convert number of shares to portfolio weights.

    Parameters
    ----------
    x : np.ndarray
        Number of shares for each asset.
    v : np.ndarray
        Current prices of the option values.

    Returns
    -------
    np.ndarray
        Portfolio weights.
    """
    w = (v.ravel() * x.ravel()) / np.dot(v.ravel(), x.ravel())
    return w


def weights_to_number_of_shares(
    w: np.ndarray,
    v: np.ndarray,
    V: float = 1.0,
) -> np.ndarray:
    """
    Convert portfolio weights to number of shares.

    Parameters
    ----------
    w : np.ndarray
        Portfolio weights.
    v : np.ndarray
        Current prices of the option values.
    V : float
        Total portfolio value.

    Returns
    -------
    np.ndarray
        Number of shares for each asset.
    """
    x = V * w.ravel() / v.ravel()
    return x


def build_scale_matrix(
    time_period_for_risk_measurements: float,
    correlation: float | np.ndarray,
    volatilities: np.ndarray,
    spot_prices: np.ndarray,
) -> np.ndarray:
    """
    Build the scale matrix for asset *price changes* \\Delta S under
    multivariable Student t-distribution.

    For a t-distribution with degrees of freedom \\nu, the covariance of \\Delta S is
    ``Cov[\\Delta S] = \\nu/(\\nu-2) \\times \\Sigma`` where \\Sigma is the scale matrix returned here.
    No ``(\\nu-2)/\\nu`` correction is applied; the OptionPortfolio formulas
    already embed the appropriate \\nu-dependent coefficients.

    The computation is:

        \\Sigma = T \\times diag(S) @ diag(\\sigma) @ C @ diag(\\sigma) @ diag(S)

    where *T* is the time period (years), *S* the spot prices, *\\sigma* the
    annualized GARCH volatilities, and *C* the fitted correlation matrix.
    Equivalently, with daily quantities: ``\\Sigma = T_days \\times diag(S) @ diag(\\sigma_d)
    @ C @ diag(\\sigma_d) @ diag(S)`` since ``T \\times \\sigma_annual^2 = T_days \\times \\sigma_daily^2``.

    Parameters
    ----------
    time_period_for_risk_measurements : float
        Time period for risk measurements in years (e.g. ``1/252`` for one
        trading day).
    correlation : float | np.ndarray
        Pairwise correlation coefficient (float for uniform off-diagonal)
        or full (n \\times n) correlation matrix.
    volatilities : np.ndarray
        Annualized GARCH unconditional volatilities of the underlying assets.
    spot_prices : np.ndarray
        Spot prices of the underlying assets.

    Returns
    -------
    np.ndarray
        Scale matrix for the multivariable t-distribution of \\Delta S
        over the specified time period.
    """
    volatilities_diag = np.diag(volatilities)
    spot_prices_diag_matrix = np.diag(spot_prices)

    n = len(volatilities)

    if isinstance(correlation, np.ndarray) and correlation.shape != (n, n):
        raise ValueError(
            "If passing a matrix: correlation matrix must be of shape (n, n)."
        )

    if isinstance(correlation, float):
        correlation_matrix = np.full((n, n), correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
    else:
        correlation_matrix = correlation

    scale_matrix = time_period_for_risk_measurements * (
        spot_prices_diag_matrix
        @ volatilities_diag
        @ correlation_matrix
        @ volatilities_diag
        @ spot_prices_diag_matrix
    )

    return scale_matrix
