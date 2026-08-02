"""
Utils for the skew t distribution
"""

import functools

import numpy as np
from scipy.integrate import quad
from scipy.stats import t


def _skewt_pdf(xi: float, alpha_skew: float, nu: float) -> float:
    """
    1-D Azzalini skew-t probability density function.

    Parameters
    ----------
    xi : float
        Point at which to evaluate the density.
    alpha_skew : float
        Skewness (slant) parameter.
    nu : float
        Degrees of freedom (> 0).

    Returns
    -------
    float
        Density value f(xi; alpha_skew, nu).
    """
    inner = alpha_skew * xi * np.sqrt((nu + 1) / (xi**2 + nu))
    return 2.0 * t.pdf(xi, df=nu) * t.cdf(inner, df=nu + 1)


@functools.lru_cache(maxsize=None)
def _skewt_cdf(x: float, alpha_skew: float, nu: float) -> float:
    """
    Cumulative distribution function of the 1-D Azzalini skew-t.

    Computed via numerical integration of the density from ``-inf`` to ``x``.
    Results are cached so that repeated calls with the same arguments (e.g.
    from ``_compute_price_with_*`` bump methods that keep ``x_p``, ``x_c``,
    ``alpha_skew``, and ``nu`` fixed) are free after the first evaluation.

    Parameters
    ----------
    x : float
        Upper integration limit.
    alpha_skew : float
        Skewness (slant) parameter.
    nu : float
        Degrees of freedom (> 0).

    Returns
    -------
    float
        P(X <= x) for X ~ skew-t(0, 1, alpha_skew, nu).
    """
    result, _ = quad(
        lambda xi: _skewt_pdf(xi, alpha_skew, nu),
        -np.inf,
        x,
    )
    return float(result)


def compute_alpha_1d_from_omega(omega: np.ndarray, sigma_dp: np.ndarray) -> np.ndarray:
    """
    Derive the per-asset 1-D Azzalini skewness parameters from the multivariate
    AC skew-t DP parameterisation.

    For the Azzalini-Capitanio multivariate skew-t with DP scale matrix
    ``sigma_dp`` and slant vector ``omega``, the i-th component's marginal
    skewness parameter is::

        h_i     = (sigma_dp @ omega)_i / sqrt(1 + omega^T sigma_dp omega)
        delta_i = h_i / sqrt(sigma_dp[i, i])
        alpha_i = delta_i / sqrt(1 - delta_i^2)

    Parameters
    ----------
    omega : np.ndarray
        Slant (skewness) vector of the multivariate skew-t, shape ``(N,)`` or
        ``(N, 1)``.
    sigma_dp : np.ndarray
        DP scale matrix of the multivariate skew-t, shape ``(N, N)``.
        Typically on the standardized-residual scale so that the derived
        ``alpha_i`` values are dimensionless and independent of Delta_t.

    Returns
    -------
    np.ndarray
        1-D Azzalini skewness parameter for each asset, shape ``(N,)``.

    Raises
    ------
    ValueError
        If any diagonal entry of ``sigma_dp`` is non-positive, or if any
        implied ``|delta_i| >= 1`` (which would make ``alpha_i`` undefined).
    """
    diag_entries = np.diag(sigma_dp)
    if not np.all(diag_entries > 0):
        bad = np.where(diag_entries <= 0)[0].tolist()
        raise ValueError(
            f"sigma_dp has non-positive diagonal entries at indices {bad}. "
            "All diagonal entries must be strictly positive."
        )

    omega_col = np.asarray(omega, dtype=float).reshape(-1, 1)
    h = (
        sigma_dp
        @ omega_col
        / np.sqrt(1.0 + (omega_col.T @ sigma_dp @ omega_col).item())
    )
    delta = h.ravel() / np.sqrt(diag_entries)
    delta2 = delta**2
    if not np.all(delta2 < 1.0):
        bad = np.where(delta2 >= 1.0)[0].tolist()
        raise ValueError(
            f"Implied |delta_i| >= 1 at indices {bad} (delta2={delta2[bad]}). "
            "This indicates numerically invalid inputs (e.g., a near-singular or "
            "corrupted scale matrix). Ensure sigma_dp is a valid positive-definite "
            "matrix and omega has been computed from compatible parameters."
        )
    return delta / np.sqrt(1.0 - delta2)


def compute_omega_delta_s(
    omega_dp: np.ndarray,
    sigma_dp: np.ndarray,
    volatilities: np.ndarray,
    spot_prices: np.ndarray,
    time_period: float,
) -> np.ndarray:
    """
    Transform the AC skew-t slant vector from the DP (standardized-residual)
    scale to the Delta_S (price-change) scale.

    ``SkewTOptionPortfolio`` operates on price changes Delta_S and receives a
    scale matrix built via :func:`~src.portfolio.utils.build_scale_matrix`,
    which is on the Delta_S scale::

        Sigma_Delta_S = T * diag(S) @ diag(sigma) @ C @ diag(sigma) @ diag(S)
                     = D_norm @ Sigma_DP @ D_norm^T

    where ``D_norm = diag(sqrt(T) * S_i * sigma_i / sigma_dp_i)`` and
    ``sigma_dp_i = sqrt(Sigma_DP[i, i])``.

    Because the slant vector is tied to the scale matrix in the AC skew-t
    parameterisation, changing scales requires transforming omega accordingly so
    that the implied skewness direction ``h`` is consistent::

        h_Delta_S = D_norm @ h_DP

    This is achieved by::

        omega_Delta_S = D_norm^{-1} @ omega_DP
                     = diag(sigma_dp_i / (sqrt(T) * S_i * sigma_i)) @ omega_DP

    Parameters
    ----------
    omega_dp : np.ndarray
        Slant vector on the DP (standardized-residual) scale, shape ``(N,)``
        or ``(N, 1)``.
    sigma_dp : np.ndarray
        DP scale matrix on the standardized-residual scale, shape ``(N, N)``.
        Must have strictly positive diagonal entries.
    volatilities : np.ndarray
        Annualized asset volatilities sigma_i, shape ``(N,)``.
    spot_prices : np.ndarray
        Asset spot prices S_i, shape ``(N,)``.
    time_period : float
        Risk measurement period Delta_t in years (e.g. ``1/252`` for one trading
        day).

    Returns
    -------
    np.ndarray
        Slant vector on the Delta_S scale, shape ``(N,)``.

    Raises
    ------
    ValueError
        If any diagonal entry of ``sigma_dp`` is non-positive.
    """
    diag_entries = np.diag(sigma_dp)
    if not np.all(diag_entries > 0):
        bad = np.where(diag_entries <= 0)[0].tolist()
        raise ValueError(
            f"sigma_dp has non-positive diagonal entries at indices {bad}. "
            "All diagonal entries must be strictly positive."
        )

    sigma_dp_diag = np.sqrt(diag_entries)
    vols = np.asarray(volatilities, dtype=float)
    spots = np.asarray(spot_prices, dtype=float)
    scale = sigma_dp_diag / (np.sqrt(time_period) * spots * vols)
    return scale * np.asarray(omega_dp, dtype=float).ravel()
