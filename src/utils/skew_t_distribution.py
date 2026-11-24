"""
Utils for the skew t distribution
"""

import numpy as np
from scipy.special import gammaln


def omega_from_alpha(alpha: np.ndarray, sigma_dp: np.ndarray) -> np.ndarray:
    """
    Skewness vector from the Azzalini shape vector.

    The shape vector ``alpha`` (R's ``dp$alpha``) and the skewness vector
    ``omega`` are different objects, related by::

        omega = (diag[sigma_dp]^(1/2))^(-1) alpha

    Parameters
    ----------
    alpha : np.ndarray
        Shape vector (R's ``dp$alpha``), shape ``(N,)`` or ``(N, 1)``.
    sigma_dp : np.ndarray
        DP scale matrix on the standardized-residual scale, shape ``(N, N)``.

    Returns
    -------
    np.ndarray
        Skewness vector ``omega``, shape ``(N,)``.
    """
    return np.asarray(alpha, dtype=float).ravel() / np.sqrt(np.diag(sigma_dp))


def c_nu(nu: float) -> float:
    """
    Skewness scaling constant ``c`` of the skew-t distribution.

    The mean of a location-0, scale-s skew-t with direction ``eta`` is
    ``c * eta * s``.

    Parameters
    ----------
    nu : float
        Degrees of freedom (> 1).

    Returns
    -------
    float
        ``sqrt(nu / pi) * Gamma((nu - 1) / 2) / Gamma(nu / 2)``, evaluated in
        log-space to avoid overflow for large ``nu``.
    """
    return float(
        np.exp(0.5 * np.log(nu / np.pi) + gammaln((nu - 1) / 2) - gammaln(nu / 2))
    )


def marginal_return_parameters(
    shape_vector: np.ndarray,
    sigma_dp: np.ndarray,
    daily_vol: np.ndarray,
    trading_days: int = 252,
) -> dict[str, np.ndarray]:
    """
    Marginal one-period skew-t parameters of each asset from the AC fit.

    Takes the shape vector ``alpha`` (R's ``dp$alpha``) and the DP scale matrix
    ``sigma_dp``.  With ``C`` the correlation matrix of ``sigma_dp`` and
    ``sd = sqrt(diag(sigma_dp))``::

        eta      = C @ alpha / sqrt(1 + alpha^T C alpha)
        sigma    = sqrt(trading_days) * daily_vol * sd
        shape    = eta / sqrt(1 - eta^2)

    ``sigma`` is the annual scale of the price-change distribution: the GARCH volatility times
    ``sqrt(sigma_dp[n, n])``.  The Kim-Mallick parameter of the one-period
    marginal of asset ``n`` is ``shape[n] / s[n]`` with the one-period scale
    ``s = sigma * sqrt(period)``.

    Parameters
    ----------
    shape_vector : np.ndarray
        Shape vector ``alpha`` (R's ``dp$alpha``), shape ``(N,)`` or ``(N, 1)``.
    sigma_dp : np.ndarray
        DP scale matrix on the standardized-residual scale, shape ``(N, N)``.
    daily_vol : np.ndarray
        Daily GARCH volatilities, shape ``(N,)``.
    trading_days : int, optional
        Trading days per year used to annualize, by default 252.

    Returns
    -------
    dict[str, np.ndarray]
        ``sigma`` (annual **scale**, not a volatility), ``eta`` (marginal
        skewness direction) and ``shape`` (one-period Azzalini shape), each of
        shape ``(N,)``.

    Raises
    ------
    ValueError
        If any ``|eta_i| >= 1``.
    """
    alpha = np.asarray(shape_vector, dtype=float).ravel()
    sd = np.sqrt(np.diag(sigma_dp))
    corr = sigma_dp / np.outer(sd, sd)
    eta = corr @ alpha / np.sqrt(1.0 + alpha @ corr @ alpha)
    if not np.all(np.abs(eta) < 1.0):
        raise ValueError(f"Implied |eta| >= 1 (eta={eta}).")
    return {
        "sigma": np.sqrt(trading_days) * np.asarray(daily_vol, dtype=float) * sd,
        "eta": eta,
        "shape": eta / np.sqrt(1.0 - eta**2),
    }
