"""
Monte Carlo helpers: skew-t sampling and moment estimators.
"""

import numpy as np
from scipy.stats import norm

from src.portfolio.option_portfolio import OptionPortfolio
from src.pricing.models import SkewTConvolutionModel

N_SPOT_GRID = 801
SPOT_RANGE = (0.2, 1.8)  # spot grid as a multiple of the current spot


def repricing_grids(
    pf: OptionPortfolio, rate: float, period: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Exact option prices at ``t + Delta t`` on a grid of spots, one grid per option.

    Parameters
    ----------
    pf : OptionPortfolio
        Portfolio whose options are repriced.
    rate : float
        Risk-free rate.
    period : float
        Risk-measurement period ``Delta t`` in years.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Spot grid and prices of each option.
    """
    grids = []
    for opt in pf.options:
        spots = opt.underlying.spot * np.linspace(*SPOT_RANGE, N_SPOT_GRID)
        tau = opt.time_to_maturity - period
        prices = np.array(
            [opt.model.price(s, opt.strike, tau, rate, opt.kind) for s in spots]
        )
        grids.append((spots, prices))
    return grids


def option_gains(
    pf: OptionPortfolio,
    grids: list[tuple[np.ndarray, np.ndarray]],
    d_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact and delta-gamma option gains for simulated price changes.

    Parameters
    ----------
    pf : OptionPortfolio
        The portfolio.
    grids : list[tuple[np.ndarray, np.ndarray]]
        Output of :func:`repricing_grids`.
    d_s : np.ndarray
        Simulated price changes, shape ``(n, N)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Full-repricing and delta-gamma gains of each option, shape ``(n, M)`` each.
    """
    dt = pf.time_period_for_risk_measurements
    exact = np.empty((d_s.shape[0], pf.M))
    approx = np.empty_like(exact)
    for m, (opt, (spots, prices)) in enumerate(zip(pf.options, grids)):
        move = d_s[:, pf.underlying_map[opt.underlying.name]]
        exact[:, m] = (
            np.interp(opt.underlying.spot + move, spots, prices)
            - pf.option_values[m, 0]
        )
        approx[:, m] = (
            pf.thetas[m, 0] * dt
            + pf.delta_per_option_list[m, 0] * move
            + 0.5 * pf.gammas[m, 0] * move**2
        )
    return exact, approx


def sample_skew_t(
    n: int,
    location: np.ndarray,
    scale_matrix: np.ndarray,
    omega: np.ndarray,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample the multivariate skew-t ``t_N^skew(location, scale_matrix, omega, nu)``.

    With ``h = Sigma omega / sqrt(1 + omega^T Sigma omega)``, ``U0 ~ N(0, 1)``,
    ``V ~ N(0, Sigma - h h^T)`` and ``W ~ chi^2_nu`` independent, the sample is
    ``location + (h |U0| + V) / sqrt(W / nu)``; ``nu = inf`` gives the skew normal.

    Parameters
    ----------
    n : int
        Number of samples.
    location : np.ndarray
        Location vector, shape ``(N,)``.
    scale_matrix : np.ndarray
        Scale matrix ``Sigma``, shape ``(N, N)``.
    omega : np.ndarray
        Skewness vector, shape ``(N,)``.
    nu : float
        Degrees of freedom.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Samples, shape ``(n, N)``.
    """
    location = np.asarray(location, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    h = scale_matrix @ omega / np.sqrt(1.0 + omega @ scale_matrix @ omega)
    residual = scale_matrix - np.outer(h, h)
    chol = np.linalg.cholesky(residual + 1e-14 * np.eye(len(h)))
    z = (
        np.abs(rng.standard_normal((n, 1))) * h
        + rng.standard_normal((n, len(h))) @ chol.T
    )
    if np.isfinite(nu):
        z = z / np.sqrt(rng.chisquare(nu, size=(n, 1)) / nu)
    return location + z


def sample_skew_t_1d(
    size: tuple[int, ...],
    scale: float,
    shape: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample the one-dimensional Azzalini skew-t with location 0.

    Parameters
    ----------
    size : tuple[int, ...]
        Shape of the output.
    scale : float
        Scale (> 0).
    shape : float
        Azzalini shape.
    nu : float
        Degrees of freedom; ``np.inf`` gives the skew normal.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Samples of shape ``size``.
    """
    delta = shape / np.sqrt(1.0 + shape**2)
    z = delta * np.abs(rng.standard_normal(size)) + np.sqrt(
        1.0 - delta**2
    ) * rng.standard_normal(size)
    if np.isfinite(nu):
        z = z / np.sqrt(rng.chisquare(nu, size=size) / nu)
    return scale * z


def simulate_log_returns(
    model: SkewTConvolutionModel,
    tau: float,
    rng: np.random.Generator,
    n_batches: int = 20,
    paths_per_batch: int = 50_000,
) -> np.ndarray:
    """
    Simulate the truncated centered log return over the maturity, in batches.

    Parameters
    ----------
    model : SkewTConvolutionModel
        Pricing model, giving the one-period law and the truncation.
    tau : float
        Time to maturity in years.
    rng : np.random.Generator
        Random number generator.
    n_batches : int, optional
        Number of independent batches, by default 20.
    paths_per_batch : int, optional
        Number of paths of each batch, by default 50,000.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_batches, paths_per_batch)``; paths outside the
        truncation window are discarded and redrawn.
    """
    d = model.n_periods(tau)
    scale = model.sigma * np.sqrt(model.period)
    out = np.empty((n_batches, paths_per_batch))
    for i in range(n_batches):
        kept = np.empty(0)
        while kept.size < paths_per_batch:
            y = sample_skew_t_1d(
                (paths_per_batch, d), scale, model.shape, model.nu, rng
            ).sum(axis=1)
            kept = np.concatenate([kept, y[(y >= model.a) & (y <= model.b)]])
        out[i] = kept[:paths_per_batch]
    return out


def monte_carlo_price(
    y: np.ndarray, spot: float, strike: float, tau: float, rate: float, kind: str
) -> tuple[float, float]:
    """
    Monte Carlo price and standard error from batches of centered log returns.

    Parameters
    ----------
    y : np.ndarray
        Batches of centered log returns.
    spot : float
        Spot price.
    strike : float
        Strike price.
    tau : float
        Time to maturity in years.
    rate : float
        Risk-free rate.
    kind : str
        ``"call"`` or ``"put"``.

    Returns
    -------
    tuple[float, float]
        Price and standard error.
    """
    s_t = spot * np.exp(rate * tau) * np.exp(y) / np.exp(y).mean()
    payoff = np.maximum(s_t - strike if kind == "call" else strike - s_t, 0.0)
    disc = np.exp(-rate * tau)
    se = disc * payoff.mean(axis=1).std(ddof=1) / np.sqrt(y.shape[0])
    return float(disc * payoff.mean()), float(se)


def cfvar3_estimate(gains: np.ndarray, alpha: float) -> float:
    """
    Cornish-Fisher VaR with three terms, from the moment estimators of a sample.

    The mean is the sample mean, the variance is the unbiased sample variance,
    and the skewness is the adjusted Fisher-Pearson standardized moment
    coefficient ``G1 = sqrt(n (n - 1)) / (n - 2) * m3 / m2^(3/2)``, where ``m2``
    and ``m3`` are the biased second and third central sample moments.  The
    third central moment is estimated by ``G1 s^3``, ``s^2`` being the variance
    estimate, so that ``kappa_3 / variance = G1 s``.

    Parameters
    ----------
    gains : np.ndarray
        Sample of portfolio gains, shape ``(n,)``, ``n > 2``.
    alpha : float
        Tail risk.

    Returns
    -------
    float
        ``-mean - z s - (z^2 - 1) / 6 * G1 s`` with ``z = Phi^{-1}(alpha)``.
    """
    n = gains.size
    centered = gains - gains.mean()
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)
    std = np.sqrt(m2 * n / (n - 1))
    g1 = np.sqrt(n * (n - 1)) / (n - 2) * m3 / m2**1.5
    z = norm.ppf(alpha)
    return float(-gains.mean() - z * std - (z**2 - 1.0) / 6.0 * g1 * std)
