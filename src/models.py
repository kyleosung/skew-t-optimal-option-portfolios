"""
Models of the standardized residuals, and the option portfolios they give.

A :class:`Model` is a distribution of the standardized GARCH residuals ``Z``: the skew-t,
its Student's t counterpart without skewness, the refitted Student's t, or the normal.
:func:`build_portfolio` prices at-the-money options on the assets with it, using the
GARCH inputs of a :class:`Market`, and returns the portfolio whose weights the paper
optimizes.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.estimation import Fit
from src.options.european_option import EuropeanOption, Underlying
from src.portfolio.option_portfolio import OptionPortfolio
from src.portfolio.utils import build_scale_matrix
from src.pricing.base import PricingModel
from src.pricing.models import BlackScholesModel, SkewTConvolutionModel
from src.settings import (
    MIN_T_NU,
    NORMAL_NU,
    RISK_FREE_RATE,
    TIME_PERIOD,
    TIME_TO_MATURITY,
    TRADING_DAYS_PER_YEAR,
)
from src.utils.skew_t_distribution import (
    c_nu,
    marginal_return_parameters,
    omega_from_alpha,
)


@dataclass(frozen=True)
class Model:
    """
    Distribution of the standardized residuals ``Z ~ t_N^skew(mu, Sigma, alpha, nu)``.

    Parameters
    ----------
    nu : float
        Degrees of freedom.
    mean : np.ndarray
        Mean ``E[Z]``.
    scale : np.ndarray
        Scale matrix ``Sigma^Z``.
    shape : np.ndarray
        Shape vector ``alpha^Z`` of Azzalini and Capitanio, zero without skewness.
    black_scholes : bool, optional
        Whether the options are priced by Black-Scholes (the normal model) instead of
        the skew-t pricing model, by default ``False``.
    """

    nu: float
    mean: np.ndarray
    scale: np.ndarray
    shape: np.ndarray
    black_scholes: bool = False


@dataclass(frozen=True)
class Market:
    """
    Inputs of the options and of their portfolios on a day.

    Parameters
    ----------
    tickers : tuple[str, ...]
        Assets.
    spots : np.ndarray
        Prices, which are also the strikes of the at-the-money options.
    iota : np.ndarray
        GARCH conditional mean of the daily log returns.
    daily_vol : np.ndarray
        GARCH volatility forecast of the day.
    rate : float, optional
        Continuously compounded risk-free rate, by default ``RISK_FREE_RATE``.
    period : float, optional
        Risk-measurement period ``Delta t`` in years, by default one trading day.
    maturity : float, optional
        Time to maturity of the options in years, by default two months.
    """

    tickers: tuple[str, ...]
    spots: np.ndarray
    iota: np.ndarray
    daily_vol: np.ndarray
    rate: float = RISK_FREE_RATE
    period: float = TIME_PERIOD
    maturity: float = TIME_TO_MATURITY

    @classmethod
    def from_fit(cls, fit: Fit) -> "Market":
        """
        Market on the last day of the estimation sample.

        Parameters
        ----------
        fit : Fit
            Fit of the estimation sample.

        Returns
        -------
        Market
            Market with the last prices and the GARCH inputs of ``fit``.
        """
        return cls(
            fit.tickers,
            fit.spots,
            fit.garch["iota"].to_numpy(),
            fit.garch["sigma_forecast"].to_numpy(),
        )


@dataclass(frozen=True)
class Models:
    """
    The models of a :class:`Fit`.

    Parameters
    ----------
    skew_t : Model
        The skew-t fit.
    student_t : Model
        The Student's t with the mean, scale matrix and degrees of freedom of the
        skew-t, and no skewness.
    refit_t : Model
        The Student's t fitted to the same residuals, with the degrees of freedom of
        the skew-t if its own are below ``MIN_T_NU``.
    normal : Model
        The normal fit.
    """

    skew_t: Model
    student_t: Model
    refit_t: Model
    normal: Model

    @classmethod
    def from_fit(cls, fit: Fit) -> "Models":
        """
        Models of a fit.

        Parameters
        ----------
        fit : Fit
            Fit of the estimation sample.

        Returns
        -------
        Models
            The skew-t, Student's t, refitted Student's t and normal models.
        """
        dp, t_fit = fit.skew_t["dp"], fit.student_t
        skew_t = Model(
            float(dp["nu"]),
            np.asarray(fit.skew_t["cp"]["mean"], dtype=float),
            np.asarray(dp["Sigma"], dtype=float),
            np.asarray(dp["alpha"], dtype=float),
        )
        no_shape = np.zeros(len(fit.tickers))
        t_nu = float(t_fit["nu"])
        return cls(
            skew_t=skew_t,
            student_t=replace(skew_t, shape=no_shape),
            refit_t=Model(
                t_nu if t_nu >= MIN_T_NU else skew_t.nu,
                np.asarray(t_fit["mu"], dtype=float),
                np.asarray(t_fit["Sigma"], dtype=float),
                no_shape,
            ),
            normal=Model(
                NORMAL_NU,
                fit.normal_mean,
                fit.normal_cov,
                no_shape,
                black_scholes=True,
            ),
        )


def build_portfolio(model: Model, kind: str, market: Market) -> OptionPortfolio:
    """
    Portfolio of at-the-money options, one per asset, under a model.

    The price change of asset ``n`` over ``Delta t`` has the scale matrix
    ``Sigma = Delta t S diag(sigma) C diag(sigma) S`` with ``sigma_n`` the annual scale
    ``sqrt(252) * sigma^G_n * sqrt(Sigma^Z_nn)`` and ``C`` the correlation matrix of
    ``Sigma^Z``, the skewness vector ``omega = alpha^Z / sqrt(diag(Sigma))`` and the
    location ``S Delta t r - c h`` with ``h = Sigma omega / sqrt(1 + omega' Sigma omega)``
    and ``r`` the annual mean return, so that its mean is ``S Delta t r``.

    Parameters
    ----------
    model : Model
        Distribution of the standardized residuals.
    kind : str
        ``"call"`` or ``"put"``.
    market : Market
        Prices and GARCH inputs of the day.

    Returns
    -------
    OptionPortfolio
        The portfolio.
    """
    sd = np.sqrt(np.diag(model.scale))
    corr = model.scale / np.outer(sd, sd)
    par = marginal_return_parameters(model.shape, model.scale, market.daily_vol)
    scale_matrix = build_scale_matrix(market.period, corr, par["sigma"], market.spots)
    omega = omega_from_alpha(model.shape, scale_matrix)
    h = scale_matrix @ omega / np.sqrt(1.0 + omega @ scale_matrix @ omega)
    mean_return = (market.iota + model.mean * market.daily_vol) * TRADING_DAYS_PER_YEAR
    returns = mean_return - c_nu(model.nu) * h / (market.spots * market.period)

    options = []
    for i, ticker in enumerate(market.tickers):
        pricing: PricingModel = (
            BlackScholesModel(par["sigma"][i], market.period)
            if model.black_scholes
            else SkewTConvolutionModel(
                par["sigma"][i], par["shape"][i], model.nu, market.period
            )
        )
        options.append(
            EuropeanOption(
                kind,
                market.spots[i],
                market.maturity,
                Underlying(ticker, market.spots[i]),
                market.rate,
                pricing,
            )
        )
    return OptionPortfolio(
        underlyings=[option.underlying for option in options],
        options=options,
        time_period_for_risk_measurements=market.period,
        scale_matrix=scale_matrix,
        omega=omega,
        degrees_of_freedom=model.nu,
        returns=returns,
    )
