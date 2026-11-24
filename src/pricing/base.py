"""
Common interface of the option-pricing models.

Every model prices European calls and puts on one security and exposes the
quantities read by :class:`~src.portfolio.option_portfolio.OptionPortfolio`
(price, Delta, Gamma and Theta) through
:class:`~src.options.european_option.EuropeanOption`.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.settings import TRADING_DAYS_PER_YEAR

KINDS = ("call", "put")
TIME_TOL = 1e-9  # tolerance when comparing times (in years)


def check_kind(kind: str) -> None:
    """
    Validate the option kind.

    Parameters
    ----------
    kind : str
        Option kind.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"call"`` or ``"put"``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}.")


class PricingModel(ABC):
    """
    Abstract pricing model for European calls and puts on one security.

    Subclasses implement :meth:`price`, :meth:`delta` and :meth:`gamma`; Theta is the
    one-period forward difference for every model.

    Parameters
    ----------
    period : float, optional
        Length of one period in years (``Delta t``), by default ``1/252``.

    Attributes
    ----------
    period : float
        Length of one period in years.
    """

    def __init__(self, period: float = 1 / TRADING_DAYS_PER_YEAR) -> None:
        if period <= 0:
            raise ValueError(f"period must be positive; got {period}.")
        self.period = float(period)

    @abstractmethod
    def price(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Price of a European option.

        Parameters
        ----------
        spot : float
            Spot price.
        strike : float
            Strike price.
        tau : float
            Time to maturity in years.
        rate : float
            Continuously compounded risk-free rate.
        kind : str
            ``"call"`` or ``"put"``.

        Returns
        -------
        float
            Option price.
        """

    @abstractmethod
    def delta(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Delta of a European option.

        Parameters
        ----------
        spot : float
            Spot price.
        strike : float
            Strike price.
        tau : float
            Time to maturity in years.
        rate : float
            Continuously compounded risk-free rate.
        kind : str
            ``"call"`` or ``"put"``.

        Returns
        -------
        float
            ``dV/dS``.
        """

    @abstractmethod
    def gamma(
        self,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        kind: str = "call",
    ) -> float:
        """
        Gamma of a European option, the same for calls and puts.

        Parameters
        ----------
        spot : float
            Spot price.
        strike : float
            Strike price.
        tau : float
            Time to maturity in years.
        rate : float
            Continuously compounded risk-free rate.
        kind : str, optional
            ``"call"`` or ``"put"``, by default ``"call"``.

        Returns
        -------
        float
            ``d^2 V / dS^2``.
        """

    def theta(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Theta as the one-period forward difference, per year.

        ``(V(spot, strike, tau - period) - V(spot, strike, tau)) / period``.

        Parameters
        ----------
        spot : float
            Spot price.
        strike : float
            Strike price.
        tau : float
            Time to maturity in years.
        rate : float
            Continuously compounded risk-free rate.
        kind : str
            ``"call"`` or ``"put"``.

        Returns
        -------
        float
            ``dV/dt``.
        """
        later = self.price(spot, strike, tau - self.period, rate, kind)
        return (later - self.price(spot, strike, tau, rate, kind)) / self.period

    def n_periods(self, tau: float) -> int:
        """
        Number of whole periods to maturity.

        Parameters
        ----------
        tau : float
            Time to maturity in years.

        Returns
        -------
        int
            ``d = round(tau / period)``.

        Raises
        ------
        ValueError
            If ``d < 1`` or ``tau`` is not a multiple of ``period``.
        """
        d = int(np.round(tau / self.period))
        if d < 1 or abs(d * self.period - tau) >= TIME_TOL:
            raise ValueError(
                f"tau={tau} must be a positive multiple of period={self.period}."
            )
        return d
