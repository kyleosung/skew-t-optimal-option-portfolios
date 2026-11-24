"""
European option priced by a swappable pricing model.
"""

from dataclasses import dataclass
from functools import cached_property

from src.pricing.base import PricingModel, check_kind


@dataclass
class Underlying:
    """
    Underlying asset of an option.

    Attributes
    ----------
    name : str
        Ticker of the asset.
    spot : float
        Spot price.
    """

    name: str
    spot: float


class EuropeanOption:
    """
    European call or put priced by a :class:`~src.pricing.base.PricingModel`.

    Exposes ``price``, ``delta``, ``gamma``, ``theta`` and ``underlying``, the
    attributes read by :class:`~src.portfolio.option_portfolio.OptionPortfolio`.

    Parameters
    ----------
    kind : str
        ``"call"`` or ``"put"``.
    strike : float
        Strike price.
    time_to_maturity : float
        Time to maturity in years.
    underlying : Underlying
        The underlying asset; its ``spot`` is the pricing spot.
    rate : float
        Continuously compounded risk-free rate.
    model : PricingModel
        The pricing model.
    """

    def __init__(
        self,
        kind: str,
        strike: float,
        time_to_maturity: float,
        underlying: Underlying,
        rate: float,
        model: PricingModel,
    ) -> None:
        check_kind(kind)
        self.kind = kind
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.underlying = underlying
        self.rate = rate
        self.model = model

    def _inputs(self) -> tuple[float, float, float, float]:
        """
        Pricing inputs ``(spot, strike, tau, rate)``.

        Returns
        -------
        tuple[float, float, float, float]
            Spot, strike, time to maturity and risk-free rate.
        """
        return (
            self.underlying.spot,
            self.strike,
            self.time_to_maturity,
            self.rate,
        )

    @cached_property
    def price(self) -> float:
        """
        Option price under the model.

        Returns
        -------
        float
            Option price.
        """
        return self.model.price(*self._inputs(), self.kind)

    @cached_property
    def delta(self) -> float:
        """
        Option Delta under the model.

        Returns
        -------
        float
            ``dV/dS``.
        """
        return self.model.delta(*self._inputs(), self.kind)

    @cached_property
    def gamma(self) -> float:
        """
        Option Gamma under the model.

        Returns
        -------
        float
            ``d^2 V / dS^2``.
        """
        return self.model.gamma(*self._inputs(), self.kind)

    @cached_property
    def theta(self) -> float:
        """
        Option Theta under the model, per year.

        Returns
        -------
        float
            ``dV/dt``.
        """
        return self.model.theta(*self._inputs(), self.kind)
