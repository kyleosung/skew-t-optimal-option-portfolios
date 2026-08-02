"""
Abstract base class for options and related market environment and underlying asset classes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np


@dataclass
class MarketEnvironment:
    """
    Global market parameters shared across instruments.

    Attributes
    ----------
    annual_risk_free_rate : float
        Continuously compounded risk-free rate (annualized).
    """

    annual_risk_free_rate: float = 0.05

    def update(self, **kwargs: Any) -> None:
        """Update one or more market parameters dynamically."""
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class Underlying:
    """
    Represents a tradable underlying asset (e.g., stock, index, FX).

    Attributes
    ----------
    name : str
        Identifier or ticker of the asset.
    spot : float
        Current spot price of the underlying.
    volatility : float
        Annualized volatility (standard deviation of log-returns).
    """

    name: str
    spot: float
    volatility: float

    def update(self, **kwargs: Any) -> None:
        """Update one or more attributes (e.g., spot or volatility)."""
        for k, v in kwargs.items():
            setattr(self, k, v)


class Option(ABC):
    """
    Abstract base class for all options.

    Parameters
    ----------
    strike : float
        Strike price of the option.
    time_to_maturity : float
        Time to maturity in years.
    underlying : Underlying
        The underlying asset object.
    market_env : MarketEnvironment
        The global market environment (e.g., risk-free rate).
    """

    def __init__(
        self,
        strike: float,
        time_to_maturity: float,
        underlying: Underlying,
        market_env: MarketEnvironment,
        dividend_yield: float = 0.0,
    ) -> None:
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.underlying = underlying
        self.market_env = market_env
        self.dividend_yield = dividend_yield

    @abstractmethod
    def payoff(self, S: float | np.ndarray) -> float:
        """Option payoff as a function of underlying price S."""
        pass

    @abstractmethod
    def price(self) -> float:
        """Return the Black-Scholes price of the option."""
        pass

    @abstractmethod
    def delta(self) -> float:
        """Return the option Delta."""
        pass

    @abstractmethod
    def gamma(self) -> float:
        """Return the option Gamma."""
        pass

    @cached_property
    def d1(self) -> float:
        """Intermediate variable d1 in Black-Scholes formula."""
        S = self.underlying.spot
        K = self.strike
        T = self.time_to_maturity
        r = self.market_env.annual_risk_free_rate
        q = self.dividend_yield
        sigma = self.underlying.volatility

        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @cached_property
    def d2(self) -> float:
        """Intermediate variable d2 in Black-Scholes formula."""
        return self.d1 - self.underlying.volatility * np.sqrt(self.time_to_maturity)

    def _reset_cache(self) -> None:
        """
        Reset cached properties. Should be called when inputs change
        (e.g., after updating spot, volatility, or interest rate).
        """
        for attr in ("d1", "d2"):
            if attr in self.__dict__:
                del self.__dict__[attr]  # type: ignore
