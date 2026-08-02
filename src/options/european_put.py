"""
European Put option priced under the Black-Scholes model.
"""

from functools import cached_property

import numpy as np
from scipy.stats import norm

from src.options.option import MarketEnvironment, Option, Underlying


class EuropeanPut(Option):
    """European Put option priced under the Black-Scholes model."""

    def __init__(
        self,
        strike: float,
        time_to_maturity: float,
        underlying: Underlying,
        market_env: MarketEnvironment,
        dividend_yield: float = 0.0,
    ) -> None:
        super().__init__(
            strike, time_to_maturity, underlying, market_env, dividend_yield
        )

    def payoff(self, S: float | np.ndarray) -> float | np.ndarray:
        """Compute put payoff max(K - S, 0)."""
        return np.maximum(self.strike - S, 0.0)

    @cached_property
    def price(self) -> float:
        """Compute and cache the Black-Scholes price."""
        S = self.underlying.spot
        K = self.strike
        r = self.market_env.annual_risk_free_rate
        tau = self.time_to_maturity
        q = self.dividend_yield
        price = K * np.exp(-r * tau) * norm.cdf(-self.d2) - S * np.exp(
            -q * tau
        ) * norm.cdf(-self.d1)
        return price.item()

    @cached_property
    def delta(self) -> float:
        """Black-Scholes Delta."""
        q = self.dividend_yield
        tau = self.time_to_maturity
        return (np.exp(-q * tau) * (norm.cdf(self.d1) - 1.0)).item()

    @cached_property
    def gamma(self) -> float:
        """Black-Scholes Gamma."""
        S = self.underlying.spot
        sigma = self.underlying.volatility
        tau = self.time_to_maturity
        q = self.dividend_yield
        return (
            np.exp(-q * tau) * norm.pdf(self.d1) / (S * sigma * np.sqrt(tau))
        ).item()

    @cached_property
    def vega(self) -> float:
        """Black-Scholes Vega (per 1.0 volatility, not 1%)."""
        S = self.underlying.spot
        tau = self.time_to_maturity
        q = self.dividend_yield
        return (S * np.exp(-q * tau) * norm.pdf(self.d1) * np.sqrt(tau)).item()

    @cached_property
    def theta(self) -> float:
        """Black-Scholes Theta (per year)."""
        S = self.underlying.spot
        K = self.strike
        r = self.market_env.annual_risk_free_rate
        q = self.dividend_yield
        sigma = self.underlying.volatility
        tau = self.time_to_maturity
        d1 = self.d1
        d2 = self.d2

        term1 = -np.exp(-q * tau) * (S * sigma * norm.pdf(d1)) / (2 * np.sqrt(tau))
        term2 = r * K * np.exp(-r * tau) * norm.cdf(-d2)
        term3 = q * S * np.exp(-q * tau) * norm.cdf(-d1)

        theta = term1 + term2 - term3
        return theta.item()
