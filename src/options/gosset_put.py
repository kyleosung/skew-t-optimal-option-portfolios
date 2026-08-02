"""
European Put option priced under the Gosset (truncated Student-t) formula.
"""

from functools import cached_property

import numpy as np
from scipy.integrate import quad
from scipy.stats import t

from src.options.option import MarketEnvironment, Option, Underlying


class GossetPut(Option):
    """European Put option priced under the Gosset formula with truncated density."""

    def __init__(
        self,
        strike: float,
        time_to_maturity: float,
        underlying: Underlying,
        market_env: MarketEnvironment,
        x_p: float,
        x_c: float,
        p_p: float,
        p_c: float,
        degrees_of_freedom: float = 5.0,
        dividend_yield: float = 0.0,
    ) -> None:
        """
        Initialize a Gosset Put option.

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
        x_p : float
            Lower truncation bound for the standardized variable.
        x_c : float
            Upper truncation bound for the standardized variable.
        p_p : float
            CDF value at x_p for the Student-t distribution.
        p_c : float
            CDF value at x_c for the Student-t distribution.
        degrees_of_freedom : float, optional
            Degrees of freedom for the Student-t distribution. Default is 5.0.
        dividend_yield : float, optional
            Dividend yield of the underlying. Default is 0.0.

        Raises
        ------
        ValueError
            If x_p >= x_c, p_p or p_c are not in [0, 1], p_p >= p_c,
            or degrees_of_freedom <= 0.
        """
        super().__init__(
            strike, time_to_maturity, underlying, market_env, dividend_yield
        )

        # Validate input parameters
        if x_p >= x_c:
            raise ValueError(f"x_p must be less than x_c. Got x_p={x_p}, x_c={x_c}")
        if not (0 <= p_p <= 1):
            raise ValueError(f"p_p must be in [0, 1]. Got p_p={p_p}")
        if not (0 <= p_c <= 1):
            raise ValueError(f"p_c must be in [0, 1]. Got p_c={p_c}")
        if p_p >= p_c:
            raise ValueError(f"p_p must be less than p_c. Got p_p={p_p}, p_c={p_c}")
        if degrees_of_freedom <= 0:
            raise ValueError(
                f"degrees_of_freedom must be positive. Got {degrees_of_freedom}"
            )

        self.x_p = x_p
        self.x_c = x_c
        self.p_p = p_p
        self.p_c = p_c
        self.degrees_of_freedom = degrees_of_freedom

    def payoff(self, S: float | np.ndarray) -> float | np.ndarray:
        """Compute put payoff max(K - S, 0)."""
        return np.maximum(self.strike - S, 0.0)

    def _f_r(self, xi: float) -> float:
        """Student-t p.d.f. with the specified degrees of freedom."""
        return t.pdf(xi, df=self.degrees_of_freedom)  # type: ignore

    @cached_property
    def _Z_trunc(self) -> float:
        """
        Compute Z^trunc using numerical integration.

        Z^trunc = integral from x_p to x_c of exp(sigma_T * xi) * f_r(xi) / (p_c - p_p) dxi

        Raises
        ------
        ValueError
            If the computed Z^trunc is too small (near zero).
        """
        sigma = self.underlying.volatility
        T = self.time_to_maturity
        sigma_T = sigma * np.sqrt(T)

        def integrand(xi: float) -> float:
            return np.exp(sigma_T * xi) * self._f_r(xi) / (self.p_c - self.p_p)

        result, _ = quad(integrand, self.x_p, self.x_c)

        # Check for near-zero result which would cause division issues
        if abs(result) < 1e-10:
            raise ValueError(
                f"Z^trunc computed to near-zero value ({result}). "
                "This indicates invalid truncation parameters."
            )

        return result

    @cached_property
    def _A_T_trunc(self) -> float:
        """
        Compute A_T^trunc, the average price of the stock at time T.

        A_T^trunc = S_0 * exp(r*T) / Z^trunc
        """
        S_0 = self.underlying.spot
        r = self.market_env.annual_risk_free_rate
        T = self.time_to_maturity
        return S_0 * np.exp(r * T) / self._Z_trunc

    @cached_property
    def price(self) -> float:
        """
        Compute the Gosset formula price for a European put option.

        P_T^trunc = integral from x_p to log(K_T / A_T^trunc) / sigma_T
                    of (K_T - A_T^trunc * exp(sigma_T * xi)) * f_r(xi) / (p_c - p_p) dxi
        """
        K = self.strike
        sigma = self.underlying.volatility
        T = self.time_to_maturity
        sigma_T = sigma * np.sqrt(T)
        A_T = self._A_T_trunc

        upper_limit = np.log(K / A_T) / sigma_T

        # Ensure upper_limit is within bounds
        if upper_limit >= self.x_c:
            # If strike is too high, the option is deeply in-the-money
            # We integrate to x_c instead
            upper_limit = self.x_c

        def integrand(xi: float) -> float:
            return (
                (K - A_T * np.exp(sigma_T * xi)) * self._f_r(xi) / (self.p_c - self.p_p)
            )

        result, _ = quad(integrand, self.x_p, upper_limit)
        return max(0.0, result)

    def _compute_price_with_spot(self, spot: float) -> float:
        """
        Helper method to compute price with a different spot price.

        Parameters
        ----------
        spot : float
            Temporary spot price to use for calculation.

        Returns
        -------
        float
            Option price with the temporary spot value.
        """
        # Create temporary copies to avoid mutating shared state
        temp_underlying = Underlying(
            name=self.underlying.name,
            spot=spot,
            volatility=self.underlying.volatility,
        )
        temp_option = GossetPut(
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            underlying=temp_underlying,
            market_env=self.market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            p_p=self.p_p,
            p_c=self.p_c,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def _compute_price_with_volatility(self, volatility: float) -> float:
        """
        Helper method to compute price with a different volatility.

        Parameters
        ----------
        volatility : float
            Temporary volatility to use for calculation.

        Returns
        -------
        float
            Option price with the temporary volatility value.
        """
        temp_underlying = Underlying(
            name=self.underlying.name,
            spot=self.underlying.spot,
            volatility=volatility,
        )
        temp_option = GossetPut(
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            underlying=temp_underlying,
            market_env=self.market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            p_p=self.p_p,
            p_c=self.p_c,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def _compute_price_with_ttm(self, time_to_maturity: float) -> float:
        """
        Helper method to compute price with a different time to maturity.

        Parameters
        ----------
        time_to_maturity : float
            Temporary time to maturity to use for calculation.

        Returns
        -------
        float
            Option price with the temporary time to maturity value.
        """
        temp_option = GossetPut(
            strike=self.strike,
            time_to_maturity=time_to_maturity,
            underlying=self.underlying,
            market_env=self.market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            p_p=self.p_p,
            p_c=self.p_c,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def _compute_price_with_rate(self, risk_free_rate: float) -> float:
        """
        Helper method to compute price with a different risk-free rate.

        Parameters
        ----------
        risk_free_rate : float
            Temporary risk-free rate to use for calculation.

        Returns
        -------
        float
            Option price with the temporary risk-free rate value.
        """
        temp_market_env = MarketEnvironment(annual_risk_free_rate=risk_free_rate)
        temp_option = GossetPut(
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            underlying=self.underlying,
            market_env=temp_market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            p_p=self.p_p,
            p_c=self.p_c,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def delta_numerical(self, epsilon: float = 1e-5) -> float:
        """
        Compute Delta numerically using finite differences.

        Parameters
        ----------
        epsilon : float, optional
            Step size for finite difference. Default is 1e-5.

        Returns
        -------
        float
            Numerical approximation of Delta.
        """
        spot = self.underlying.spot
        price_up = self._compute_price_with_spot(spot + epsilon)
        price_down = self._compute_price_with_spot(spot - epsilon)
        return (price_up - price_down) / (2 * epsilon)

    def gamma_numerical(self, epsilon: float = 1e-5) -> float:
        """
        Compute Gamma numerically using finite differences.

        Parameters
        ----------
        epsilon : float, optional
            Step size for finite difference. Default is 1e-5.

        Returns
        -------
        float
            Numerical approximation of Gamma.
        """
        spot = self.underlying.spot
        price_up = self._compute_price_with_spot(spot + epsilon)
        price_mid = self.price
        price_down = self._compute_price_with_spot(spot - epsilon)
        return (price_up - 2 * price_mid + price_down) / (epsilon**2)

    def vega_numerical(self, epsilon: float = 1e-5) -> float:
        """
        Compute Vega numerically using finite differences.

        Parameters
        ----------
        epsilon : float, optional
            Step size for finite difference. Default is 1e-5.

        Returns
        -------
        float
            Numerical approximation of Vega.
        """
        vol = self.underlying.volatility
        price_up = self._compute_price_with_volatility(vol + epsilon)
        price_down = self._compute_price_with_volatility(vol - epsilon)
        return (price_up - price_down) / (2 * epsilon)

    def theta_numerical(self, epsilon: float = 1e-5) -> float:
        """
        Compute Theta numerically using finite differences.

        Parameters
        ----------
        epsilon : float, optional
            Step size for finite difference. Default is 1e-5.

        Returns
        -------
        float
            Numerical approximation of Theta.
        """
        ttm = self.time_to_maturity
        price_future = self._compute_price_with_ttm(ttm - epsilon)
        price_now = self.price
        return (price_future - price_now) / epsilon

    def rho_numerical(self, epsilon: float = 1e-5) -> float:
        """
        Compute Rho numerically using finite differences.

        Parameters
        ----------
        epsilon : float, optional
            Step size for finite difference. Default is 1e-5.

        Returns
        -------
        float
            Numerical approximation of Rho.
        """
        rate = self.market_env.annual_risk_free_rate
        price_up = self._compute_price_with_rate(rate + epsilon)
        price_down = self._compute_price_with_rate(rate - epsilon)
        return (price_up - price_down) / (2 * epsilon)

    @property
    def delta(self) -> float:
        """Delta is not available analytically for Gosset put options."""
        return self.delta_numerical()

    @property
    def gamma(self) -> float:
        """Gamma is not available analytically for Gosset put options."""
        return self.gamma_numerical()

    @property
    def vega(self) -> float:
        """Vega is not available analytically for Gosset put options."""
        raise NotImplementedError(
            "Analytical vega not available for Gosset put options. Use vega_numerical() instead."
        )

    @property
    def theta(self) -> float:
        """Theta is not available analytically for Gosset put options."""
        return self.theta_numerical()

    @property
    def rho(self) -> float:
        """Rho is not available analytically for Gosset put options."""
        raise NotImplementedError(
            "Analytical rho not available for Gosset put options. Use rho_numerical() instead."
        )
