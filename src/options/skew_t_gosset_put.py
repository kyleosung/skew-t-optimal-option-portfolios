"""
European Put option priced under the skew-t modified Gosset formula.

The "skew-t Gosset" formula replaces the symmetric truncated Student-t density
``f_r`` in the Gosset formula with the one-dimensional Azzalini skew-t density.

For a skewness parameter ``alpha_skew`` and degrees of freedom ``nu``, the
1-D Azzalini skew-t p.d.f. is::

    f(xi; alpha, nu) = 2 * t_nu(xi) * T_{nu+1}(alpha * xi * sqrt((nu+1) / (xi^2 + nu)))

where ``t_nu`` is the Student-t p.d.f. and ``T_{nu+1}`` is its CDF.

The truncation normalization constants ``p_p`` and ``p_c`` are computed
internally as the skew-t CDF values at ``x_p`` and ``x_c`` respectively,
so callers only need to supply ``x_p``, ``x_c``, and ``alpha_skew``.

See :mod:`src.options.skew_t_gosset_call` for the derivation of the
1-D skewness parameter from the multivariate AC skew-t parameters.
"""

from src.options.gosset_put import GossetPut
from src.options.option import MarketEnvironment, Underlying
from src.utils.skew_t_distribution import _skewt_cdf, _skewt_pdf


class SkewTGossetPut(GossetPut):
    """
    European Put option priced under the Gosset formula with a 1-D skew-t density.

    Identical to :class:`~src.options.gosset_put.GossetPut` except that the
    driving density ``f_r`` is the Azzalini skew-t instead of the symmetric
    Student-t.  The truncation normalization constants ``p_p`` and ``p_c`` are
    set to the skew-t CDF values at ``x_p`` and ``x_c`` so that the density
    ``f_r(xi) / (p_c - p_p)`` integrates to 1 over ``[x_p, x_c]``.
    """

    def __init__(
        self,
        strike: float,
        time_to_maturity: float,
        underlying: Underlying,
        market_env: MarketEnvironment,
        x_p: float,
        x_c: float,
        alpha_skew: float,
        degrees_of_freedom: float = 5.0,
        dividend_yield: float = 0.0,
    ) -> None:
        """
        Initialize a SkewTGossetPut option.

        Parameters
        ----------
        strike : float
            Strike price.
        time_to_maturity : float
            Time to maturity in years.
        underlying : Underlying
            The underlying asset.
        market_env : MarketEnvironment
            Market environment (risk-free rate etc.).
        x_p : float
            Lower truncation bound for the standardized variable.
        x_c : float
            Upper truncation bound for the standardized variable.
        alpha_skew : float
            1-D Azzalini skewness parameter for the underlying's return
            distribution.  Derived from the multivariate skew-t slant ``omega``
            as ``delta / sqrt(1 - delta^2)`` where
            ``delta = h_i / sqrt(Sigma_ii)`` and
            ``h = Sigma @ omega / sqrt(1 + omega^T Sigma omega)``.
        degrees_of_freedom : float, optional
            Degrees of freedom ``nu`` for the skew-t distribution.  Default 5.
        dividend_yield : float, optional
            Dividend yield of the underlying.  Default 0.
        """
        # Store skewness before calling super so that _f_r works from the start.
        self.alpha_skew = alpha_skew

        # Validate degrees_of_freedom before computing the CDF to give a clear error.
        if degrees_of_freedom <= 0:
            raise ValueError(
                f"degrees_of_freedom must be positive. Got {degrees_of_freedom}"
            )

        # Compute the correct skew-t CDF at the truncation bounds.
        p_p = _skewt_cdf(x_p, alpha_skew, degrees_of_freedom)
        p_c = _skewt_cdf(x_c, alpha_skew, degrees_of_freedom)

        super().__init__(
            strike=strike,
            time_to_maturity=time_to_maturity,
            underlying=underlying,
            market_env=market_env,
            x_p=x_p,
            x_c=x_c,
            p_p=p_p,
            p_c=p_c,
            degrees_of_freedom=degrees_of_freedom,
            dividend_yield=dividend_yield,
        )

    # ------------------------------------------------------------------
    # Override the density
    # ------------------------------------------------------------------

    def _f_r(self, xi: float) -> float:
        """1-D Azzalini skew-t p.d.f. with the configured alpha and nu."""
        return _skewt_pdf(xi, self.alpha_skew, self.degrees_of_freedom)

    # ------------------------------------------------------------------
    # Override helper methods so numerical Greeks use the skew-t density
    # ------------------------------------------------------------------

    def _compute_price_with_spot(self, spot: float) -> float:
        """Compute price with a different spot price (for delta/gamma)."""
        temp_underlying = Underlying(
            name=self.underlying.name,
            spot=spot,
            volatility=self.underlying.volatility,
        )
        temp_option = SkewTGossetPut(
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            underlying=temp_underlying,
            market_env=self.market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            alpha_skew=self.alpha_skew,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def _compute_price_with_ttm(self, time_to_maturity: float) -> float:
        """Compute price with a different time to maturity (for theta)."""
        temp_option = SkewTGossetPut(
            strike=self.strike,
            time_to_maturity=time_to_maturity,
            underlying=self.underlying,
            market_env=self.market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            alpha_skew=self.alpha_skew,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def _compute_price_with_volatility(self, volatility: float) -> float:
        """Compute option price using a different volatility value (used for numerical vega calculation)."""
        temp_underlying = Underlying(
            name=self.underlying.name,
            spot=self.underlying.spot,
            volatility=volatility,
        )
        temp_option = SkewTGossetPut(
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            underlying=temp_underlying,
            market_env=self.market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            alpha_skew=self.alpha_skew,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price

    def _compute_price_with_rate(self, risk_free_rate: float) -> float:
        """Compute price with a different risk-free rate (for numerical rho)."""
        temp_market_env = MarketEnvironment(annual_risk_free_rate=risk_free_rate)
        temp_option = SkewTGossetPut(
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            underlying=self.underlying,
            market_env=temp_market_env,
            x_p=self.x_p,
            x_c=self.x_c,
            alpha_skew=self.alpha_skew,
            degrees_of_freedom=self.degrees_of_freedom,
            dividend_yield=self.dividend_yield,
        )
        return temp_option.price
