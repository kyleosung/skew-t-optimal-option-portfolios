"""
Option-pricing models for skew-t log returns, and their Gaussian counterpart.

:class:`SkewTConvolutionModel` treats the one-period log returns under the
pricing measure as i.i.d. location-0 skew-t variables plus a drift, builds the
law of their ``d``-period sum by FFT convolution on a log-return grid, truncates
it above ``b`` and fixes the drift so that ``E[S_T] = S exp(r tau)``.
:class:`BlackScholesModel` is the Black-Scholes model.
"""

from typing import Optional

import numpy as np
from scipy import fft
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm, t

from src.pricing.base import PricingModel, check_kind
from src.settings import TRADING_DAYS_PER_YEAR

N_GRID = 2**18
HALF_WIDTH = 12.0
TRUNCATION_B = 4.0
MIN_KEPT_MASS = 0.99  # minimum share of the probability mass inside [a, b]


def skew_t_pdf(y: np.ndarray, scale: float, shape: float, nu: float) -> np.ndarray:
    """
    Density of the one-dimensional Azzalini skew-t with location 0.

    With ``z = y / scale``::

        f(y) = (2 / scale) * t_nu(z) * T_{nu+1}(shape * z * sqrt((nu + 1) / (z^2 + nu)))

    where ``t_nu`` is the Student's t pdf and ``T_{nu+1}`` the Student's t cdf.

    Parameters
    ----------
    y : np.ndarray
        Points at which to evaluate the density.
    scale : float
        Scale (> 0).
    shape : float
        Azzalini shape.
    nu : float
        Degrees of freedom.

    Returns
    -------
    np.ndarray
        Density values, same shape as ``y``.
    """
    z = np.asarray(y, dtype=float) / scale
    inner = shape * z * np.sqrt((nu + 1.0) / (z**2 + nu))
    return 2.0 / scale * t.pdf(z, df=nu) * t.cdf(inner, df=nu + 1.0)


def log_return_grid(
    n_grid: int = N_GRID, half_width: float = HALF_WIDTH
) -> tuple[np.ndarray, float]:
    """
    Uniform log-return grid with ``y = 0`` at index ``n_grid // 2``.

    Parameters
    ----------
    n_grid : int, optional
        Number of grid points, by default ``2**18``.
    half_width : float, optional
        The grid covers ``[-half_width, half_width)``, by default 12.

    Returns
    -------
    tuple[np.ndarray, float]
        Grid ``y_j = (j - n_grid / 2) * dy`` and spacing ``dy``.
    """
    dy = 2.0 * half_width / n_grid
    return (np.arange(n_grid) - n_grid // 2) * dy, dy


def check_grid(dy: float, scale: float, half_width: float, b: float) -> None:
    """
    Check that the grid resolves the density and is wide enough for the truncation.

    Parameters
    ----------
    dy : float
        Grid spacing.
    scale : float
        Smallest scale whose density is sampled on the grid.
    half_width : float
        Half-width of the grid.
    b : float
        Upper truncation point of the log return.

    Raises
    ------
    ValueError
        If ``dy > scale / 10``, or ``b`` is finite and ``half_width < 2 * b``.
    """
    if dy > scale / 10.0:
        raise ValueError(
            f"Grid spacing {dy:.3g} exceeds scale/10 = {scale / 10.0:.3g}; "
            "increase n_grid."
        )
    if np.isfinite(b) and half_width < 2.0 * b:
        raise ValueError(f"half_width={half_width} must be at least 2*b = {2.0 * b}.")


def one_period_spectrum(f1: np.ndarray, dy: float) -> np.ndarray:
    """
    Real FFT of the one-period probabilities with ``y = 0`` at index 0.

    Parameters
    ----------
    f1 : np.ndarray
        One-period density sampled on the grid.
    dy : float
        Grid spacing.

    Returns
    -------
    np.ndarray
        ``rfft`` of the probability masses, normalized to unit total mass.
    """
    p = fft.ifftshift(f1) * dy
    p /= p.sum()
    return fft.rfft(p)


def d_fold_density(
    f1: np.ndarray,
    dy: float,
    d: int,
    spectrum: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Density of the sum of ``d`` i.i.d. variables with density ``f1`` on the grid.

    Parameters
    ----------
    f1 : np.ndarray
        One-period density sampled on the grid with ``y = 0`` at the centre.
    dy : float
        Grid spacing.
    d : int
        Number of periods.
    spectrum : np.ndarray, optional
        Precomputed :func:`one_period_spectrum`; each maturity then costs one
        inverse FFT.

    Returns
    -------
    np.ndarray
        Density of the ``d``-fold sum, with negative round-off clipped to 0.
    """
    if spectrum is None:
        spectrum = one_period_spectrum(f1, dy)
    g = fft.irfft(spectrum**d, n=f1.size)
    return np.clip(fft.fftshift(g), 0.0, None) / dy


class TruncatedGridLaw:
    """
    Truncated law of the centered log return on a grid, and the prices it implies.

    Under the pricing measure ``S_T = S * exp(r * tau) * exp(Y) / Lambda`` where
    ``Y`` has the truncated density ``g`` and ``Lambda = E[exp(Y)]``, so that
    ``E[S_T] = S * exp(r * tau)``.  The location of the log return is
    ``d * mu_q = r * tau - log(Lambda)``; it is not stored because it depends on
    the rate, and the truncation window ``[a, b]`` is measured from it.

    Parameters
    ----------
    y : np.ndarray
        Uniform log-return grid.
    density : np.ndarray
        Density of the centered log return on the grid.
    a : float, optional
        Lower truncation point, by default ``-inf``.
    b : float, optional
        Upper truncation point, by default ``inf``.

    Attributes
    ----------
    y : np.ndarray
        Log-return grid.
    g : np.ndarray
        Truncated, renormalized density on the grid.
    lam : float
        ``Lambda = E[exp(min(Y, b))]``.
    pi1 : np.ndarray
        ``int_y^b exp(u) g(u) du / Lambda`` on the grid, equal to 1 at the bottom.
    pi2 : np.ndarray
        ``int_y^b g(u) du`` on the grid, equal to 1 at the bottom.
    lo1 : np.ndarray
        ``int_{-inf}^y exp(u) g(u) du / Lambda`` on the grid, equal to 0 at the bottom.
    lo2 : np.ndarray
        ``int_{-inf}^y g(u) du`` on the grid, equal to 0 at the bottom.
    """

    def __init__(
        self,
        y: np.ndarray,
        density: np.ndarray,
        a: float = -np.inf,
        b: float = np.inf,
    ) -> None:
        dy = float(y[1] - y[0])
        g = np.clip(density, 0.0, None) * ((y >= a) & (y <= b))
        g /= g.sum() * dy
        weighted = np.exp(np.minimum(y, b)) * g
        self.y = y
        self.g = g
        self.lam = float(weighted.sum() * dy)
        self.pi1 = self._upper_tail(weighted, dy)
        self.pi2 = self._upper_tail(g, dy)
        self.lo1 = cumulative_trapezoid(weighted, dx=dy, initial=0.0) / self.lam
        self.lo2 = cumulative_trapezoid(g, dx=dy, initial=0.0)

    @staticmethod
    def _upper_tail(f: np.ndarray, dy: float) -> np.ndarray:
        """
        Integral of ``f`` from each grid point to the top, equal to 1 at the bottom.

        Parameters
        ----------
        f : np.ndarray
            Integrand on the grid.
        dy : float
            Grid spacing.

        Returns
        -------
        np.ndarray
            Cumulative trapezoid from the top, normalized by its value at the
            bottom of the grid.
        """
        tail = cumulative_trapezoid(f[::-1], dx=dy, initial=0.0)[::-1]
        return tail / tail[0]

    def price_delta_gamma(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> tuple[float, float, float]:
        """
        Option price, Delta and Gamma.

        The call and the put are the integrals of their payoffs against the law, over
        the upper and the lower tail of the log return respectively, so the put is not
        priced by put-call parity; parity holds because ``E[S_T] = S*exp(r*tau)`` and
        validates the integrals.

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
        tuple[float, float, float]
            ``(price, delta, gamma)`` with ``gamma = K*exp(-r*tau)*g(y*) / S^2``, where
            ``y* = log(K*Lambda*exp(-r*tau) / S)``. For a call
            ``price = S*pi1 - K*exp(-r*tau)*pi2`` and ``delta = pi1``; for a put
            ``price = K*exp(-r*tau)*lo2 - S*lo1`` and ``delta = -lo1``, the tails
            being evaluated at ``y*``.
        """
        discounted_strike = strike * np.exp(-rate * tau)
        y_star = np.log(discounted_strike * self.lam / spot)
        dens = float(np.interp(y_star, self.y, self.g, left=0.0, right=0.0))
        gamma = discounted_strike * dens / spot**2
        if kind == "call":
            pi1 = float(np.interp(y_star, self.y, self.pi1, left=1.0, right=0.0))
            pi2 = float(np.interp(y_star, self.y, self.pi2, left=1.0, right=0.0))
            return spot * pi1 - discounted_strike * pi2, pi1, gamma
        lo1 = float(np.interp(y_star, self.y, self.lo1, left=0.0, right=1.0))
        lo2 = float(np.interp(y_star, self.y, self.lo2, left=0.0, right=1.0))
        return discounted_strike * lo2 - spot * lo1, -lo1, gamma


class SkewTConvolutionModel(PricingModel):
    """
    I.i.d. skew-t one-period log returns summed by FFT convolution.

    The centered ``d``-period log return ``z = L - d * mu_q`` has density
    ``f^{*d}``, truncated to ``a <= z <= b`` (the window is measured from the
    location, so it moves with it) and renormalized.  ``Lambda = E[exp(z)]`` sets
    the location through ``d * mu_q = r * tau - log(Lambda)``, so that
    ``E[S_T] = S * exp(r * tau)``.  ``tau`` must be a multiple of ``period``.
    Calls and puts are the integrals of their payoffs against this law; put-call
    parity, ``call - put = spot - K*exp(-r*tau)`` and ``delta_call - delta_put = 1``,
    holds because of the choice of the location and is not used for pricing.

    Parameters
    ----------
    sigma : float
        Annualized **scale** parameter (not a volatility); the one-period scale
        is ``sigma * sqrt(period)``.
    shape : float
        Azzalini shape of the one-period return.
    nu : float
        Degrees of freedom.
    period : float, optional
        Length of one period in years, by default ``1/252``.
    a : float, optional
        Lower truncation point (the paper's ell_-), measured from the location of the
        log return, by default ``-inf``.
    b : float, optional
        Upper truncation point (the paper's ell_+), measured from the location of the
        log return, by default 4.
    n_grid : int, optional
        Number of grid points, by default ``2**18``.
    half_width : float, optional
        Half-width of the grid, by default 12.

    Raises
    ------
    ValueError
        If ``a >= b``, or the grid does not resolve the one-period density.
    """

    def __init__(
        self,
        sigma: float,
        shape: float,
        nu: float,
        period: float = 1 / TRADING_DAYS_PER_YEAR,
        a: float = -np.inf,
        b: float = TRUNCATION_B,
        n_grid: int = N_GRID,
        half_width: float = HALF_WIDTH,
    ) -> None:
        super().__init__(period)
        if not a < b:
            raise ValueError(f"Truncation points must satisfy a < b; got {a}, {b}.")
        self.sigma = float(sigma)
        self.shape = float(shape)
        self.nu = float(nu)
        self.a = float(a)
        self.b = float(b)
        self.half_width = float(half_width)
        self.y, self.dy = log_return_grid(n_grid, half_width)
        scale_1 = self.sigma * np.sqrt(self.period)
        check_grid(self.dy, scale_1, self.half_width, self.b)
        self._f1 = skew_t_pdf(self.y, scale_1, self.shape, self.nu)
        self._spectrum = one_period_spectrum(self._f1, self.dy)
        self._laws: dict[float, TruncatedGridLaw] = {}

    def _build_law(self, tau: float) -> TruncatedGridLaw:
        """
        Law of the centered ``d``-period log return.

        Parameters
        ----------
        tau : float
            Time to maturity in years, a multiple of ``period``.

        Returns
        -------
        TruncatedGridLaw
            Truncated law of the sum.

        Raises
        ------
        ValueError
            If the truncation window holds less than ``MIN_KEPT_MASS`` of the
            probability mass of the sum.
        """
        g = d_fold_density(self._f1, self.dy, self.n_periods(tau), self._spectrum)
        inside = (self.y >= self.a) & (self.y <= self.b)
        if g[inside].sum() < MIN_KEPT_MASS * g.sum():
            raise ValueError(
                f"The window [{self.a}, {self.b}] holds less than "
                f"{MIN_KEPT_MASS:.0%} of the law of the sum for tau={tau}; "
                "increase b or half_width."
            )
        return TruncatedGridLaw(self.y, g, self.a, self.b)

    def law(self, tau: float) -> TruncatedGridLaw:
        """
        Law of the maturity ``tau``, built once and cached.

        The law depends on ``tau`` only, not on the spot, the strike or the rate.

        Parameters
        ----------
        tau : float
            Time to maturity in years.

        Returns
        -------
        TruncatedGridLaw
            The law, with a ``price_delta_gamma(spot, strike, tau, rate, kind)`` method.
        """
        key = round(tau, 12)
        if key not in self._laws:
            self._laws[key] = self._build_law(tau)
        return self._laws[key]

    def price(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Option price, the integral of the payoff against the law of the maturity.

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
        check_kind(kind)
        return self.law(tau).price_delta_gamma(spot, strike, tau, rate, kind)[0]

    def delta(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Closed-form Delta: ``pi1`` for a call and ``-lo1`` for a put.

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
        check_kind(kind)
        return self.law(tau).price_delta_gamma(spot, strike, tau, rate, kind)[1]

    def gamma(
        self,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        kind: str = "call",
    ) -> float:
        """
        Closed-form Gamma, the same for calls and puts.

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
        check_kind(kind)
        return self.law(tau).price_delta_gamma(spot, strike, tau, rate, kind)[2]


class BlackScholesModel(PricingModel):
    """
    Black-Scholes prices of European calls and puts.

    The log return over the maturity ``tau`` is normal with variance
    ``sigma^2 * tau`` and the location that makes ``E[S_T] = S * exp(r * tau)``, the
    Gaussian limit of :class:`SkewTConvolutionModel` (``nu = inf``, no skewness, no
    truncation).  Delta and Gamma are closed-form and Theta is the one-period forward
    difference of :class:`~src.pricing.base.PricingModel`.

    Parameters
    ----------
    sigma : float
        Annualized volatility of the log return.
    period : float, optional
        Length of one period in years, by default ``1/252``.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive.
    """

    def __init__(self, sigma: float, period: float = 1 / TRADING_DAYS_PER_YEAR) -> None:
        super().__init__(period)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}.")
        self.sigma = float(sigma)

    def _d1(self, spot: float, strike: float, tau: float, rate: float) -> float:
        """
        Black-Scholes ``d1``.

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

        Returns
        -------
        float
            ``(log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))``.
        """
        return (np.log(spot / strike) + (rate + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )

    def price(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Black-Scholes price.

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
        check_kind(kind)
        d1 = self._d1(spot, strike, tau, rate)
        d2 = d1 - self.sigma * np.sqrt(tau)
        discounted_strike = strike * np.exp(-rate * tau)
        if kind == "call":
            return float(spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
        return float(discounted_strike * norm.cdf(-d2) - spot * norm.cdf(-d1))

    def delta(
        self, spot: float, strike: float, tau: float, rate: float, kind: str
    ) -> float:
        """
        Closed-form Delta: ``N(d1)`` for a call and ``N(d1) - 1`` for a put.

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
        check_kind(kind)
        delta_call = float(norm.cdf(self._d1(spot, strike, tau, rate)))
        return delta_call if kind == "call" else delta_call - 1.0

    def gamma(
        self,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        kind: str = "call",
    ) -> float:
        """
        Closed-form Gamma, the same for calls and puts.

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
        check_kind(kind)
        d1 = self._d1(spot, strike, tau, rate)
        return float(norm.pdf(d1) / (spot * self.sigma * np.sqrt(tau)))
