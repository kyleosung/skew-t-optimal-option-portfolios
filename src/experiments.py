"""
The experiments of Section 5 and of Appendix B, for a set of assets.

Each function takes the :class:`~src.estimation.Fit` of the estimation sample of the
assets, and returns the numbers of a table or a figure of the paper.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from src.estimation import Fit
from src.models import Market, Model, Models, build_portfolio
from src.portfolio.option_portfolio import OptionPortfolio
from src.portfolio.utils import (
    number_of_shares_to_weights,
    weights_to_number_of_shares,
)
from src.settings import (
    BATCH_SIZE,
    LONG_TIME_TO_MATURITY,
    N_BATCHES,
    N_SEEDS,
    TIME_TO_MATURITY,
)
from src.utils.monte_carlo import (
    cfvar3_estimate,
    monte_carlo_price,
    option_gains,
    repricing_grids,
    sample_skew_t,
    simulate_log_returns,
)

KINDS = ("call", "put")

# Labels of the models of the out-of-sample test, in the order of the tables
MODEL_LABELS = {
    "skew_t": r"Skew-$t$",
    "refit_t": r"Student's $t$",
    "normal": "Normal",
}


def optimal_shares(
    portfolio: OptionPortfolio, objective: str, tail_risk: float
) -> np.ndarray:
    """
    Numbers of contracts of the optimal portfolio of an objective, for a budget of 1.

    Parameters
    ----------
    portfolio : OptionPortfolio
        Portfolio of options.
    objective : str
        ``"var"`` (variance), ``"cf2"`` (CFVaR2) or ``"cf3"`` (CFVaR3).
    tail_risk : float
        Tail risk ``alpha`` of the CFVaR objectives.

    Returns
    -------
    np.ndarray
        Optimal numbers of contracts, shape ``(M, 1)``: the closed-form solution for the
        variance and CFVaR2, and the numerical minimum of CFVaR3 started from the
        CFVaR2 weights.
    """
    if objective == "var":
        return portfolio.get_optimal_variance_weights_lagrange()[0]
    if objective == "cf2":
        return portfolio.get_optimal_CFVaR2_weights_lagrange(tail_risk)[0]
    return portfolio.get_optimal_CFVaR3_weights_numeric(tail_risk)[0]


def risk_comparison(
    fit: Fit, kind: str, tail_risk: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Optimal weights of Section 5.1 and CFVaR3 of the optimal portfolios of Section 5.2.

    Parameters
    ----------
    fit : Fit
        Fit of the estimation sample.
    kind : str
        ``"call"`` or ``"put"``.
    tail_risk : float
        Tail risk ``alpha``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        Weights (fractions of the budget) of the optimal portfolios, with the columns
        ``"Skew-t var"``, ``"Skew-t cf2"``, ``"Skew-t cf3"``, ``"Student t var"``,
        ``"Student t cf2"`` and ``"Student t cf3"``, indexed by the assets; and the
        CFVaR3 under the skew-t model of the variance, CFVaR2 and CFVaR3 optimal skew-t
        portfolios (``"var"``, ``"cf2"``, ``"cf3"``), under the Student's t model of its
        optimal portfolio (``"t_own"``), and under the skew-t model of the Student's t and
        normal optimal portfolios, converted to the same budget at the skew-t prices
        (``"t"`` and ``"normal"``).
    """
    models, market = Models.from_fit(fit), Market.from_fit(fit)
    portfolios = {
        "Skew-t": build_portfolio(models.skew_t, kind, market),
        "Student t": build_portfolio(models.student_t, kind, market),
    }
    shares = {
        (dist, objective): optimal_shares(portfolio, objective, tail_risk)
        for dist, portfolio in portfolios.items()
        for objective in ("var", "cf2", "cf3")
    }
    weights = pd.DataFrame(
        {
            f"{dist} {objective}": number_of_shares_to_weights(
                x, portfolios[dist].option_values
            )
            for (dist, objective), x in shares.items()
        },
        index=list(fit.tickers),
    )

    skew_t = portfolios["Skew-t"]
    normal = build_portfolio(models.normal, kind, market)

    def cfvar3_at_skew_t_prices(portfolio: OptionPortfolio, x: np.ndarray) -> float:
        """CFVaR3 under the skew-t model of the same weights, at the skew-t prices."""
        w = number_of_shares_to_weights(x, portfolio.option_values)
        return skew_t.get_CFVaR3_portfolio_loss(
            weights_to_number_of_shares(w, skew_t.option_values), tail_risk
        )

    risks = {
        objective: skew_t.get_CFVaR3_portfolio_loss(
            shares[("Skew-t", objective)], tail_risk
        )
        for objective in ("var", "cf2", "cf3")
    }
    risks["t_own"] = portfolios["Student t"].get_CFVaR3_portfolio_loss(
        shares[("Student t", "cf3")], tail_risk
    )
    risks["t"] = cfvar3_at_skew_t_prices(
        portfolios["Student t"], shares[("Student t", "cf3")]
    )
    risks["normal"] = cfvar3_at_skew_t_prices(
        normal, optimal_shares(normal, "cf3", tail_risk)
    )
    return weights, risks


def volatility_path(prices: np.ndarray, garch: pd.DataFrame) -> np.ndarray:
    """
    One-day-ahead GARCH(1,1) volatility forecasts over the test window.

    Row ``t`` is the forecast of the volatility of the return from day ``t`` to day
    ``t + 1``, so it only uses returns up to day ``t``.  Row 0 is the forecast at the end
    of the estimation sample, and the later rows follow the recursion
    ``sigma(t)^2 = a0 + a1 (L[t - 1, t] - iota)^2 + b1 sigma(t - 1)^2`` with the
    parameters fixed at their estimates.

    Parameters
    ----------
    prices : np.ndarray
        Closing prices, shape ``(n_days, N)``; row 0 is the last day of the estimation
        sample.
    garch : pd.DataFrame
        GARCH estimates of a :class:`~src.estimation.Fit`.

    Returns
    -------
    np.ndarray
        Daily volatility forecasts, shape ``(n_days - 1, N)``.
    """
    log_returns = np.diff(np.log(prices), axis=0)
    iota, a0, a1, b1 = (
        garch[c].to_numpy() for c in ("iota", "alpha0", "alpha1", "beta1")
    )
    variance = np.empty_like(log_returns)
    variance[0] = garch["sigma_forecast"].to_numpy() ** 2
    for t in range(1, len(variance)):
        variance[t] = a0 + a1 * (log_returns[t - 1] - iota) ** 2 + b1 * variance[t - 1]
    return np.sqrt(variance)


def _day_pnl(
    models: dict[str, Model],
    kind: str,
    market: Market,
    next_spots: np.ndarray,
    tail_risk: float,
) -> dict[str, float]:
    """
    Profit and loss of the CFVaR3-optimal portfolio of every model on one day.

    Each model prices the options at ``t`` and reprices them at ``t + Delta t`` with the
    volatility of the day, and its portfolio is optimized for a budget of 1.

    Parameters
    ----------
    models : dict[str, Model]
        Models by label.
    kind : str
        ``"call"`` or ``"put"``.
    market : Market
        Prices and GARCH inputs of day ``t``.
    next_spots : np.ndarray
        Prices on day ``t + 1``.
    tail_risk : float
        Tail risk ``alpha``.

    Returns
    -------
    dict[str, float]
        ``x' (V(S(t + Delta t), t + Delta t) - V(S(t), t))`` of each model.
    """
    pnl = {}
    for label, model in models.items():
        portfolio = build_portfolio(model, kind, market)
        tau = market.maturity - market.period
        next_values = np.array(
            [
                option.model.price(spot, option.strike, tau, market.rate, kind)
                for option, spot in zip(portfolio.options, next_spots)
            ]
        )
        x = optimal_shares(portfolio, "cf3", tail_risk).ravel()
        pnl[label] = float(x @ (next_values - portfolio.option_values.ravel()))
    return pnl


def out_of_sample(
    fit: Fit,
    prices: pd.DataFrame,
    num_days: int,
    kind: str,
    tail_risk: float,
) -> pd.DataFrame:
    """
    Out-of-sample test of Section 5.3.

    On each of the ``num_days`` trading days after the estimation sample, every model
    prices at-the-money options, optimizes its CFVaR3 portfolio for a budget of 1, and
    its profit and loss over the next day is recorded.  The parameters stay at their
    estimates, and the GARCH volatility is updated daily.

    Parameters
    ----------
    fit : Fit
        Fit of the estimation sample.
    prices : pd.DataFrame
        Closing prices indexed by date, which cover the test window.
    num_days : int
        Number of days of the test window.
    kind : str
        ``"call"`` or ``"put"``.
    tail_risk : float
        Tail risk ``alpha`` of the CFVaR3 portfolios.

    Returns
    -------
    pd.DataFrame
        Profit and loss of each model (the columns, labelled as ``MODEL_LABELS``) on
        each of the ``num_days`` days.

    Raises
    ------
    ValueError
        If ``prices`` do not cover the test window.
    """
    window = prices.loc[fit.last_day :].iloc[: num_days + 1]
    if len(window) < num_days + 1 or window.index[0] != fit.last_day:
        raise ValueError(f"The prices must cover {num_days} days after {fit.last_day}.")
    models, market = Models.from_fit(fit), Market.from_fit(fit)
    labelled = {label: getattr(models, name) for name, label in MODEL_LABELS.items()}
    spots = window.to_numpy(dtype=float)
    volatilities = volatility_path(spots, fit.garch)
    pnl = Parallel(n_jobs=-1)(
        delayed(_day_pnl)(
            labelled,
            kind,
            replace(market, spots=spots[day], daily_vol=volatilities[day]),
            spots[day + 1],
            tail_risk,
        )
        for day in range(num_days)
    )
    return pd.DataFrame(pnl, index=window.index[:num_days])


def empirical_risks(pnl: np.ndarray, tail_risk: float) -> np.ndarray:
    """
    Empirical variance, CFVaR2 and CFVaR3 of a sample of profits and losses.

    Parameters
    ----------
    pnl : np.ndarray
        Profit and loss observations.
    tail_risk : float
        Tail risk ``alpha``.

    Returns
    -------
    np.ndarray
        The variance, ``eCFVaR_2 = -mean - z sd`` and
        ``eCFVaR_3 = eCFVaR_2 - (z^2 - 1) / 6 * kappa_3 / variance`` with the empirical
        moments (the variance and the third central moment are not corrected for
        bias) and ``z = Phi^{-1}(alpha)``.
    """
    mean, variance = pnl.mean(), pnl.var()
    z = norm.ppf(tail_risk)
    ecfvar2 = -mean - z * np.sqrt(variance)
    kappa3 = np.mean((pnl - mean) ** 3)
    return np.array([variance, ecfvar2, ecfvar2 - (z**2 - 1) / 6 * kappa3 / variance])


def monte_carlo(
    fit: Fit,
    tail_risk: float,
    kinds: tuple[str, ...] = KINDS,
    nu: float | None = None,
) -> pd.DataFrame:
    """
    CFVaR3 of the optimal skew-t portfolio by Monte Carlo (Section 5.4).

    The price changes are simulated, and the CFVaR3 of the gain is estimated from the
    sample twice: with the delta-gamma gain, and with every option fully repriced at
    ``t + Delta t``.  The experiment is run with options of two months and of one year,
    and repeated with ``N_SEEDS`` independent sets of draws, which are averaged.

    Parameters
    ----------
    fit : Fit
        Fit of the estimation sample.
    tail_risk : float
        Tail risk ``alpha``.
    kinds : tuple[str, ...], optional
        Option types, in the order of the rows, by default calls then puts.
    nu : float, optional
        Degrees of freedom that replace the fitted ones, the other fitted parameters
        being fixed, by default the fitted ones.

    Returns
    -------
    pd.DataFrame
        For each option type and maturity (the index), the CFVaR3 estimated with the
        delta-gamma gain (``"delta_gamma"``) and with full repricing (``"full"``), and
        the percentage change from the first to the second (``"pct_change"``).
    """
    model, market = Models.from_fit(fit).skew_t, Market.from_fit(fit)
    if nu is not None:
        model = replace(model, nu=nu)
    runs = []
    for seed in range(N_SEEDS):
        generators = {
            TIME_TO_MATURITY: np.random.default_rng([42, seed]),
            LONG_TIME_TO_MATURITY: np.random.default_rng([43, seed]),
        }
        rows = {}
        for kind in kinds:
            for maturity, rng in generators.items():
                portfolio = build_portfolio(
                    model, kind, replace(market, maturity=maturity)
                )
                x = optimal_shares(portfolio, "cf3", tail_risk).ravel()
                moves = sample_skew_t(
                    N_BATCHES * BATCH_SIZE,
                    portfolio.mu.ravel(),
                    portfolio.scale_matrix,
                    portfolio.omega.ravel(),
                    portfolio.degrees_of_freedom,
                    rng,
                )
                exact, approx = option_gains(
                    portfolio,
                    repricing_grids(portfolio, market.rate, market.period),
                    moves,
                )
                delta_gamma = cfvar3_estimate(approx @ x, tail_risk)
                full = cfvar3_estimate(exact @ x, tail_risk)
                rows[(kind, maturity)] = {
                    "delta_gamma": delta_gamma,
                    "full": full,
                    "pct_change": 100 * (full / delta_gamma - 1),
                }
        runs.append(pd.DataFrame.from_dict(rows, orient="index"))
    return sum(runs) / N_SEEDS


def pricing_check(fit: Fit) -> pd.DataFrame:
    """
    Monte Carlo check of the skew-t option prices (Appendix B).

    The log return over the maturity is simulated as the sum of i.i.d. skew-t one-period
    returns truncated above, and the price is the discounted expected payoff.

    Parameters
    ----------
    fit : Fit
        Fit of the estimation sample.

    Returns
    -------
    pd.DataFrame
        For each option type and asset (the index), the price of the pricing model
        (``"fft"``), the Monte Carlo price (``"monte_carlo"``) and its standard error
        (``"std_error"``), from independent batches.
    """
    model, market = Models.from_fit(fit).skew_t, Market.from_fit(fit)
    rng = np.random.default_rng(42)
    rows = {}
    for kind in KINDS:
        for option in build_portfolio(model, kind, market).options:
            returns = simulate_log_returns(option.model, option.time_to_maturity, rng)
            price, std_error = monte_carlo_price(
                returns,
                option.underlying.spot,
                option.strike,
                option.time_to_maturity,
                market.rate,
                kind,
            )
            rows[(kind, option.underlying.name)] = {
                "fft": option.price,
                "monte_carlo": price,
                "std_error": std_error,
            }
    return pd.DataFrame.from_dict(rows, orient="index")
