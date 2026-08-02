"""
Section 5.5 - Out-of-Sample Empirical CFVaR3 (eCFVaR3) Test.

For the five-stock dataset (DIS, XOM, PFE, MO, INTC), this script:

1. Loads historical adjusted close prices (most recent 750 trading days
   from the dataset, following Hu and Kercheval (2010)).
2. For each day t = 0, ..., 748 in the 750-day period, performs an
   **independent** experiment:
     - Constructs an ATM option portfolio at current spot prices S_t
       with a fixed time-to-maturity T.
     - Computes optimal portfolio weights (initial budget = $1) under
       multiple distributional assumptions.
     - Reports the empirical change in portfolio value:
         (Delta V^E)_t = x^T [V(S_{t+1}, T - 1/252) - V(S_t, T)]
       using **exact** option repricing (not delta-gamma approximation).
3. **Caches** the computed weights and price changes so that subsequent
   risk-measure computations do not require refitting portfolios.
4. Computes the **Empirical CFVaR3** (eCFVaR3) using empirical moments:

       E-CFVaR_3^alpha[DeltaV(x)]
           = -E^{emp}[DeltaV(x)]
             - Phi^{-1}(alpha) * sqrt(Var^{emp}[DeltaV(x)])
             - ((Phi^{-1}(alpha))^2 - 1) / 6
               * kappa_3^{emp}[DeltaV(x)] / Var^{emp}[DeltaV(x)]

   where E^{emp}, Var^{emp}, kappa_3^{emp} are computed from the 749
   discrete daily observations.

Distributional models tested
-----------------------------
All strategies use **skew-t Gosset option pricing** for both the budget
constraint and empirical P&L measurement, ensuring a fair comparison.
The strategies differ **only** in the distributional assumption used
for the optimization criterion:

- **Skew-t model** (AC estimated parameters):
    Variance, CFVaR2, CFES2, CFVaR3.
- **Student-t model (HK)** (Hu & Kercheval 2010 literature parameters, omega = 0):
    Variance, CFVaR2, CFVaR3.
- **Student-t model (New)** (AC estimated, omega = 0):
    Variance, CFVaR2, CFVaR3.
- **Normal model** (omega = 0, nu_eff = 1000):
    Variance, CFVaR2, CFVaR3.

Outputs
-------
Per parameter set:
- ``cache/weights_{param_set}_{option_type}.pkl``          -- cached weights per strategy per day (pickle dict).
- ``out_of_sample_ecfvar3_summary_{param_set}_{option_type}.csv``  -- eCFVaR3 summary.
- ``out_of_sample_ecfvar3_summary_{param_set}_{option_type}.tex``  -- LaTeX table.

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiments/paper1/sec_5_4_out_of_sample/eCFVaR3_consistent_prices.py
"""

import csv
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.constants.dataset_config import (
    DEFAULT_SKEWT_CONFIG,
    DEFAULT_T_CONFIG,
    HK_NORMAL_CONFIG,
    HK_T_CONFIG,
    SKEWT_CONFIGS,
    DatasetConfig,
)
from src.options.option import MarketEnvironment, Underlying
from src.options.skew_t_gosset_call import SkewTGossetCall
from src.options.skew_t_gosset_put import SkewTGossetPut
from src.portfolio.option_portfolio import OptionPortfolio
from src.portfolio.utils import build_scale_matrix
from src.utils.skew_t_distribution import (
    compute_alpha_1d_from_omega,
    compute_omega_delta_s,
)

# ======================================================================
# Configuration
# ======================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_DIR / "data"
CACHE_DIR = SCRIPT_DIR / "cache"

# Use the default (best-MLE) skew-t config for pricing
_PRICING_CFG = DEFAULT_SKEWT_CONFIG

STOCK_NAMES = list(_PRICING_CFG.asset_tickers)
N_STOCKS = len(STOCK_NAMES)

RISK_FREE_RATE = 0.05
TIME_PERIOD = 1 / 252  # one trading day in years
TAIL_RISK = 0.01  # alpha = 1%
DEGREES_OF_FREEDOM = _PRICING_CFG.nu

N_DAYS = 750  # number of price observations in the test window
N_OBS = N_DAYS - 1  # number of one-day P&L observations (749)

# Confidence levels for eCFVaR3 reporting.
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Fixed time-to-maturity for each day's independent option position.
TIME_TO_MATURITY = 1.0  # 1 year

volatilities = _PRICING_CFG.annual_vol.tolist()

# Per-asset 1-D Azzalini skewness parameters (constant, derived from AC fit)
_ALPHA_1D = compute_alpha_1d_from_omega(_PRICING_CFG.omega, _PRICING_CFG.sigma_dp)

# Effective degrees of freedom for the Normal model (nu -> inf approximation)
_NORMAL_NU_EFF = 1000.0

# Normal and t configs for the baseline comparison models
_NORMAL_CFG = HK_NORMAL_CONFIG
_T_CFG = DEFAULT_T_CONFIG


# ======================================================================
# Empirical CFVaR3 computation
# ======================================================================
def empirical_cfvar3(pnl: np.ndarray, alpha: float) -> float:
    """Compute the Empirical CFVaR3 from discrete P&L observations.

    Parameters
    ----------
    pnl : np.ndarray
        Array of portfolio P&L (DeltaV) observations, shape ``(n,)``.
    alpha : float
        Tail risk level (e.g., 0.05 for 95% confidence).

    Returns
    -------
    float
        The empirical CFVaR3 value (positive means loss).

    Formula
    -------
    E-CFVaR_3^alpha[DeltaV(x)]
        = -E^{emp}[DeltaV] - Phi^{-1}(alpha) * sqrt(Var^{emp}[DeltaV])
          - ((Phi^{-1}(alpha))^2 - 1) / 6 * kappa_3^{emp}[DeltaV] / Var^{emp}[DeltaV]

    where kappa_3^{emp} is the empirical third central moment.
    """
    mean_pnl = np.mean(pnl)
    var_pnl = np.var(pnl, ddof=0)  # empirical variance (population)
    # Third central moment: E[(X - mu)^3]
    if var_pnl == 0:
        raise ValueError("Empirical variance is zero; eCFVaR3 is undefined.")
    kappa3_pnl = np.mean((pnl - mean_pnl) ** 3)

    z_alpha = norm.ppf(alpha)

    term1 = -mean_pnl
    term2 = -z_alpha * np.sqrt(var_pnl)
    term3 = -((z_alpha**2 - 1) / 6) * (kappa3_pnl / var_pnl)

    return float(term1 + term2 + term3)


# ======================================================================
# Option construction helpers (always skew-t Gosset for consistent pricing)
# ======================================================================
def _build_skew_t_options(spot_prices: np.ndarray, option_type: str = "call") -> tuple:
    """Build ATM skew-t Gosset options and underlyings at given spots."""
    OptionClass = SkewTGossetCall if option_type == "call" else SkewTGossetPut

    underlyings = []
    options = []
    for i, (sp, vol) in enumerate(zip(spot_prices.tolist(), volatilities)):
        underlying = Underlying(name=STOCK_NAMES[i], spot=sp, volatility=vol)
        underlyings.append(underlying)
        market_env = MarketEnvironment(annual_risk_free_rate=RISK_FREE_RATE)
        option = OptionClass(
            strike=sp,  # ATM
            time_to_maturity=TIME_TO_MATURITY,
            underlying=underlying,
            market_env=market_env,
            x_p=0,
            x_c=np.exp(4),
            alpha_skew=float(_ALPHA_1D[i]),
            degrees_of_freedom=DEGREES_OF_FREEDOM,
            dividend_yield=0.0,
        )
        options.append(option)
    return underlyings, options


# ======================================================================
# Portfolio builders
# ======================================================================
def build_skew_t_portfolio(
    spot_prices: np.ndarray,
    option_type: str = "call",
    skewt_config: Optional[DatasetConfig] = None,
) -> OptionPortfolio:
    """Build a skew-t portfolio for optimization (full skew-t parameters)."""
    if skewt_config is None:
        skewt_config = _PRICING_CFG

    underlyings, options = _build_skew_t_options(spot_prices, option_type)

    omega_delta_s = compute_omega_delta_s(
        omega_dp=skewt_config.omega,
        sigma_dp=skewt_config.sigma_dp,
        volatilities=np.array(volatilities),
        spot_prices=spot_prices,
        time_period=TIME_PERIOD,
    )

    scale_matrix = build_scale_matrix(
        time_period_for_risk_measurements=TIME_PERIOD,
        correlation=skewt_config.corr,
        volatilities=np.array(volatilities),
        spot_prices=spot_prices,
    )
    return OptionPortfolio(
        underlyings=underlyings,
        options=options,
        time_period_for_risk_measurements=TIME_PERIOD,
        scale_matrix=scale_matrix,
        omega=omega_delta_s,
        degrees_of_freedom=skewt_config.nu,
        returns=skewt_config.mean_annual,
    )


def build_t_portfolio(
    spot_prices: np.ndarray,
    option_type: str = "call",
    t_config: Optional[DatasetConfig] = None,
) -> OptionPortfolio:
    """Build a Student-t portfolio for optimization (omega = 0).

    Notes
    -----
    If ``t_config.nu < 6`` (e.g. the HK fit returns nu = 5.87), the
    degrees-of-freedom value is replaced by the skew-t pricing config's
    nu to avoid numerical issues with very low d.o.f.
    """
    if t_config is None:
        t_config = _T_CFG

    # Guard against very low d.o.f. (can occur with HK parameters).
    nu = t_config.nu if t_config.nu >= 6.0 else _PRICING_CFG.nu

    underlyings, options = _build_skew_t_options(spot_prices, option_type)

    scale_matrix = build_scale_matrix(
        time_period_for_risk_measurements=TIME_PERIOD,
        correlation=t_config.corr,
        volatilities=np.array(volatilities),
        spot_prices=spot_prices,
    )
    return OptionPortfolio(
        underlyings=underlyings,
        options=options,
        time_period_for_risk_measurements=TIME_PERIOD,
        scale_matrix=scale_matrix,
        omega=None,
        degrees_of_freedom=nu,
        returns=t_config.mu_annual,
    )


def build_normal_portfolio(
    spot_prices: np.ndarray,
    option_type: str = "call",
    normal_config: Optional[DatasetConfig] = None,
) -> OptionPortfolio:
    """Build a Normal portfolio for optimization (omega = 0, nu -> inf)."""
    if normal_config is None:
        normal_config = _NORMAL_CFG

    underlyings, options = _build_skew_t_options(spot_prices, option_type)

    scale_matrix = build_scale_matrix(
        time_period_for_risk_measurements=TIME_PERIOD,
        correlation=normal_config.corr,
        volatilities=np.array(volatilities),
        spot_prices=spot_prices,
    )
    return OptionPortfolio(
        underlyings=underlyings,
        options=options,
        time_period_for_risk_measurements=TIME_PERIOD,
        scale_matrix=scale_matrix,
        omega=None,
        degrees_of_freedom=_NORMAL_NU_EFF,
        returns=normal_config.mu_annual,
    )


# ======================================================================
# Option repricing helper (always skew-t Gosset)
# ======================================================================
def _price_options(
    spot_prices: np.ndarray,
    strikes: np.ndarray,
    ttm: float,
    option_type: str = "call",
) -> np.ndarray:
    """Price N_STOCKS skew-t Gosset options at given (spot, strike, ttm)."""
    OptionClass = SkewTGossetCall if option_type == "call" else SkewTGossetPut
    prices = np.zeros(N_STOCKS)
    for i in range(N_STOCKS):
        underlying = Underlying(
            name=STOCK_NAMES[i],
            spot=float(spot_prices[i]),
            volatility=volatilities[i],
        )
        market_env = MarketEnvironment(annual_risk_free_rate=RISK_FREE_RATE)
        option = OptionClass(
            strike=float(strikes[i]),
            time_to_maturity=ttm,
            underlying=underlying,
            market_env=market_env,
            x_p=0,
            x_c=np.exp(4),
            alpha_skew=float(_ALPHA_1D[i]),
            degrees_of_freedom=DEGREES_OF_FREEDOM,
            dividend_yield=0.0,
        )
        prices[i] = option.price
    return prices


# ======================================================================
# Load historical data
# ======================================================================
def load_price_data() -> pd.DataFrame:
    """Load the most recent 750 trading days from the five-stock price CSV."""
    csv_path = DATA_DIR / "five_stock_prices.csv"
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices = prices[STOCK_NAMES]
    prices = prices.tail(N_DAYS).reset_index(drop=True)
    return prices


# ======================================================================
# Strategy definitions
# ======================================================================
STRATEGY_CONFIGS = {
    "skew_t": {
        "builder": build_skew_t_portfolio,
        "methods": {
            "SkewT-Var": "variance",
            "SkewT-CFVaR2": "cfvar2",
            "SkewT-CFES2": "cfes2",
            "SkewT-CFVaR3": "cfvar3",
        },
    },
    "t_hk": {
        "builder": lambda sp, ot: build_t_portfolio(sp, ot, t_config=HK_T_CONFIG),
        "methods": {
            "HK-t-Var": "variance",
            "HK-t-CFVaR2": "cfvar2",
            "HK-t-CFVaR3": "cfvar3",
        },
    },
    "t_new": {
        "builder": build_t_portfolio,
        "methods": {
            "New-t-Var": "variance",
            "New-t-CFVaR2": "cfvar2",
            "New-t-CFVaR3": "cfvar3",
        },
    },
    "normal": {
        "builder": build_normal_portfolio,
        "methods": {
            "Normal-Var": "variance",
            "Normal-CFVaR2": "cfvar2",
            "Normal-CFVaR3": "cfvar3",
        },
    },
}


def _get_optimal_shares(portfolio: OptionPortfolio, method: str) -> np.ndarray:
    """Extract optimal shares vector from a portfolio for a given method."""
    if method == "variance":
        x, _ = portfolio.get_optimal_variance_weights_lagrange()
    elif method == "cfvar2":
        x, _ = portfolio.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    elif method == "cfes2":
        x, _ = portfolio.get_optimal_CFES2_weights_lagrange(TAIL_RISK)
    elif method == "cfvar3":
        x, _ = portfolio.get_optimal_CFVaR3_weights_numeric(TAIL_RISK)
    else:
        raise ValueError(f"Unknown method: {method}")
    return x.flatten()


# ======================================================================
# Cache management
# ======================================================================
def _get_cache_path(param_set_name: str, option_type: str) -> Path:
    """Return the pickle cache path for a given configuration."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{param_set_name}_{option_type}"
    return CACHE_DIR / f"weights_{suffix}.pkl"


def _fit_and_cache(
    prices: pd.DataFrame,
    option_type: str,
    param_set_name: str,
    strategy_configs: dict,
) -> tuple:
    """Fit all portfolios daily and cache weights + price changes via pickle.

    Returns
    -------
    tuple
        (all_strategy_names, weights_dict, price_changes)
        - all_strategy_names: list of strategy name strings
        - weights_dict: {strategy_name: np.ndarray of shape (N_OBS, N_STOCKS)}
        - price_changes: np.ndarray of shape (N_OBS, N_STOCKS) -- option dV per day
    """
    cache_path = _get_cache_path(param_set_name, option_type)

    # Collect all strategy names in order
    all_strategy_names = []
    for config in strategy_configs.values():
        all_strategy_names.extend(config["methods"].keys())

    # Check if cache exists
    if cache_path.exists():
        print(f"  Loading cached data from {cache_path.name}...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        weights_dict = {name: cached["weights"][name] for name in all_strategy_names}
        price_changes = cached["price_changes"]
        return all_strategy_names, weights_dict, price_changes

    # Otherwise, fit portfolios and cache
    print("  Fitting portfolios (will cache for future use)...")

    price_array = prices.values  # (N_DAYS, N_STOCKS)
    ttm_next = TIME_TO_MATURITY - TIME_PERIOD  # T - 1/252

    # Initialize storage
    weights_dict = {name: np.zeros((N_OBS, N_STOCKS)) for name in all_strategy_names}
    price_changes = np.zeros((N_OBS, N_STOCKS))

    for t in range(N_OBS):
        if t % 50 == 0:
            print(f"    Day {t}/{N_OBS}...")

        spot_t = price_array[t]  # S_t
        spot_t1 = price_array[t + 1]  # S_{t+1}
        strikes = spot_t.copy()  # ATM at day t

        # V(S_t, T) and V(S_{t+1}, T-1/252) using consistent skew-t pricing
        v_t = _price_options(spot_t, strikes, TIME_TO_MATURITY, option_type)
        v_t1 = _price_options(spot_t1, strikes, ttm_next, option_type)
        dv = v_t1 - v_t  # option price change vector (N_STOCKS,)
        price_changes[t] = dv

        for model_name, config in strategy_configs.items():
            builder = config["builder"]
            portfolio = builder(spot_t, option_type)

            for strat_name, method in config["methods"].items():
                x = _get_optimal_shares(portfolio, method)
                weights_dict[strat_name][t] = x

    print(f"    Day {N_OBS}/{N_OBS}... done.")

    # Save cache as a single pickle dictionary
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = {
        "weights": weights_dict,
        "price_changes": price_changes,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)
    print(f"  Cached weights and price changes to {cache_path.name}")

    return all_strategy_names, weights_dict, price_changes


# ======================================================================
# Main experiment
# ======================================================================
def _run_experiment(
    prices: pd.DataFrame,
    option_type: str,
    param_set_name: str = "default",
    strategy_configs: Optional[dict] = None,
) -> None:
    """Run the eCFVaR3 experiment for one option type.

    Steps:
      1. Fit (or load cached) weights and price changes.
      2. Compute daily P&L: DeltaV_t = x_t^T * dv_t.
      3. Compute eCFVaR3 from the 749 P&L observations.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data, shape ``(N_DAYS, N_STOCKS)``.
    option_type : str
        ``"call"`` or ``"put"``.
    param_set_name : str
        Label for the parameter set (used in output filenames).
    strategy_configs : dict, optional
        Strategy configuration. If None, uses module-level STRATEGY_CONFIGS.
    """
    if strategy_configs is None:
        strategy_configs = STRATEGY_CONFIGS

    label = option_type.capitalize()
    print(f"\n{'-' * 70}")
    print(f"  Option type: {label} | Params: {param_set_name}")
    print(f"{'-' * 70}")

    # Step 1: Fit (or load from cache) weights and price changes
    all_strategy_names, weights_dict, price_changes = _fit_and_cache(
        prices, option_type, param_set_name, strategy_configs
    )

    # Step 2: Compute daily P&L for each strategy
    pnl_results = {}
    for name in all_strategy_names:
        # P&L_t = x_t^T * dv_t (element-wise dot product per day)
        pnl = np.sum(weights_dict[name] * price_changes, axis=1)
        pnl_results[name] = pnl

    # Step 3: Compute eCFVaR3 at multiple confidence levels
    ecfvar3_multi = {}  # {conf_level: {strategy_name: value}}
    for cl in CONFIDENCE_LEVELS:
        alpha = 1.0 - cl  # tail probability
        ecfvar3_multi[cl] = {}
        for name in all_strategy_names:
            ecfvar3_multi[cl][name] = empirical_cfvar3(pnl_results[name], alpha)

    # Also compute empirical mean, variance, and third central moment for reporting
    stats = {}
    for name in all_strategy_names:
        pnl = pnl_results[name]
        stats[name] = {
            "mean": float(np.mean(pnl)),
            "variance": float(np.var(pnl, ddof=0)),
            "kappa3": float(np.mean((pnl - np.mean(pnl)) ** 3)),
        }

    # Print results
    print("\nResults (Empirical CFVaR3 at multiple confidence levels):")
    header = f"  {'Strategy':<16s}"
    for cl in CONFIDENCE_LEVELS:
        pct = int(cl * 100)
        header += f" {'eCFVaR3 ' + str(pct) + '%':<16s}"
    print(header)
    print("  " + "-" * (16 + 16 * len(CONFIDENCE_LEVELS)))

    for name in all_strategy_names:
        row = f"  {name:<16s}"
        for cl in CONFIDENCE_LEVELS:
            row += f" {ecfvar3_multi[cl][name]:<16.6f}"
        print(row)
    print()

    # Print empirical moments
    print("  Empirical Moments:")
    print(f"  {'Strategy':<16s} {'Mean':<14s} {'Variance':<14s} {'Kappa3':<14s}")
    print("  " + "-" * 58)
    for name in all_strategy_names:
        s = stats[name]
        print(
            f"  {name:<16s} {s['mean']:<14.8f} {s['variance']:<14.8f} {s['kappa3']:<14.8f}"
        )
    print()

    # -- Save outputs -------------------------------------------------
    suffix = f"_{param_set_name}_{option_type}"

    # Daily P&L CSV (includes weights for reference)
    pnl_csv_path = SCRIPT_DIR / f"out_of_sample_ecfvar3_pnl{suffix}.csv"
    with open(pnl_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = ["day"] + [f"pnl_{name}" for name in all_strategy_names]
        writer.writerow(header_row)
        for d in range(N_OBS):
            row = [d + 1] + [
                f"{pnl_results[name][d]:.10f}" for name in all_strategy_names
            ]
            writer.writerow(row)
    print(f"Saved daily P&L: {pnl_csv_path.name}")

    # Summary CSV
    summary_csv_path = SCRIPT_DIR / f"out_of_sample_ecfvar3_summary{suffix}.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = [
            "strategy",
            "empirical_mean",
            "empirical_variance",
            "empirical_kappa3",
        ]
        for cl in CONFIDENCE_LEVELS:
            pct = int(cl * 100)
            header_row.append(f"ecfvar3_{pct}")
        writer.writerow(header_row)
        for name in all_strategy_names:
            row = [
                name,
                f"{stats[name]['mean']:.10f}",
                f"{stats[name]['variance']:.10f}",
                f"{stats[name]['kappa3']:.10f}",
            ]
            for cl in CONFIDENCE_LEVELS:
                row.append(f"{ecfvar3_multi[cl][name]:.6f}")
            writer.writerow(row)
    print(f"Saved eCFVaR3 summary: {summary_csv_path.name}")

    # LaTeX table
    tex_path = SCRIPT_DIR / f"out_of_sample_ecfvar3_summary{suffix}.tex"
    _write_latex_table(tex_path, option_type, all_strategy_names, ecfvar3_multi, stats)
    print(f"Saved LaTeX table: {tex_path.name}")


def _write_latex_table(
    path: Path,
    option_type: str,
    strategy_names: list,
    ecfvar3_multi: dict,
    stats: dict,
) -> None:
    """Write the eCFVaR3 summary as a LaTeX table."""
    n_levels = len(CONFIDENCE_LEVELS)
    # Columns: strategy | mean | variance | kappa3 | eCFVaR3 at each level
    col_spec = "l" + "r" * 3 + "r" * n_levels

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header_parts = [
        "Strategy",
        r"$\hat{\mu}$",
        r"$\hat{\sigma}^2$",
        r"$\hat{\kappa}_3$",
    ]
    for cl in CONFIDENCE_LEVELS:
        pct = int(cl * 100)
        header_parts.append(rf"eCFVaR3 ({pct}\%)")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    for name in strategy_names:
        s = stats[name]
        row_parts = [
            name.replace("_", r"\_"),
            f"{s['mean']:.6f}",
            f"{s['variance']:.6f}",
            f"{s['kappa3']:.6f}",
        ]
        for cl in CONFIDENCE_LEVELS:
            row_parts.append(f"{ecfvar3_multi[cl][name]:.4f}")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Out-of-sample empirical CFVaR3 (eCFVaR3) for the "
            rf"{option_type} portfolio "
            r"(749 independent one-day experiments over 750 trading days). "
            r"Each day the portfolio is reoptimised at current ATM spot prices "
            r"with initial budget \$1. "
            r"The empirical moments $\hat{\mu}$, $\hat{\sigma}^2$, $\hat{\kappa}_3$ "
            r"are computed from the 749 daily P\&L observations. "
            r"eCFVaR3 uses the Cornish--Fisher expansion with empirical moments.}}",
            rf"\label{{tab:out_of_sample_ecfvar3_{option_type}}}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    """Run the eCFVaR3 out-of-sample experiment.

    Loops over all available skew-t parameter sets. Caches weights and
    price changes so that subsequent runs (or alternative risk measures)
    do not require refitting.

    For the Student-t model, we run both:
      - HK Student-t (Hu & Kercheval 2010 literature parameters)
      - New Student-t (our custom AC-estimated parameters)
    These are labelled clearly in the output filenames and strategy names.
    """
    print("=" * 70)
    print("Section 5.5 - Out-of-Sample Empirical CFVaR3 (eCFVaR3) Test")
    print("=" * 70)

    prices = load_price_data()
    print(f"Loaded {len(prices)} days of price data.")
    print(f"Number of one-day P&L observations: {N_OBS}")
    print(f"Tail risk levels (alpha): {[(1-cl) for cl in CONFIDENCE_LEVELS]}")
    print(f"Option TTM: {TIME_TO_MATURITY} years (fixed per-day ATM position)")
    print()

    # Use only the default skew-t parameter set (AC-estimated, Gaussian-GARCH-Skew-t)
    param_sets = [("default_skewt", _PRICING_CFG)]

    print(f"Parameter set: {param_sets[0][0]} ({_PRICING_CFG.name})")
    print()

    for param_name, skewt_config in param_sets:
        print(f"\n{'═' * 70}")
        print(f"  PARAMETER SET: {param_name} ({skewt_config.name})")
        print(f"  nu = {skewt_config.nu:.4f}, logL = {skewt_config.loglik}")
        print(f"{'═' * 70}")

        # Build strategy configs with BOTH HK and New Student-t
        strategy_configs = {
            "skew_t": {
                "builder": lambda sp, ot, cfg=skewt_config: build_skew_t_portfolio(
                    sp, ot, skewt_config=cfg
                ),
                "methods": {
                    "SkewT-Var": "variance",
                    "SkewT-CFVaR2": "cfvar2",
                    "SkewT-CFES2": "cfes2",
                    "SkewT-CFVaR3": "cfvar3",
                },
            },
            "t_hk": {
                "builder": lambda sp, ot: build_t_portfolio(
                    sp, ot, t_config=HK_T_CONFIG
                ),
                "methods": {
                    "HK-t-Var": "variance",
                    "HK-t-CFVaR2": "cfvar2",
                    "HK-t-CFVaR3": "cfvar3",
                },
            },
            "t_new": {
                "builder": lambda sp, ot: build_t_portfolio(
                    sp, ot, t_config=DEFAULT_T_CONFIG
                ),
                "methods": {
                    "New-t-Var": "variance",
                    "New-t-CFVaR2": "cfvar2",
                    "New-t-CFVaR3": "cfvar3",
                },
            },
            "normal": {
                "builder": build_normal_portfolio,
                "methods": {
                    "Normal-Var": "variance",
                    "Normal-CFVaR2": "cfvar2",
                    "Normal-CFVaR3": "cfvar3",
                },
            },
        }

        for option_type in ("call", "put"):
            _run_experiment(
                prices,
                option_type,
                param_set_name=param_name,
                strategy_configs=strategy_configs,
            )

    print("\nDone. All parameter sets evaluated.")


if __name__ == "__main__":
    main()
