r"""
Section 5.2 - Effect of Skewed Distributions.

Compares portfolio weights under the skew-t distribution vs the standard
Student t distribution.  Both distributions use the same degrees of freedom
for a fair comparison (CFVaR3 requires \nu > 6).

For each (TAIL_RISK, TIME_PERIOD) combination two figures are produced:

Figure 1 - Portfolio Weights (2 by 2 grid):
  (0,0) Skew-Gosset Calls  - Skew-t weights    (Var, CFVaR2, CFVaR3 grouped bars)
  (0,1) Gosset Calls  - Standard t weights
  (1,0) Skew-Gosset Puts   - Skew-t weights
  (1,1) Gosset Puts   - Standard t weights

Figure 2 - % Change from Variance Weights (2 by 2 grid):
  Same layout; bars show (w_cfvar2 - w_var) / |w_var|  by  100  and
  (w_cfvar3 - w_var) / |w_var| by 100.

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiments/paper1/sec_5_2_numeric/section_5_2_skew_t_vs_t.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# >>> ONE-LINE DATASET SWITCH: change the imported configs to use different parameters.
# Available skew-t: DEFAULT_SKEWT_CONFIG (=GAUSSIAN_GARCH_SKEWT_CONFIG),
#                   T_GARCH_SKEWT_CONFIG, SKEWT_GARCH_SKEWT_CONFIG
# Available t:      HK_T_CONFIG, GAUSSIAN_GARCH_T_CONFIG, T_GARCH_T_CONFIG, SKEWT_GARCH_T_CONFIG
from src.constants.dataset_config import DEFAULT_SKEWT_CONFIG as skewt_cfg
from src.constants.dataset_config import HK_T_CONFIG as t_cfg
from src.constants.plotting import SEABORN_CONTEXT, SEABORN_PALETTE, SEABORN_STYLE
from src.options.gosset_call import GossetCall
from src.options.gosset_put import GossetPut
from src.options.option import MarketEnvironment, Underlying
from src.options.skew_t_gosset_call import SkewTGossetCall
from src.options.skew_t_gosset_put import SkewTGossetPut
from src.portfolio.option_portfolio import OptionPortfolio
from src.portfolio.skew_t_option_portfolio import SkewTOptionPortfolio
from src.portfolio.utils import build_scale_matrix
from src.utils.filenames import dot_to_p
from src.utils.skew_t_distribution import (
    compute_alpha_1d_from_omega,
    compute_omega_delta_s,
)

sns.set_style(SEABORN_STYLE)
sns.set_palette(SEABORN_PALETTE)
sns.set_context(SEABORN_CONTEXT)

assert skewt_cfg

SCRIPT_DIR = Path(__file__).resolve().parent / "figures"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.05
TIME_TO_MATURITY = 1.0
BAR_FORMAT = "%.3f"
DT_LABEL = {1 / 252: "1 day", 1 / 52: "1 week", 1 / 12: "1 month"}

STOCK_NAMES = list(skewt_cfg.asset_tickers)
stock_prices = skewt_cfg.stock_prices.tolist()
volatilities = skewt_cfg.annual_vol.tolist()


def _build_skew_t_portfolio(
    option_type: str,
    nu: float,
    time_period: float,
) -> SkewTOptionPortfolio:
    """Build an ATM skew-t Gosset option portfolio."""
    alpha_1d = compute_alpha_1d_from_omega(skewt_cfg.omega, skewt_cfg.sigma_dp)
    omega_delta_s = compute_omega_delta_s(
        omega_dp=skewt_cfg.omega,
        sigma_dp=skewt_cfg.sigma_dp,
        volatilities=np.array(volatilities),
        spot_prices=np.array(stock_prices),
        time_period=time_period,
    )

    underlyings = []
    options = []
    for i, (sp, vol) in enumerate(zip(stock_prices, volatilities)):
        underlying = Underlying(name=STOCK_NAMES[i], spot=sp, volatility=vol)
        underlyings.append(underlying)
        market_env = MarketEnvironment(annual_risk_free_rate=RISK_FREE_RATE)
        kw = dict(
            strike=sp,
            time_to_maturity=TIME_TO_MATURITY,
            underlying=underlying,
            market_env=market_env,
            x_p=0,
            x_c=np.exp(4),
            alpha_skew=float(alpha_1d[i]),
            degrees_of_freedom=nu,
            dividend_yield=0.0,
        )
        if option_type == "call":
            options.append(SkewTGossetCall(**kw))
        else:
            options.append(SkewTGossetPut(**kw))

    scale_matrix = build_scale_matrix(
        time_period_for_risk_measurements=time_period,
        correlation=skewt_cfg.corr,
        volatilities=np.array(volatilities),
        spot_prices=np.array(stock_prices),
    )
    return SkewTOptionPortfolio(
        underlyings=underlyings,
        options=options,
        time_period_for_risk_measurements=time_period,
        scale_matrix=scale_matrix,
        omega=omega_delta_s,
        degrees_of_freedom=nu,
        returns=skewt_cfg.mean_annual,
    )


def _build_t_portfolio(
    option_type: str,
    nu: float,
    time_period: float,
) -> OptionPortfolio:
    """Build an ATM standard-t Gosset option portfolio."""
    underlyings = []
    options = []
    for i, (sp, vol) in enumerate(zip(stock_prices, volatilities)):
        underlying = Underlying(name=STOCK_NAMES[i], spot=sp, volatility=vol)
        underlyings.append(underlying)
        market_env = MarketEnvironment(annual_risk_free_rate=RISK_FREE_RATE)
        kw = dict(
            strike=sp,
            time_to_maturity=TIME_TO_MATURITY,
            underlying=underlying,
            market_env=market_env,
            x_p=0,
            x_c=np.exp(4),
            p_p=0,
            p_c=0.99,
            degrees_of_freedom=nu,
            dividend_yield=0.0,
        )
        if option_type == "call":
            options.append(GossetCall(**kw))
        else:
            options.append(GossetPut(**kw))

    scale_matrix = build_scale_matrix(
        time_period_for_risk_measurements=time_period,
        correlation=t_cfg.corr,
        volatilities=np.array(volatilities),
        spot_prices=np.array(stock_prices),
    )
    return OptionPortfolio(
        underlyings=underlyings,
        options=options,
        time_period_for_risk_measurements=time_period,
        scale_matrix=scale_matrix,
        degrees_of_freedom=nu,
        returns=t_cfg.mu_annual,
    )


def _pct_change(w_new: np.ndarray, w_base: np.ndarray) -> np.ndarray:
    """Percentage change relative to |w_base|, zero-safe."""
    return np.where(
        np.abs(w_base) < 1e-10, 0.0, (w_new - w_base) / np.abs(w_base) * 100
    )


TAIL_RISK = 0.01
TIME_PERIOD = 1 / 252

weights = {}

for opt_type in ("call", "put"):
    # Skew-t
    pf_sk = _build_skew_t_portfolio(opt_type, skewt_cfg.nu, TIME_PERIOD)
    _, w_var_sk = pf_sk.get_optimal_variance_weights_lagrange()
    _, w_cfv2_sk = pf_sk.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    _, w_cfv3_sk = pf_sk.get_optimal_CFVaR3_weights_numeric(TAIL_RISK)
    weights[("skewt", opt_type, "variance")] = w_var_sk.ravel()
    weights[("skewt", opt_type, "cfvar2")] = w_cfv2_sk.ravel()
    weights[("skewt", opt_type, "cfvar3")] = w_cfv3_sk.ravel()

    # Standard t
    pf_t = _build_t_portfolio(opt_type, skewt_cfg.nu, TIME_PERIOD)
    _, w_var_t = pf_t.get_optimal_variance_weights_lagrange()
    _, w_cfv2_t = pf_t.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    _, w_cfv3_t = pf_t.get_optimal_CFVaR3_weights_numeric(TAIL_RISK)
    weights[("t", opt_type, "variance")] = w_var_t.ravel()
    weights[("t", opt_type, "cfvar2")] = w_cfv2_t.ravel()
    weights[("t", opt_type, "cfvar3")] = w_cfv3_t.ravel()

n_assets = len(STOCK_NAMES)
bar_width = 0.25
x_pos = np.arange(n_assets)

# ------------------------------------------------------------------
# Figure 1: Portfolio Weights
# ------------------------------------------------------------------
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
fig1.suptitle(
    r"Effect of Skewed Distributions: Portfolio Weights"
    "\n"
    rf"($\alpha={TAIL_RISK}$, $\Delta t=$ {DT_LABEL[TIME_PERIOD]}, $\nu={skewt_cfg.nu:.3f}$)",
    fontsize=11,
)

subplot_cfg = [
    (
        0,
        0,
        "call",
        "skewt",
        r"Skew-Gosset Calls: Skew-$t$",
    ),
    (
        0,
        1,
        "call",
        "t",
        r"Gosset Calls: Standard $t$",
    ),
    (
        1,
        0,
        "put",
        "skewt",
        r"Skew-Gosset Puts: Skew-$t$",
    ),
    (
        1,
        1,
        "put",
        "t",
        r"Gosset Puts: Standard $t$",
    ),
]

for row, col, opt_type, dist, title in subplot_cfg:
    ax = axes1[row][col]
    w_var = weights[(dist, opt_type, "variance")]
    w_cfv2 = weights[(dist, opt_type, "cfvar2")]
    w_cfv3 = weights[(dist, opt_type, "cfvar3")]

    b1 = ax.bar(
        x_pos - bar_width,
        w_var,
        bar_width,
        label="Variance",
        color="steelblue",
        alpha=0.8,
    )
    b2 = ax.bar(x_pos, w_cfv2, bar_width, label="CFVaR2", color="orange", alpha=0.8)
    b3 = ax.bar(
        x_pos + bar_width,
        w_cfv3,
        bar_width,
        label="CFVaR3",
        color="green",
        alpha=0.8,
    )

    ax.bar_label(b1, fmt=BAR_FORMAT, padding=2, fontsize=6)
    ax.bar_label(b2, fmt=BAR_FORMAT, padding=2, fontsize=6)
    ax.bar_label(b3, fmt=BAR_FORMAT, padding=2, fontsize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STOCK_NAMES)
    ax.set_xlabel("Underlying Asset")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

plt.tight_layout()
out1 = SCRIPT_DIR / (
    dot_to_p(f"skew_vs_t_weights_alpha{TAIL_RISK}_dt{TIME_PERIOD:.4f}") + ".png"
)
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1.name}")
plt.close(fig1)

# ------------------------------------------------------------------
# Figure 2: % Change from Variance Weights
# ------------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
fig2.suptitle(
    r"Effect of Skewed Distributions: % Change from Variance Weights"
    "\n"
    rf"($\alpha={TAIL_RISK}$, $\Delta t=$ {DT_LABEL[TIME_PERIOD]}, $\nu={skewt_cfg.nu:.3f}$)",
    fontsize=11,
)

for row, col, opt_type, dist, title in subplot_cfg:
    ax = axes2[row][col]
    w_var = weights[(dist, opt_type, "variance")]
    w_cfv2 = weights[(dist, opt_type, "cfvar2")]
    w_cfv3 = weights[(dist, opt_type, "cfvar3")]

    pct2 = _pct_change(w_cfv2, w_var)
    pct3 = _pct_change(w_cfv3, w_var)

    b1 = ax.bar(
        x_pos - bar_width / 2,
        pct2,
        bar_width,
        label=r"CFVaR2 % $\Delta$",
        color="orange",
        alpha=0.8,
    )
    b2 = ax.bar(
        x_pos + bar_width / 2,
        pct3,
        bar_width,
        label=r"CFVaR3 % $\Delta$",
        color="green",
        alpha=0.8,
    )

    ax.bar_label(b1, fmt=BAR_FORMAT, padding=2, fontsize=6)
    ax.bar_label(b2, fmt=BAR_FORMAT, padding=2, fontsize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STOCK_NAMES)
    ax.set_xlabel("Underlying Asset")
    ax.set_ylabel("% Change from Variance Weight")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

plt.tight_layout()
out2 = SCRIPT_DIR / (
    dot_to_p(f"skew_vs_t_pct_change_alpha{TAIL_RISK}_dt{TIME_PERIOD:.4f}") + ".png"
)
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2.name}")
plt.close(fig2)
