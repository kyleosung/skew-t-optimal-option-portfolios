"""
Section 5.1 - Skew-t: Variance & CFVaR2 Analytic Optimization.

Produces one 2 by 2 figure per (TAIL_RISK, TIME_PERIOD) combination showing
analytic (Lagrange) variance-optimal and CFVaR2-optimal portfolio weights
under the AC skew-t distribution.

Figure layout (2 by 2):
  (0,0) Skew-Gosset Calls  - Variance-Optimal Weights (analytic)
  (0,1) Skew-Gosset Calls  - CFVaR2-Optimal Weights   (analytic)
  (1,0) Skew-Gosset Puts   - Variance-Optimal Weights (analytic)
  (1,1) Skew-Gosset Puts   - CFVaR2-Optimal Weights   (analytic)

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiments/paper1/sec_5_1_analytic/section_5_1_analytic.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# >>> ONE-LINE DATASET SWITCH: change the imported config to use different parameters.
# Available: HK_NORMAL_CONFIG, HK_T_CONFIG, DEFAULT_SKEWT_CONFIG (=GAUSSIAN_GARCH_SKEWT_CONFIG),
#            GAUSSIAN_GARCH_T_CONFIG, T_GARCH_SKEWT_CONFIG, T_GARCH_T_CONFIG,
#            SKEWT_GARCH_SKEWT_CONFIG, SKEWT_GARCH_T_CONFIG
from src.constants.dataset_config import DEFAULT_SKEWT_CONFIG as cfg
from src.constants.plotting import SEABORN_CONTEXT, SEABORN_PALETTE, SEABORN_STYLE
from src.options.option import MarketEnvironment, Underlying
from src.options.skew_t_gosset_call import SkewTGossetCall
from src.options.skew_t_gosset_put import SkewTGossetPut
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

SCRIPT_DIR = Path(__file__).resolve().parent / "figures"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.05
TIME_TO_MATURITY = 1.0
BAR_FORMAT = "%.3f"
DT_LABEL = {1 / 252: "1 day", 1 / 52: "1 week", 1 / 12: "1 month"}

STOCK_NAMES = list(cfg.asset_tickers)
stock_prices = cfg.stock_prices.tolist()
volatilities = cfg.annual_vol.tolist()


def _build_skew_t_portfolio(
    option_type: str,
    nu: float,
    time_period: float,
) -> SkewTOptionPortfolio:
    """Build an ATM skew-t Gosset option portfolio using AC estimated parameters."""
    alpha_1d = compute_alpha_1d_from_omega(cfg.omega, cfg.sigma_dp)
    omega_delta_s = compute_omega_delta_s(
        omega_dp=cfg.omega,
        sigma_dp=cfg.sigma_dp,
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
        correlation=cfg.corr,
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
        returns=cfg.mean_annual,
    )


TAIL_RISK = 0.01
TIME_PERIOD = 1 / 252

weights = {}

for opt_type in ("call", "put"):
    portfolio = _build_skew_t_portfolio(opt_type, cfg.nu, TIME_PERIOD)
    _, w_var = portfolio.get_optimal_variance_weights_lagrange()
    _, w_cfv = portfolio.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    weights[(opt_type, "variance")] = w_var.ravel()
    weights[(opt_type, "cfvar2")] = w_cfv.ravel()

n_assets = len(STOCK_NAMES)
bar_width = 0.35
x_pos = np.arange(n_assets)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)

suptitle = (
    rf"Skew-t: Variance & CFVaR2 Analytic Optimization"
    "\n"
    rf"($\alpha={TAIL_RISK}$, $\Delta t=$ {DT_LABEL[TIME_PERIOD]}, $\nu={cfg.nu:.3f}$)"
)

subplot_cfg = [
    (
        0,
        0,
        "call",
        "variance",
        "Skew-Gosset Calls: Variance-Optimal Weights",
        "steelblue",
    ),
    (
        0,
        1,
        "call",
        "cfvar2",
        rf"Skew-Gosset Calls: CFVaR2-Optimal Weights ($\alpha={TAIL_RISK}$)",
        "steelblue",
    ),
    (
        1,
        0,
        "put",
        "variance",
        "Skew-Gosset Puts: Variance-Optimal Weights",
        "orange",
    ),
    (
        1,
        1,
        "put",
        "cfvar2",
        rf"Skew-Gosset Puts: CFVaR2-Optimal Weights ($\alpha={TAIL_RISK}$)",
        "orange",
    ),
]

for row, col, opt_type, obj, title, color in subplot_cfg:
    ax = axes[row][col]
    w = weights[(opt_type, obj)]
    label = rf"Skew $t$ ($\nu={cfg.nu:.3f}$)"
    bars = ax.bar(x_pos, w, bar_width, label=label, color=color, alpha=0.8)
    ax.bar_label(bars, fmt=BAR_FORMAT, padding=3, fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STOCK_NAMES)
    ax.set_xlabel("Underlying Asset")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title(title)
    # ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

plt.suptitle(suptitle, fontsize=11)
plt.tight_layout()

out_path = SCRIPT_DIR / (
    dot_to_p(f"skewt_analytic_alpha{TAIL_RISK}_dt{TIME_PERIOD:.4f}") + ".png"
)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path.name}")
plt.close()
