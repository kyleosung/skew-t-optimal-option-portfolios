"""
Section 5.1: Skew-t: Variance, CFVaR2 & CFVaR3 Numeric Optimization.

Produces two figures per (TAIL_RISK, TIME_PERIOD) combination:

Figure 1 - Portfolio Weights (2 by 1 grid) with all three objectives
(Variance, CFVaR2, CFVaR3) shown as grouped bars per asset.

Figure 2 - % Change from Variance Weights (2 by 1 grid) showing the
percentage change of CFVaR2 and CFVaR3 weights relative to the
variance-optimal baseline.

Figure 1 layout (2 by 1):
  (0) Skew-Gosset Calls : Variance / CFVaR2 / CFVaR3 grouped bars
  (1) Skew-Gosset Puts  : Variance / CFVaR2 / CFVaR3 grouped bars

Figure 2 layout (2 by 1):
  (0) Skew-Gosset Calls : CFVaR2 & CFVaR3 % Change from Variance
  (1) Skew-Gosset Puts  : CFVaR2 & CFVaR3 % Change from Variance

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiments/paper1/sec_5_2_numeric/section_5_1_numeric.py
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

assert cfg

STOCK_NAMES = list(cfg.asset_tickers)
stock_prices = cfg.stock_prices.tolist()
volatilities = cfg.annual_vol.tolist()


def _build_skew_t_portfolio(
    option_type: str,
    nu: float,
    time_period: float,
) -> SkewTOptionPortfolio:
    r"""Build an ATM skew-t Gosset option portfolio with the given \nu."""
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


def _pct_change(w_new: np.ndarray, w_base: np.ndarray) -> np.ndarray:
    """Percentage change relative to |w_base|, zero-safe."""
    return np.where(
        np.abs(w_base) < 1e-10, 0.0, (w_new - w_base) / np.abs(w_base) * 100
    )


TAIL_RISK = 0.01
TIME_PERIOD = 1 / 252

weights = {}

for opt_type in ("call", "put"):
    portfolio = _build_skew_t_portfolio(opt_type, cfg.nu, TIME_PERIOD)
    _, w_var = portfolio.get_optimal_variance_weights_lagrange()
    _, w_cfv2 = portfolio.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    _, w_cfv3 = portfolio.get_optimal_CFVaR3_weights_numeric(TAIL_RISK)
    weights[(opt_type, "variance")] = w_var.ravel()
    weights[(opt_type, "cfvar2")] = w_cfv2.ravel()
    weights[(opt_type, "cfvar3")] = w_cfv3.ravel()

n_assets = len(STOCK_NAMES)
bar_width = 0.25
x_pos = np.arange(n_assets)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=False)

suptitle = (
    rf"Skew-t: Variance, CFVaR2 & CFVaR3 Numeric Optimization"
    "\n"
    rf"($\alpha={TAIL_RISK}$, $\Delta t=$ {DT_LABEL[TIME_PERIOD]}, $\nu={cfg.nu:.3f}$)"
)

weights_fig1_cfg = [
    (
        0,
        "call",
        "Skew-Gosset Calls: Variance / CFVaR2 / CFVaR3 Optimal Weights",
    ),
    (1, "put", "Skew-Gosset Puts: Variance / CFVaR2 / CFVaR3 Optimal Weights"),
]

for row, opt_type, title in weights_fig1_cfg:
    ax = axes[row]
    w_var = weights[(opt_type, "variance")]
    w_cfv2 = weights[(opt_type, "cfvar2")]
    w_cfv3 = weights[(opt_type, "cfvar3")]

    b1 = ax.bar(
        x_pos - bar_width,
        w_var,
        bar_width,
        label="Variance",
        color="steelblue",
        alpha=0.8,
    )
    b2 = ax.bar(
        x_pos,
        w_cfv2,
        bar_width,
        label=rf"CFVaR2 ($\alpha={TAIL_RISK}$)",
        color="orange",
        alpha=0.8,
    )
    b3 = ax.bar(
        x_pos + bar_width,
        w_cfv3,
        bar_width,
        label=rf"CFVaR3 ($\alpha={TAIL_RISK}$)",
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

plt.suptitle(suptitle, fontsize=11)
plt.tight_layout()

out_path = SCRIPT_DIR / (
    dot_to_p(f"skewt_numeric_alpha{TAIL_RISK}_dt{TIME_PERIOD:.4f}") + ".png"
)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path.name}")
plt.close()

# ------------------------------------------------------------------
# Figure 2: % Change from Variance Weights
# ------------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 9), sharey=False)

suptitle2 = (
    rf"Skew-t: CFVaR2 & CFVaR3 % Change from Variance"
    "\n"
    rf"($\alpha={TAIL_RISK}$, $\Delta t=$ {DT_LABEL[TIME_PERIOD]}, $\nu={cfg.nu:.3f}$)"
)

pct_cfg = [
    (0, "call", "Skew-Gosset Calls: % Change from Variance Baseline"),
    (1, "put", "Skew-Gosset Puts: % Change from Variance Baseline"),
]

for row, opt_type, title in pct_cfg:
    ax = axes2[row]
    w_var = weights[(opt_type, "variance")]
    w_cfv2 = weights[(opt_type, "cfvar2")]
    w_cfv3 = weights[(opt_type, "cfvar3")]

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

fig2.suptitle(suptitle2, fontsize=11)
fig2.tight_layout()

out_path2 = SCRIPT_DIR / (
    dot_to_p(f"skewt_numeric_pct_change_alpha{TAIL_RISK}_dt{TIME_PERIOD:.4f}") + ".png"
)
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2.name}")
plt.close(fig2)
