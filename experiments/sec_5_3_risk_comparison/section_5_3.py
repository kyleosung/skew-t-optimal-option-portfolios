"""
Section 5.4 - CFVaR3 Comparison and Impact of Neglecting Skewness.

Evaluates the CFVaR3 risk of the optimal Variance, CFVaR2, and CFVaR3
portfolios under both the skew-t and standard-t distributions.

The script computes three groups of results for each (TAIL_RISK,
TIME_PERIOD, option_type) combination:

1.  **Skew-t CFVaR3 at skew-t optimal portfolios** - the CFVaR3 value
    evaluated under the skew-t distribution at portfolios optimised with
    the skew-t model (from Section 5.1).
2.  **Standard-t CFVaR3 at standard-t optimal portfolios** - the CFVaR3
    value evaluated under the standard-t distribution at portfolios
    optimised with the standard-t model (from Section 5.2).
3.  **Skew-t CFVaR3 at standard-t optimal portfolios** - the CFVaR3
    value evaluated under the *skew-t* distribution at portfolios
    optimised with the *standard-t* model.  Comparing (3) with (1) shows
    the risk increase due to neglecting skewness.

Outputs
-------
- ``section_5_4_cfvar3_comparison.csv`` - all CFVaR3 values across every
  (TAIL_RISK, TIME_PERIOD, option_type) combination, together with the
  percentage increase in risk of the standard-t optimal CFVaR3 portfolio
  (evaluated under skew-t) relative to the skew-t optimal CFVaR3 portfolio
  (rounded to six decimal places).
- ``tex/section_5_4_main_setting_table.tex`` - LaTeX table restricted to the
  main paper setting (alpha=0.01, dt=1/252, r_f=0.05).
- ``../sec_5_2_numeric/skew_vs_t_main_setting_table.tex`` - LaTeX table
  (Section 5.2's "Comparison of Portfolio Risk" table) restricted to the
  main paper setting (alpha=0.01, dt=1/252, r_f=0.05), reporting the CFVaR3
  value of the optimal Variance, CFVaR2, and CFVaR3 portfolios under each
  distribution, and the percentage increase in CFVaR3 risk of the Variance-
  and CFVaR2-optimal portfolios relative to the CFVaR3-optimal portfolio
  (evaluated under their own distribution).

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiments/paper1/sec_5_3_risk_comparison/section_5_4.py
"""

import csv
from pathlib import Path

import numpy as np

# >>> ONE-LINE DATASET SWITCH: change the imported configs to use different parameters.
# Available skew-t: DEFAULT_SKEWT_CONFIG (=GAUSSIAN_GARCH_SKEWT_CONFIG),
#                   T_GARCH_SKEWT_CONFIG, SKEWT_GARCH_SKEWT_CONFIG
# Available t:      HK_T_CONFIG, GAUSSIAN_GARCH_T_CONFIG, T_GARCH_T_CONFIG, SKEWT_GARCH_T_CONFIG
from src.constants.dataset_config import DEFAULT_SKEWT_CONFIG as skewt_cfg
from src.constants.dataset_config import HK_T_CONFIG as t_cfg
from src.options.gosset_call import GossetCall
from src.options.gosset_put import GossetPut
from src.options.option import MarketEnvironment, Underlying
from src.options.skew_t_gosset_call import SkewTGossetCall
from src.options.skew_t_gosset_put import SkewTGossetPut
from src.portfolio.option_portfolio import OptionPortfolio
from src.portfolio.skew_t_option_portfolio import SkewTOptionPortfolio
from src.portfolio.utils import (
    build_scale_matrix,
    number_of_shares_to_weights,
    weights_to_number_of_shares,
)
from src.utils.skew_t_distribution import (
    compute_alpha_1d_from_omega,
    compute_omega_delta_s,
)

SCRIPT_DIR = Path(__file__).resolve().parent

RISK_FREE_RATE = 0.05
TIME_TO_MATURITY = 1.0
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


def _pct_increase_in_risk(value: float, cfvar3_value: float) -> float:
    """Percentage increase in CFVaR3 risk of ``value`` vs. ``cfvar3_value``."""
    return round((value - cfvar3_value) / cfvar3_value * 100, 6)


def _write_latex_table(
    path: Path,
    caption: str,
    label: str,
    headers: list,
    rows: list,
) -> None:
    """Write a basic LaTeX ``table``/``tabular`` environment (booktabs style) to ``path``."""
    col_spec = "l" * len(headers)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


# ======================================================================
# Main computation loop
# ======================================================================
csv_rows: list[dict] = []

TAIL_RISK = 0.01
TIME_PERIOD = 1 / 252

for opt_type in ("call", "put"):
    # ----------------------------------------------------------
    # Build portfolios
    # ----------------------------------------------------------
    pf_sk = _build_skew_t_portfolio(opt_type, skewt_cfg.nu, TIME_PERIOD)
    pf_t = _build_t_portfolio(opt_type, skewt_cfg.nu, TIME_PERIOD)

    # ----------------------------------------------------------
    # Optimize under skew-t
    # ----------------------------------------------------------
    x_var_sk, _ = pf_sk.get_optimal_variance_weights_lagrange()
    x_cfv2_sk, _ = pf_sk.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    x_cfv3_sk, _ = pf_sk.get_optimal_CFVaR3_weights_numeric(TAIL_RISK)

    # ----------------------------------------------------------
    # Optimize under standard-t
    # ----------------------------------------------------------
    x_var_t, _ = pf_t.get_optimal_variance_weights_lagrange()
    x_cfv2_t, _ = pf_t.get_optimal_CFVaR2_weights_lagrange(TAIL_RISK)
    x_cfv3_t, _ = pf_t.get_optimal_CFVaR3_weights_numeric(TAIL_RISK)

    # ----------------------------------------------------------
    # Group 1: Skew-t CFVaR3 at skew-t optimal portfolios
    # ----------------------------------------------------------
    sk_at_sk_var = pf_sk.get_CFVaR3_portfolio_loss(x_var_sk, TAIL_RISK)
    sk_at_sk_cfv2 = pf_sk.get_CFVaR3_portfolio_loss(x_cfv2_sk, TAIL_RISK)
    sk_at_sk_cfv3 = pf_sk.get_CFVaR3_portfolio_loss(x_cfv3_sk, TAIL_RISK)

    # ----------------------------------------------------------
    # Group 2: Standard-t CFVaR3 at standard-t optimal portfolios
    # ----------------------------------------------------------
    t_at_t_var = pf_t.get_CFVaR3_portfolio_loss(x_var_t, TAIL_RISK)
    t_at_t_cfv2 = pf_t.get_CFVaR3_portfolio_loss(x_cfv2_t, TAIL_RISK)
    t_at_t_cfv3 = pf_t.get_CFVaR3_portfolio_loss(x_cfv3_t, TAIL_RISK)

    # ----------------------------------------------------------
    # Group 3: Skew-t CFVaR3 at standard-t optimal portfolios
    # (risk increase due to neglecting skewness)
    # ----------------------------------------------------------
    # The standard-t optimal share counts are computed against the
    # standard-t option prices (pf_t.option_values). To evaluate the
    # *same* portfolio (i.e. the same fractional allocation of the
    # $1 budget) under the skew-t pricing model, we must first
    # convert to weights (which are price-model independent) and
    # then back to shares using the skew-t option prices
    # (pf_sk.option_values). Reusing the standard-t share counts
    # directly would silently violate the unit-budget constraint
    # under the skew-t prices.
    w_var_t = number_of_shares_to_weights(x_var_t, pf_t.option_values)
    w_cfv2_t = number_of_shares_to_weights(x_cfv2_t, pf_t.option_values)
    w_cfv3_t = number_of_shares_to_weights(x_cfv3_t, pf_t.option_values)

    x_var_t_at_sk_prices = weights_to_number_of_shares(w_var_t, pf_sk.option_values)
    x_cfv2_t_at_sk_prices = weights_to_number_of_shares(w_cfv2_t, pf_sk.option_values)
    x_cfv3_t_at_sk_prices = weights_to_number_of_shares(w_cfv3_t, pf_sk.option_values)

    sk_at_t_var = pf_sk.get_CFVaR3_portfolio_loss(x_var_t_at_sk_prices, TAIL_RISK)
    sk_at_t_cfv2 = pf_sk.get_CFVaR3_portfolio_loss(x_cfv2_t_at_sk_prices, TAIL_RISK)
    sk_at_t_cfv3 = pf_sk.get_CFVaR3_portfolio_loss(x_cfv3_t_at_sk_prices, TAIL_RISK)

    # Percentage increase: skew-t CFVaR3 at t-optimal vs skew-t-optimal
    csv_rows.append(
        {
            "tail_risk": TAIL_RISK,
            "time_period": TIME_PERIOD,
            "option_type": opt_type,
            "skewt_cfvar3_at_skewt_var": sk_at_sk_var,
            "skewt_cfvar3_at_skewt_cfvar2": sk_at_sk_cfv2,
            "skewt_cfvar3_at_skewt_cfvar3": sk_at_sk_cfv3,
            "t_cfvar3_at_t_var": t_at_t_var,
            "t_cfvar3_at_t_cfvar2": t_at_t_cfv2,
            "t_cfvar3_at_t_cfvar3": t_at_t_cfv3,
            "skewt_cfvar3_at_t_var": sk_at_t_var,
            "skewt_cfvar3_at_t_cfvar2": sk_at_t_cfv2,
            "skewt_cfvar3_at_t_cfvar3": sk_at_t_cfv3,
            "pct_increase_risk_variance_vs_cfvar3_skewt": _pct_increase_in_risk(
                sk_at_sk_var, sk_at_sk_cfv3
            ),
            "pct_increase_risk_cfvar2_vs_cfvar3_skewt": _pct_increase_in_risk(
                sk_at_sk_cfv2, sk_at_sk_cfv3
            ),
            "pct_increase_risk_variance_vs_cfvar3_t": _pct_increase_in_risk(
                t_at_t_var, t_at_t_cfv3
            ),
            "pct_increase_risk_cfvar2_vs_cfvar3_t": _pct_increase_in_risk(
                t_at_t_cfv2, t_at_t_cfv3
            ),
            "pct_increase_neglecting_skewness_cfvar3": _pct_increase_in_risk(
                sk_at_t_cfv3, sk_at_sk_cfv3
            ),
        }
    )

    print(
        f"[alpha={TAIL_RISK}, dt={DT_LABEL[TIME_PERIOD]}, {opt_type}] "
        f"skew-t CFVaR3 at skew-t optimal CFVaR3: {sk_at_sk_cfv3:.6f}, "
        f"skew-t CFVaR3 at t-optimal CFVaR3: {sk_at_t_cfv3:.6f}, "
        f"pct increase: "
        f"{_pct_increase_in_risk(sk_at_t_cfv3, sk_at_sk_cfv3):.6f}%"
    )

# ======================================================================
# CSV output
# ======================================================================
FIELDNAMES = [
    "tail_risk",
    "time_period",
    "option_type",
    "skewt_cfvar3_at_skewt_var",
    "skewt_cfvar3_at_skewt_cfvar2",
    "skewt_cfvar3_at_skewt_cfvar3",
    "t_cfvar3_at_t_var",
    "t_cfvar3_at_t_cfvar2",
    "t_cfvar3_at_t_cfvar3",
    "skewt_cfvar3_at_t_var",
    "skewt_cfvar3_at_t_cfvar2",
    "skewt_cfvar3_at_t_cfvar3",
    "pct_increase_risk_variance_vs_cfvar3_skewt",
    "pct_increase_risk_cfvar2_vs_cfvar3_skewt",
    "pct_increase_risk_variance_vs_cfvar3_t",
    "pct_increase_risk_cfvar2_vs_cfvar3_t",
    "pct_increase_neglecting_skewness_cfvar3",
]

csv_out_path = SCRIPT_DIR / "section_5_4_cfvar3_comparison.csv"
with open(csv_out_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nSaved: {csv_out_path.name}")

# ======================================================================
# LaTeX table for the main paper setting (alpha=0.01, dt=1/252, r_f=0.05)
# ======================================================================
MAIN_TAIL_RISK = 0.01
MAIN_TIME_PERIOD = 1 / 252

main_rows = [
    row
    for row in csv_rows
    if row["tail_risk"] == MAIN_TAIL_RISK
    and abs(row["time_period"] - MAIN_TIME_PERIOD) < 1e-12
]

table_rows = [
    [
        row["option_type"].capitalize(),
        f"{row['skewt_cfvar3_at_skewt_cfvar3']:.6f}",
        f"{row['t_cfvar3_at_t_cfvar3']:.6f}",
        f"{row['skewt_cfvar3_at_t_cfvar3']:.6f}",
        f"{row['pct_increase_neglecting_skewness_cfvar3']:.6f}",
    ]
    for row in main_rows
]

tex_out_path = SCRIPT_DIR / "section_5_4_main_setting_table.tex"
_write_latex_table(
    tex_out_path,
    caption=(
        r"CFVaR3 value of the optimal CFVaR3 portfolio under the skew-t model, "
        r"the standard-t model, and the standard-t portfolio evaluated with the "
        r"skew-t CFVaR3 formula.  The final column shows the percentage increase "
        r"in risk due to neglecting skewness "
        rf"($\alpha={MAIN_TAIL_RISK}$, $\Delta t = 1/252$, $r_f=0.05$)."
    ),
    label="tab:section_5_4_main_setting",
    headers=[
        "Type",
        r"Skew-$t$ Opt.",
        r"$t$ Opt.",
        r"$t$ Opt. (Skew-$t$ eval.)",
        r"\% Incr. (Skewness)",
    ],
    rows=table_rows,
)
print(f"Saved: {tex_out_path.name}")

# ======================================================================
# LaTeX table for Section 5.2's "Comparison of Portfolio Risk" table
# (alpha=0.01, dt=1/252, r_f=0.05)
# ======================================================================
skew_vs_t_rows = []
for row in main_rows:
    skew_vs_t_rows.append(
        [
            row["option_type"].capitalize(),
            "Skew-t",
            f"{row['skewt_cfvar3_at_skewt_var']:.6f}",
            f"{row['skewt_cfvar3_at_skewt_cfvar2']:.6f}",
            f"{row['skewt_cfvar3_at_skewt_cfvar3']:.6f}",
            f"{row['pct_increase_risk_variance_vs_cfvar3_skewt']:.6f}",
            f"{row['pct_increase_risk_cfvar2_vs_cfvar3_skewt']:.6f}",
        ]
    )
    skew_vs_t_rows.append(
        [
            row["option_type"].capitalize(),
            "Standard t",
            f"{row['t_cfvar3_at_t_var']:.6f}",
            f"{row['t_cfvar3_at_t_cfvar2']:.6f}",
            f"{row['t_cfvar3_at_t_cfvar3']:.6f}",
            f"{row['pct_increase_risk_variance_vs_cfvar3_t']:.6f}",
            f"{row['pct_increase_risk_cfvar2_vs_cfvar3_t']:.6f}",
        ]
    )

skew_vs_t_tex_out_path = (
    SCRIPT_DIR / ".." / "sec_5_2_numeric" / "skew_vs_t_main_setting_table.tex"
)
_write_latex_table(
    skew_vs_t_tex_out_path,
    caption=(
        r"CFVaR3 value of the Optimal Variance, CFVaR2, and CFVaR3 portfolios "
        r"under the skew-t distribution, and the percentage increase in CFVaR3 "
        r"risk of the Optimal Variance and Optimal CFVaR2 portfolios relative "
        r"to the Optimal CFVaR3 portfolio "
        rf"($\alpha={MAIN_TAIL_RISK}$, $\Delta t = 1/252$, $r_f=0.05$)."
    ),
    label="tab:skew_vs_t_main_setting",
    headers=[
        "Option Type",
        "Distribution",
        "Variance",
        "CFVaR2",
        "CFVaR3",
        r"Variance \% Incr.",
        r"CFVaR2 \% Incr.",
    ],
    rows=skew_vs_t_rows,
)
print(f"Saved: {skew_vs_t_tex_out_path.resolve().name}")
