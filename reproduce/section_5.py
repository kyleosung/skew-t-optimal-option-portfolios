"""
Sections 4 and 5: the five stocks DIS, XOM, PFE, MO and INTC.

Estimation (Section 4), optimal weights (Section 5.1), comparison of the risk of the
optimal portfolios (Section 5.2), out-of-sample analysis (Section 5.3) and Monte Carlo
evaluation (Section 5.4).

Run from the repository root with ``Rscript`` on the PATH::

    export PYTHONPATH=$(pwd)
    python reproduce/section_5.py

Writes the tables to ``output/generated/`` and the figures to ``output/images_updated/``.
"""

from src.estimation import fit_assets, load_prices
from src.experiments import KINDS, monte_carlo, out_of_sample, risk_comparison
from src.figures import weight_figures
from src.settings import (
    BURN_IN,
    DATA_DIR,
    FIVE_STOCKS,
    NU_SENSITIVITY,
    OOS_NUM_DAYS,
    OUTPUT_DIR,
    SAMPLE_END,
    SAMPLE_START,
    TAIL_RISK,
)
from src.tables import (
    write_fit_tables,
    write_monte_carlo_table,
    write_out_of_sample_table,
    write_risk_tables,
)

if __name__ == "__main__":
    prices = load_prices(DATA_DIR / "five_stock_prices.csv", FIVE_STOCKS)
    fit = fit_assets(prices, SAMPLE_START, SAMPLE_END, BURN_IN)
    write_fit_tables(fit, OUTPUT_DIR / "generated")

    comparison = {kind: risk_comparison(fit, kind, TAIL_RISK) for kind in KINDS}
    weight_figures(
        {kind: weights for kind, (weights, _) in comparison.items()},
        OUTPUT_DIR / "images_updated",
    )
    write_risk_tables(
        {kind: risks for kind, (_, risks) in comparison.items()},
        OUTPUT_DIR / "generated",
    )

    pnl = out_of_sample(fit, prices, OOS_NUM_DAYS, "call", TAIL_RISK)
    write_out_of_sample_table(pnl, TAIL_RISK, OUTPUT_DIR / "generated")

    write_monte_carlo_table(monte_carlo(fit, TAIL_RISK), OUTPUT_DIR / "generated")
    write_monte_carlo_table(
        monte_carlo(fit, TAIL_RISK, nu=NU_SENSITIVITY),
        OUTPUT_DIR / "generated",
        "nu_sensitivity_",
    )
