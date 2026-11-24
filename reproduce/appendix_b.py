"""
Appendix B: Monte Carlo check of the option prices of the skew-t pricing model.

The prices of the at-the-money calls and puts of Section 5 (the five stocks) are compared
with their Monte Carlo estimates.

Run from the repository root with ``Rscript`` on the PATH::

    export PYTHONPATH=$(pwd)
    python reproduce/appendix_b.py

Writes the tables to ``output/generated/``.
"""

from src.estimation import fit_assets, load_prices
from src.experiments import pricing_check
from src.settings import (
    BURN_IN,
    DATA_DIR,
    FIVE_STOCKS,
    OUTPUT_DIR,
    SAMPLE_END,
    SAMPLE_START,
)
from src.tables import write_pricing_table

if __name__ == "__main__":
    prices = load_prices(DATA_DIR / "five_stock_prices.csv", FIVE_STOCKS)
    fit = fit_assets(prices, SAMPLE_START, SAMPLE_END, BURN_IN)
    write_pricing_table(pricing_check(fit), OUTPUT_DIR / "generated")
