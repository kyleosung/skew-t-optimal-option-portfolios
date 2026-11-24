"""
Download the closing prices of the paper with yfinance, adjusted for dividends and splits.

The five stocks DIS, XOM, PFE, MO and INTC (Hu and Kercheval, 2010) cover the estimation
sample of Section 4 and the days of the out-of-sample test of Section 5.3 that follow it.

Adjusted prices change with every later dividend, so the committed file is the one
the paper is computed from; rerun this script only to refresh it.

Output:
    data/five_stock_prices.csv
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent

# (tickers, first day, last day, file); the days are inclusive
DOWNLOADS = (
    (
        ["DIS", "XOM", "PFE", "MO", "INTC"],
        "2002-07-01",
        "2005-09-30",
        "five_stock_prices",
    ),
)

if __name__ == "__main__":
    for tickers, first, last, name in DOWNLOADS:
        data = yf.download(
            tickers,
            start=first,
            end=(pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        data["Close"][tickers].to_csv(DATA_DIR / f"{name}.csv")
