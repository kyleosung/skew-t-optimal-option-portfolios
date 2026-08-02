"""
Download real stock data for the five-stock dataset (Hu and Kercheval 2010).

Downloads adjusted daily closing prices for DIS, XOM, PFE, MO, INTC
from January 7, 2002 to April 8, 2005 via yfinance, and saves to CSV.

The resulting CSV is used by the estimation scripts to fit Student t
and skew t distributions.

Output:
    examples/five_stock_prices.csv
"""

from pathlib import Path

TICKERS = ["DIS", "XOM", "PFE", "MO", "INTC"]
START_DATE = "2002-01-07"
END_DATE = "2005-09-05"  # inclusive of April 8

OUTPUT_CSV = Path(__file__).resolve().parent / "five_stock_prices.csv"


def download_stock_data(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    output_path=OUTPUT_CSV,
):
    """
    Download adjusted daily close prices via yfinance and save to CSV.

    Parameters
    ----------
    tickers : list[str]
        Stock ticker symbols.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format (exclusive in yfinance).
    output_path : Path or str
        Path to save the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.
    """
    import pandas as pd
    import yfinance as yf

    print(f"Downloading {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # yfinance returns MultiIndex columns (Price, Ticker) for multi-ticker
    if isinstance(data.columns, pd.MultiIndex):  # type: ignore
        prices = data["Close"][tickers]  # type: ignore
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})  # type: ignore

    prices = prices.dropna()

    output_path = Path(output_path)
    prices.to_csv(output_path)
    print(f"  Saved {len(prices)} trading days to {output_path}")
    return prices


if __name__ == "__main__":
    download_stock_data()
    print("\nDone. The CSV can now be committed to the repository.")
