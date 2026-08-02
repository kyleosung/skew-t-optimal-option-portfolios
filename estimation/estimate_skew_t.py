r"""
Perform parameter estimation using minimally new code.

This requires use of R to estimate the parameters using the Azzalini and Capitanio skew t characterization.

The sample period in \citep{Hu01012010} appears to be ambiguous: Section 4 refers to ``\textit{daily closing prices
for the period 7 January 2002 to 8 April 2005}'' and Section 5 states to ``\textit{Suppose we are standing at 4 August
2005, the last date in our data set, and the holding period is one day. 750 sample data are used in the estimation}''.
These statements are most naturally reconciled using the interpretation from Section 5, assuming that the statement in
Section 4 is a typo from reading month/day/year. There are 820 trading days between 7 January 2002 to 8 April 2005 and
there are 781 trading days between July 1 2002 and 4 August 2005.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from experiment_t_vs_skew_t_estimation.tex_utils import json_to_tex_table

from src.constants.five_option_dataset import ASSET_TICKERS


def main():
    """
    Perform parameter estimation using minimally new code.

    This requires use of R to estimate the parameters using the Azzalini and Capitanio skew t characterization.
    """
    repo_dir = Path(__file__).resolve().parent.parent
    data_dir = repo_dir / "data"

    csv_path = data_dir / "five_stock_prices.csv"
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices = prices[ASSET_TICKERS]

    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")  # type: ignore
    # Hu's thesis: adjusted close from 2002-07-01 to 2005-08-05, "most recent 750 returns"
    log_returns = log_returns.loc["2002-07-01":"2005-08-05"].tail(750)

    print(f"Log returns: {log_returns.shape[0]} observations")
    print(
        f"Date range: {log_returns.index.min().date()} to {log_returns.index.max().date()}"
    )

    filtered = pd.DataFrame(
        index=log_returns.index, columns=log_returns.columns, dtype=float
    )
    garch_vol_1d = {}

    for col in log_returns.columns:
        # Hu-Kercheval use Gaussian GARCH(1,1) filtering.
        # Rescale to percentage returns to improve optimizer convergence
        # (avoids DataScaleWarning).  Standardized residuals are invariant
        # to this linear rescaling.
        am = arch_model(
            log_returns[col].astype(float) * 100.0,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
        )
        res = am.fit(disp="off")

        # standardized residuals = filtered returns
        filtered[col] = res.std_resid

        # optional: one-step-ahead forecasted volatility (convert back from %)
        fcast = res.forecast(horizon=1)
        garch_vol_1d[col] = float(np.sqrt(fcast.variance.iloc[-1, 0]) / 100.0)  # type: ignore

    workdir = Path(__file__).resolve().parent / "estimation_results"
    workdir.mkdir(exist_ok=True)
    filtered_csv = workdir / "filtered_returns.csv"
    out_json = workdir / "skew_t_ac_fit.json"

    script_path = Path(__file__).parent / "fit_ac_skewt.R"

    filtered.to_csv(filtered_csv, index=False)

    subprocess.run(
        ["Rscript", str(script_path), str(filtered_csv), str(out_json)],
        check=True,
    )

    with open(out_json, "r") as f:
        fit = json.load(f)

    print(fit.keys())  # e.g. dp, cp, logL
    print(fit["dp"])  # direct parameters

    # Generate LaTeX table
    tex_table = json_to_tex_table(out_json, digits=5)
    print(tex_table)

    with open(workdir / "skew_t_fit_table.tex", "w") as f:
        f.write(tex_table)


if __name__ == "__main__":
    main()
