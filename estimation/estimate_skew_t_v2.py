r"""
Improved parameter estimation for the AC skew-t model (v2).

Changes from the original ``estimate_skew_t.py``:

1. **Configurable GARCH(1,1) innovations** -- supports ``"normal"``,
   ``"t"``, or ``"skewt"`` via the ``--dist`` flag.  Fat-tailed
   innovations let the GARCH filter absorb univariate tail heaviness,
   so the standardized residuals better reflect cross-sectional
   dependence rather than also absorbing marginal tail effects.

2. **GARCH convergence validation** -- checks ``res.convergence_flag``
   and prints a warning if the optimizer did not converge.  Also
   reports GARCH parameters (omega, alpha, beta) and persistence for
   each asset.

3. **Outputs are renamed** with a ``_v2`` suffix so that the original
   parameter set is preserved alongside the new one.

The R fitting step is unchanged -- ``fit_ac_skewt.R`` receives the
(better-filtered) residuals and produces ``skew_t_ac_fit_v2.json``.

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiment_t_vs_skew_t_estimation/estimate_skew_t_v2.py
    python experiment_t_vs_skew_t_estimation/estimate_skew_t_v2.py --dist skewt
    python experiment_t_vs_skew_t_estimation/estimate_skew_t_v2.py --dist normal
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from experiment_t_vs_skew_t_estimation.tex_utils import json_to_tex_table

from src.constants.five_option_dataset import ASSET_TICKERS

VALID_GARCH_DISTS = ("normal", "t", "skewt")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AC skew-t parameter estimation (v2) with configurable GARCH."
    )
    parser.add_argument(
        "--dist",
        choices=VALID_GARCH_DISTS,
        default="t",
        help=(
            "GARCH(1,1) innovation distribution. "
            "Options: 'normal', 't' (default), 'skewt'."
        ),
    )
    return parser.parse_args()


def main(garch_dist: str = "t"):
    """
    Perform improved parameter estimation (v2) with configurable GARCH filtering.

    Parameters
    ----------
    garch_dist : str
        GARCH innovation distribution: "normal", "t", or "skewt".

    This requires use of R to estimate the parameters using the Azzalini
    and Capitanio skew t characterization.
    """
    if garch_dist not in VALID_GARCH_DISTS:
        raise ValueError(
            f"Invalid GARCH dist '{garch_dist}'. Must be one of {VALID_GARCH_DISTS}."
        )

    repo_dir = Path(__file__).resolve().parent.parent
    data_dir = repo_dir / "data"

    csv_path = data_dir / "five_stock_prices.csv"
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices = prices[ASSET_TICKERS]

    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
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

    print(f"\nGARCH(1,1) filtering with dist='{garch_dist}' innovations:")
    print("-" * 65)

    for col in log_returns.columns:
        # Rescale to percentage returns to avoid DataScaleWarning and
        # improve optimizer convergence.  Standardized residuals are
        # invariant to this linear rescaling.
        am = arch_model(
            log_returns[col].astype(float) * 100.0,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist=garch_dist,
        )
        res = am.fit(disp="off")

        # v2: check convergence
        if res.convergence_flag != 0:
            print(
                f"  WARNING: GARCH did not converge for {col} "
                f"(flag={res.convergence_flag})"
            )

        # Report GARCH parameters
        omega = res.params.get("omega", float("nan"))
        alpha = res.params.get("alpha[1]", float("nan"))
        beta = res.params.get("beta[1]", float("nan"))
        persistence = alpha + beta

        extra = ""
        if garch_dist in ("t", "skewt"):
            nu_garch = res.params.get("nu", float("nan"))
            extra += f", nu={nu_garch:.2f}"
        if garch_dist == "skewt":
            lam = res.params.get("lambda", float("nan"))
            extra += f", lambda={lam:.4f}"

        print(
            f"  {col:5s}: omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}, "
            f"persist={persistence:.4f}{extra}, "
            f"conv={res.convergence_flag}"
        )

        # standardized residuals = filtered returns
        filtered[col] = res.std_resid

        # one-step-ahead forecasted volatility (convert back from % scale)
        fcast = res.forecast(horizon=1)
        garch_vol_1d[col] = float(np.sqrt(fcast.variance.iloc[-1, 0]) / 100.0)

    print("-" * 65)

    workdir = Path(__file__).resolve().parent / "estimation_results"
    workdir.mkdir(exist_ok=True)

    # Use per-dist suffix so all three can coexist on disk
    suffix = "" if garch_dist == "normal" else f"_{garch_dist}"
    filtered_csv = workdir / f"filtered_returns{suffix}.csv"
    out_json = workdir / f"skew_t_ac_fit{suffix}.json"

    script_path = Path(__file__).parent / "fit_ac_skewt.R"

    filtered.to_csv(filtered_csv, index=False)

    print(f"\nRunning R script: {script_path.name}")
    subprocess.run(
        ["Rscript", str(script_path), str(filtered_csv), str(out_json)],
        check=True,
    )

    with open(out_json, "r") as f:
        fit = json.load(f)

    print(fit.keys())  # e.g. dp, cp, logL
    print(fit["dp"])  # direct parameters

    # Generate LaTeX table
    tex_table = json_to_tex_table(str(out_json), digits=5)
    print(tex_table)

    with open(workdir / f"skew_t_fit_table{suffix}.tex", "w") as f:
        f.write(tex_table)

    # Save GARCH volatilities and config for reference
    garch_meta = {"dist": garch_dist, "volatilities": garch_vol_1d}
    garch_json = workdir / f"garch_vol{suffix}.json"
    with open(garch_json, "w") as f:
        json.dump(garch_meta, f, indent=2)
    print(f"\nGARCH config+vol saved to: {garch_json.name}")
    print(f"GARCH innovation distribution used: '{garch_dist}'")


if __name__ == "__main__":
    args = parse_args()
    main(garch_dist=args.dist)
