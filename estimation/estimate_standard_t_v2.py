"""
Estimate standard (non-skew) multivariate t distribution parameters (v2).

Uses the GARCH filtered residuals from ``estimate_skew_t_v2.py``
(configurable innovations: normal, t, or skewt).  The R fitting step is
the same as the original (``fit_standard_t.R`` with ``alpha`` fixed to zero).

Outputs are saved with a per-dist suffix so that all parameter sets
coexist on disk:
    - ``t_fit.json``       (normal GARCH)
    - ``t_fit_t.json``     (t-GARCH)
    - ``t_fit_skewt.json`` (skewt-GARCH)

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiment_t_vs_skew_t_estimation/estimate_standard_t_v2.py
    python experiment_t_vs_skew_t_estimation/estimate_standard_t_v2.py --dist skewt
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from src.constants.five_option_dataset import (
    ASSET_TICKERS,
    t_CORR,
    t_MU_HAT,
    t_NU,
)

VALID_GARCH_DISTS = ("normal", "t", "skewt")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "estimation_results"
RESULTS_DIR.mkdir(exist_ok=True)
R_SCRIPT = Path(__file__).parent / "fit_standard_t.R"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Standard multivariate-t estimation (v2) with configurable GARCH."
    )
    parser.add_argument(
        "--dist",
        choices=VALID_GARCH_DISTS,
        default="t",
        help=(
            "GARCH(1,1) innovation distribution used for filtering. "
            "Must match what was used in estimate_skew_t_v2.py. "
            "Options: 'normal', 't' (default), 'skewt'."
        ),
    )
    return parser.parse_args()


def main(garch_dist: str = "t") -> None:
    """
    Fit standard t via R and compare with Hu and Kercheval (2010).

    Parameters
    ----------
    garch_dist : str
        Which GARCH innovation distribution was used to filter the residuals.
    """
    suffix = "" if garch_dist == "normal" else f"_{garch_dist}"
    filtered_csv = RESULTS_DIR / f"filtered_returns{suffix}.csv"
    out_json = RESULTS_DIR / f"t_fit{suffix}.json"

    print("=" * 60)
    print(f"Standard multivariate-t parameter estimation (GARCH dist='{garch_dist}')")
    print("=" * 60)

    if not filtered_csv.exists():
        raise FileNotFoundError(
            f"Filtered residuals not found at {filtered_csv}. "
            f"Run estimate_skew_t_v2.py --dist {garch_dist} first."
        )

    print(f"\nFitting multivariate t using R ({R_SCRIPT.name}) ...")
    subprocess.run(
        ["Rscript", str(R_SCRIPT), str(filtered_csv), str(out_json)],
        check=True,
    )

    with open(out_json, "r") as f:
        fit = json.load(f)

    mu_r = np.array(fit["mu"])
    Sigma_r = np.array(fit["Sigma"])
    Corr_r = np.array(fit["Corr"])
    nu_r = float(fit["nu"])
    loglik_r = float(fit["logL"])

    # ----------- Comparison table -----------
    header = f"{'Quantity':<28}  {'Estimated (R)':<22}  {'HK (Table 4)'}"
    sep = "-" * len(header)

    lines = [header, sep]

    lines.append(f"{'Degrees of freedom nu':<28}  {nu_r:<22.4f}  {t_NU:.4f}")
    lines.append(sep)

    for i, ticker in enumerate(ASSET_TICKERS):
        lines.append(f"{'mu_hat ' + ticker:<28}  {mu_r[i]:<22.4f}  {t_MU_HAT[i]:.4f}")
    lines.append(sep)

    for i, ticker in enumerate(ASSET_TICKERS):
        lines.append(
            f"{'Sigma_diag ' + ticker:<28}  {Sigma_r[i, i]:<22.5f}"
            f"  {(nu_r - 2) / nu_r:.5f}  (expected (nu_r-2)/nu_r)"
        )
    lines.append(sep)

    asset_pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    for i, j in asset_pairs:
        pair_label = f"corr({ASSET_TICKERS[i]},{ASSET_TICKERS[j]})"
        lines.append(f"{pair_label:<28}  {Corr_r[i, j]:<22.4f}  {t_CORR[i, j]:.4f}")
    lines.append(sep)

    lines.append(f"{'Log-likelihood':<28}  {loglik_r:<22.2f}  (HK not reported)")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    out_path = RESULTS_DIR / f"t_reconciliation{suffix}.txt"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nTable saved to: {out_path}")

    nu_diff = abs(nu_r - t_NU)
    mu_diff = float(np.max(np.abs(mu_r - t_MU_HAT)))
    corr_diff = float(
        np.max(
            np.abs(Corr_r[np.triu_indices(5, k=1)] - t_CORR[np.triu_indices(5, k=1)])
        )
    )
    print(
        f"\nMax absolute differences vs HK:"
        f"  nu: {nu_diff:.4f}   mu_hat: {mu_diff:.4f}   corr: {corr_diff:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(garch_dist=args.dist)
