"""
Estimate standard (non-skew) multivariate t distribution parameters and
compare with Hu and Kercheval (2010) to validate the estimation pipeline.

The fitting is delegated entirely to R: ``fit_standard_t.R`` uses the
``sn`` package (``mst.mple`` with ``alpha`` fixed to zero), giving a
symmetric multivariate t with no skewness parameter.  The same
GARCH-filtered residuals produced by ``estimate_skew_t.py`` are reused so
that both fits are directly comparable.

Outputs
-------
``experiment_t_vs_skew_t_estimation/estimation_results/t_fit.json``
    Raw fit results from R.
``experiment_t_vs_skew_t_estimation/estimation_results/t_reconciliation.txt``
    Comparison table vs Hu and Kercheval (2010).

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiment_t_vs_skew_t_estimation/estimate_standard_t.py
"""

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "estimation_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Re-use the filtered residuals produced by estimate_skew_t.py so that both
# the skew-t and standard-t fits use identical input data.
FILTERED_CSV = RESULTS_DIR / "filtered_returns.csv"
OUT_JSON = RESULTS_DIR / "t_fit.json"
R_SCRIPT = Path(__file__).parent / "fit_standard_t.R"


def main() -> None:
    """Fit standard t via R and compare with Hu and Kercheval (2010)."""

    print("=" * 60)
    print("Standard multivariate-t parameter estimation (via R)")
    print("=" * 60)

    if not FILTERED_CSV.exists():
        raise FileNotFoundError(
            f"Filtered residuals not found at {FILTERED_CSV}. "
            "Run experiment_t_vs_skew_t_estimation/estimate_skew_t.py first."
        )

    print(f"\nFitting multivariate t using R ({R_SCRIPT.name}) …")
    subprocess.run(
        ["Rscript", str(R_SCRIPT), str(FILTERED_CSV), str(OUT_JSON)],
        check=True,
    )

    with open(OUT_JSON, "r") as f:
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
        # For a GARCH-standardized residual with unit variance,
        # Cov_ii = nu/(nu-2) * Sigma_ii = 1  =>  Sigma_ii = (nu-2)/nu
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

    out_path = RESULTS_DIR / "t_reconciliation.txt"
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
    main()
