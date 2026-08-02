r"""
Compare GARCH innovation distributions for multivariate skew-t MLE quality.

Tests three GARCH(1,1) filtering variants:
    1. ``dist="normal"`` -- Gaussian innovations (original Hu-Kercheval approach)
    2. ``dist="t"``      -- Student-t innovations (absorbs univariate fat tails)
    3. ``dist="skewt"``  -- Hansen's skew-t innovations (absorbs univariate
                           fat tails AND asymmetry)

For each variant, the script:
    - Filters returns via GARCH(1,1) with the chosen innovation distribution
    - Passes the standardized residuals to the *same* R routine
      (``fit_ac_skewt.R`` and ``fit_standard_t.R``)
    - Reports the multivariate log-likelihood from each fit
    - Saves the fitted JSON with a per-dist suffix so that ALL results
      coexist on disk simultaneously in ``comparison_results/``
    - Generates a LaTeX table for each distribution

The variant with the highest multivariate skew-t log-likelihood is the best
pre-filter for our downstream experiments.

Usage
-----
    export PYTHONPATH=$(pwd)
    python experiment_t_vs_skew_t_estimation/compare_garch_innovations.py

Requires R with the ``sn`` and ``jsonlite`` packages installed.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from experiment_t_vs_skew_t_estimation.tex_utils import json_to_tex_table

from src.constants.five_option_dataset import ASSET_TICKERS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GARCH_DISTRIBUTIONS = ["normal", "t", "skewt"]

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "comparison_results"
R_SCRIPT_SKEWT = Path(__file__).parent / "fit_ac_skewt.R"
R_SCRIPT_T = Path(__file__).parent / "fit_standard_t.R"


def _suffix_for_dist(dist: str) -> str:
    """Return the file suffix for a given GARCH innovation distribution."""
    return f"_{dist}"


def filter_returns(log_returns: pd.DataFrame, dist: str) -> pd.DataFrame:
    """
    Apply GARCH(1,1) filtering with the specified innovation distribution.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log-return series for each asset (columns).
    dist : str
        Innovation distribution: "normal", "t", or "skewt".

    Returns
    -------
    pd.DataFrame
        Standardized residuals (filtered returns).
    """
    filtered = pd.DataFrame(
        index=log_returns.index, columns=log_returns.columns, dtype=float
    )

    print(f"\n  GARCH(1,1) with dist='{dist}':")
    header = f"  {'Asset':<6} {'omega':>10} {'alpha':>8} {'beta':>8} {'persist':>8}"
    if dist in ("t", "skewt"):
        header += f" {'nu':>8}"
    if dist == "skewt":
        header += f" {'lambda':>8}"
    header += f" {'conv':>5}"
    print(header)
    print("  " + "-" * 70)

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
            dist=dist,  # type: ignore
        )
        res = am.fit(disp="off")

        omega = res.params.get("omega", float("nan"))
        alpha = res.params.get("alpha[1]", float("nan"))
        beta = res.params.get("beta[1]", float("nan"))
        persistence = alpha + beta

        line = (
            f"  {col:<6} {omega:>10.6f} {alpha:>8.4f}"
            f" {beta:>8.4f} {persistence:>8.4f}"
        )
        if dist in ("t", "skewt"):
            nu_g = res.params.get("nu", float("nan"))
            line += f" {nu_g:>8.2f}"
        if dist == "skewt":
            lam = res.params.get("lambda", float("nan"))
            line += f" {lam:>8.4f}"
        line += f" {res.convergence_flag:>5d}"
        print(line)

        if res.convergence_flag != 0:
            print(f"    *** WARNING: did not converge for {col} ***")

        filtered[col] = res.std_resid

    return filtered


def fit_multivariate(
    filtered: pd.DataFrame, dist: str
) -> tuple[dict | None, dict | None]:
    """
    Fit multivariate skew-t and symmetric t via R on the filtered residuals.

    Saves results with per-dist suffixes so all coexist on disk.

    Parameters
    ----------
    filtered : pd.DataFrame
        Standardized GARCH residuals.
    dist : str
        The GARCH innovation distribution label (for file naming).

    Returns
    -------
    tuple of (skew_t_fit, t_fit)
        Parsed JSON dictionaries, or None if fitting failed.
    """
    suffix = _suffix_for_dist(dist)
    RESULTS_DIR.mkdir(exist_ok=True)

    filtered_csv = RESULTS_DIR / f"filtered_returns{suffix}.csv"
    skewt_json = RESULTS_DIR / f"skew_t_ac_fit{suffix}.json"
    t_json = RESULTS_DIR / f"t_fit{suffix}.json"

    filtered.to_csv(filtered_csv, index=False)

    # Fit AC skew-t
    skewt_fit = None
    try:
        subprocess.run(
            ["Rscript", str(R_SCRIPT_SKEWT), str(filtered_csv), str(skewt_json)],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(skewt_json, "r") as f:
            skewt_fit = json.load(f)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    WARNING: AC skew-t fit failed for dist='{dist}': {e}")

    # Fit symmetric t
    t_fit = None
    try:
        subprocess.run(
            ["Rscript", str(R_SCRIPT_T), str(filtered_csv), str(t_json)],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(t_json, "r") as f:
            t_fit = json.load(f)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    WARNING: symmetric t fit failed for dist='{dist}': {e}")

    # Generate LaTeX tables for each fitted model
    if skewt_fit:
        tex = json_to_tex_table(str(skewt_json), digits=5)
        tex_path = RESULTS_DIR / f"skew_t_fit_table{suffix}.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"    LaTeX table saved: {tex_path.name}")

    return skewt_fit, t_fit


def main():
    """
    Compare GARCH innovation distributions by multivariate MLE quality.

    For each of normal, t, and skewt GARCH innovations, fits the
    multivariate AC skew-t and symmetric t models on the standardized
    residuals and reports the log-likelihoods. The best innovation
    distribution is the one producing the highest multivariate
    skew-t log-likelihood.

    All fitted JSON files are saved with per-dist suffixes so that the
    constants module can load all three simultaneously.
    """
    print("=" * 70)
    print("GARCH Innovation Distribution Comparison")
    print("=" * 70)

    # Load data
    csv_path = DATA_DIR / "five_stock_prices.csv"
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices = prices[ASSET_TICKERS]

    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")  # type: ignore
    log_returns = log_returns.loc["2002-07-01":"2005-08-05"].tail(750)

    print(
        f"\nData: {log_returns.shape[0]} observations, "
        f"{log_returns.index.min().date()} to {log_returns.index.max().date()}"
    )

    # Run all variants
    results = {}
    for dist in GARCH_DISTRIBUTIONS:
        print(f"\n{'=' * 70}")
        print(f"Testing GARCH dist = '{dist}'")
        print("=" * 70)

        filtered = filter_returns(log_returns, dist)
        skewt_fit, t_fit = fit_multivariate(filtered, dist)

        results[dist] = {
            "skewt_logL": float(skewt_fit["logL"]) if skewt_fit else None,
            "skewt_nu": float(skewt_fit["dp"]["nu"]) if skewt_fit else None,
            "t_logL": float(t_fit["logL"]) if t_fit else None,
            "t_nu": float(t_fit["nu"]) if t_fit else None,
        }

        if skewt_fit:
            omega = np.array(skewt_fit["dp"]["omega"])
            results[dist]["skewt_omega_norm"] = float(np.linalg.norm(omega))
            print(
                f"\n  AC skew-t logL = {results[dist]['skewt_logL']:.4f}, "
                f"nu = {results[dist]['skewt_nu']:.4f}, "
                f"||omega|| = {results[dist]['skewt_omega_norm']:.4f}"
            )
        if t_fit:
            print(
                f"  Symmetric t logL = {results[dist]['t_logL']:.4f}, "
                f"nu = {results[dist]['t_nu']:.4f}"
            )

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY: Multivariate Log-Likelihood Comparison")
    print("=" * 70)
    print(
        f"\n{'GARCH dist':<12} {'Skew-t logL':>14} {'Skew-t nu':>10} "
        f"{'||omega||':>10} {'Symm-t logL':>14} {'Symm-t nu':>10}"
    )
    print("-" * 72)

    best_dist = None
    best_logL = -np.inf

    for dist in GARCH_DISTRIBUTIONS:
        r = results[dist]
        skewt_str = (
            f"{r['skewt_logL']:.4f}" if r["skewt_logL"] is not None else "FAILED"
        )
        nu_str = f"{r['skewt_nu']:.4f}" if r["skewt_nu"] is not None else "N/A"
        omega_str = (
            f"{r.get('skewt_omega_norm', 0):.4f}"
            if r["skewt_logL"] is not None
            else "N/A"
        )
        t_str = f"{r['t_logL']:.4f}" if r["t_logL"] is not None else "FAILED"
        t_nu_str = f"{r['t_nu']:.4f}" if r["t_nu"] is not None else "N/A"

        if r["skewt_logL"] is not None and r["skewt_logL"] > best_logL:
            best_logL = r["skewt_logL"]
            best_dist = dist

        print(
            f"{dist:<12} {skewt_str:>14} {nu_str:>10} "
            f"{omega_str:>10} {t_str:>14} {t_nu_str:>10}"
        )

    print("-" * 72)
    if best_dist:
        print(
            f"\n*** BEST GARCH innovation for multivariate skew-t MLE: "
            f"dist='{best_dist}' (logL = {best_logL:.4f}) ***"
        )

    # Save summary to JSON
    summary_path = RESULTS_DIR / "garch_innovation_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"results": results, "best_dist": best_dist, "best_logL": best_logL},
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_path}")

    # Provide guidance
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(
        f"\nAll three parameter sets are now saved on disk.\n"
        f"Import them in downstream code via:\n\n"
        f"    from src.constants.five_option_dataset_estimated_v2 import (\n"
        f"        AC_SKEW_T_PARAMS,\n"
        f"        EST_T_PARAMS,\n"
        f"    )\n\n"
        f"    params = AC_SKEW_T_PARAMS['{best_dist}']  # best by MLE\n"
        f"    nu = params['nu']\n"
        f"    corr = params['corr']\n"
    )


if __name__ == "__main__":
    main()
