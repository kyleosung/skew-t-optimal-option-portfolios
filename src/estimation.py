"""
Estimation of Section 4: a GARCH(1,1) filter for each asset, and the skew-t, Student's t
and normal fits of the standardized residuals.

The GARCH(1,1) models are fitted by Gaussian quasi-maximum likelihood with the ``arch``
package, and the skew-t and Student's t distributions to the standardized residuals
with ``mst.mple`` of the R package ``sn`` (``estimation/*.R``; ``Rscript`` must be on
the PATH).
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

R_DIR = Path(__file__).resolve().parent.parent / "estimation"

# Log returns are multiplied by this factor before the GARCH fit, for the conditioning
# of the optimizer; the standardized residuals do not depend on it
PERCENT_SCALE = 100.0


@dataclass(frozen=True)
class Fit:
    """
    Fitted models of the standardized residuals ``Z`` of a set of assets.

    Parameters
    ----------
    tickers : tuple[str, ...]
        Assets.
    last_day : pd.Timestamp
        Last day of the estimation sample.
    residuals : pd.DataFrame
        Standardized residuals ``Z``.
    garch : pd.DataFrame
        GARCH(1,1) estimates on the original return scale, indexed by the assets:
        ``iota`` (conditional mean), ``alpha0``, ``alpha1``, ``beta1`` (the paper's
        ``a^[0]``, ``a^[1]`` and ``b^[1]``) and ``sigma_forecast`` (volatility forecast
        for the day after the sample).
    spots : np.ndarray
        Prices on the last day of the sample.
    skew_t : dict
        Skew-t fit as written by ``estimation/fit_ac_skewt.R``.
    student_t : dict
        Student's t fit as written by ``estimation/fit_standard_t.R``.
    normal_mean : np.ndarray
        Sample mean of ``Z``.
    normal_cov : np.ndarray
        Sample covariance matrix of ``Z`` (maximum likelihood).
    """

    tickers: tuple[str, ...]
    last_day: pd.Timestamp
    residuals: pd.DataFrame
    garch: pd.DataFrame
    spots: np.ndarray
    skew_t: dict
    student_t: dict
    normal_mean: np.ndarray
    normal_cov: np.ndarray


def load_prices(path: Path, tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Closing prices of some assets.

    Parameters
    ----------
    path : Path
        CSV file of the prices, indexed by date.
    tickers : tuple[str, ...]
        Columns to read.

    Returns
    -------
    pd.DataFrame
        Prices of ``tickers``, indexed by date.
    """
    return pd.read_csv(path, index_col=0, parse_dates=True)[list(tickers)]


def estimation_returns(
    prices: pd.DataFrame, start: str, end: str, burn_in: int = 0
) -> pd.DataFrame:
    """
    Daily log returns of the estimation sample.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices indexed by date.
    start : str
        First price used.
    end : str
        Last price used.
    burn_in : int, optional
        Number of the first returns that are discarded before any model is fitted: they
        enter neither the GARCH fit nor the initialization of its recursion, by default 0.

    Returns
    -------
    pd.DataFrame
        Log returns of the prices from ``start`` to ``end``, without the first
        ``burn_in``.
    """
    window = prices.loc[start:end]
    return np.log(window / window.shift(1)).dropna(how="any").iloc[burn_in:]


def garch_filter(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a GARCH(1,1) model with Gaussian innovations to each asset.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns of each asset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Standardized residuals, and the GARCH estimates as in :class:`Fit`.
    """
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    rows = {}
    for ticker in returns.columns:
        result = arch_model(
            returns[ticker].astype(float) * PERCENT_SCALE,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
        ).fit(disp="off")
        forecast = result.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
        residuals[ticker] = result.std_resid
        rows[ticker] = {
            "iota": result.params["mu"] / PERCENT_SCALE,
            "alpha0": result.params["omega"] / PERCENT_SCALE**2,
            "alpha1": result.params["alpha[1]"],
            "beta1": result.params["beta[1]"],
            "sigma_forecast": np.sqrt(forecast) / PERCENT_SCALE,
        }
    return residuals, pd.DataFrame(rows).T


def fit_with_r(script: str, residuals: pd.DataFrame) -> dict:
    """
    Fit a distribution to the standardized residuals with an R script of ``estimation/``.

    Parameters
    ----------
    script : str
        ``"fit_ac_skewt.R"`` or ``"fit_standard_t.R"``.
    residuals : pd.DataFrame
        Standardized residuals.

    Returns
    -------
    dict
        The JSON written by the script.

    Raises
    ------
    RuntimeError
        If ``Rscript`` is not on the PATH.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript (R with the packages sn and jsonlite) is required.")
    with tempfile.TemporaryDirectory() as folder:
        data, output = Path(folder) / "z.csv", Path(folder) / "fit.json"
        residuals.to_csv(data, index=False)
        subprocess.run(
            [rscript, str(R_DIR / script), str(data), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(output.read_text())


def fit_assets(prices: pd.DataFrame, start: str, end: str, burn_in: int = 0) -> Fit:
    """
    Fit the models of Section 4 to some assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices of the assets, indexed by date.
    start : str
        First price used.
    end : str
        Last price used.
    burn_in : int, optional
        Number of discarded first returns, by default 0.

    Returns
    -------
    Fit
        The GARCH, skew-t, Student's t and normal fits.
    """
    returns = estimation_returns(prices, start, end, burn_in)
    residuals, garch = garch_filter(returns)
    return Fit(
        tickers=tuple(prices.columns),
        last_day=returns.index[-1],
        residuals=residuals,
        garch=garch,
        spots=prices.loc[returns.index[-1]].to_numpy(dtype=float),
        skew_t=fit_with_r("fit_ac_skewt.R", residuals),
        student_t=fit_with_r("fit_standard_t.R", residuals),
        normal_mean=residuals.mean().to_numpy(),
        normal_cov=np.cov(residuals.to_numpy(), rowvar=False, ddof=0),
    )
