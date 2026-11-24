"""
Tables of the paper, written as LaTeX ``tabular`` files.

The captions and the text stay in the paper, which ``\\input`` these files.  Each writer
takes the folder of the files and the ``prefix`` of their names, none by default.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.estimation import Fit
from src.experiments import MODEL_LABELS, empirical_risks
from src.settings import TRADING_DAYS_PER_YEAR
from src.utils.latex import fmt_num, fmt_pct, stack_cell, write_latex_tabular


def write_fit_tables(fit: Fit, folder: Path, prefix: str = "") -> None:
    """
    Tables of the estimation of Section 4: the skew-t fit and the GARCH(1,1) fits.

    The first gives the direct parameters ``(mu^Z, alpha^Z, omega^Z, Sigma^Z)`` of the
    skew-t distribution fitted to the standardized residuals, with
    ``omega^Z = diag(Sigma^Z)^(-1/2) alpha^Z``, the mean ``E[Z_n]`` they imply, the
    degrees of freedom and the log likelihood.  The second gives the GARCH(1,1)
    estimates, the volatility forecast ``sigma_n(t)`` for the day after the sample, the
    annual scale ``sqrt(252) sigma_n(t) sqrt(Sigma^Z_nn)`` and the last price.

    Parameters
    ----------
    fit : Fit
        Fit of the estimation sample.
    folder : Path
        Folder of the files.
    prefix : str, optional
        Prefix of the names of the files, by default none.
    """
    dp, n = fit.skew_t["dp"], len(fit.tickers)

    def num(x: float, digits: int = 4) -> str:
        """Number in math mode."""
        return f"${fmt_num(float(x), digits)}$"

    columns = [
        (r"$\bs\mu^{\vec{Z}}$", dp["mu"]),
        (r"$\bs\alpha^{\vec{Z}}$", dp["alpha"]),
        (r"$\bs\omega^{\vec{Z}}$", dp["omega"]),
        (r"$\E[Z_n]$", fit.skew_t["cp"]["mean"]),
    ]
    lines = [
        rf"\begin{{tabular}}{{l*{{{n + 4}}}{{r}}}}",
        r"\toprule",
        rf"\multicolumn{{5}}{{c}}{{}} & \multicolumn{{{n}}}{{c}}{{$\bs\Sigma^{{\vec{{Z}}}}$}} \\",
        rf"\cmidrule(lr){{6-{n + 5}}}",
        " & ".join(["Asset"] + [label for label, _ in columns] + list(fit.tickers))
        + r" \\",
        r"\midrule",
    ]
    for i, ticker in enumerate(fit.tickers):
        cells = [ticker] + [num(values[i]) for _, values in columns]
        lines.append(" & ".join(cells + [num(x) for x in dp["Sigma"][i]]) + r" \\")
    lines += [
        r"\midrule",
        rf"\multicolumn{{5}}{{c}}{{$\nu^{{\vec{{Z}}}}$}} & \multicolumn{{{n}}}{{c}}{{Log likelihood}} \\",
        rf"\multicolumn{{5}}{{c}}{{{num(dp['nu'])}}} "
        rf"& \multicolumn{{{n}}}{{c}}{{{num(fit.skew_t['logL'])}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path = folder / f"{prefix}skew_t_fit_tabular.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    scale = np.sqrt(np.diag(np.asarray(dp["Sigma"], dtype=float)))
    garch = fit.garch
    write_latex_tabular(
        folder / f"{prefix}garch_tabular.tex",
        "lrrrrrrr",
        [
            "Asset",
            r"$\hat{\iota}_n$ (\%)",
            r"$\hat{\garchmean}_n^{[0]} \times 10^{6}$",
            r"$\hat{\garchmean}_n^{[1]}$",
            r"$\hat{\garchvar}_n^{[1]}$",
            r"$\varsigma_n(t)$ (\%)",
            r"$\sigma_n$ (\%)",
            r"$S_n(t)$",
        ],
        [
            [
                ticker,
                num(100 * garch.loc[ticker, "iota"], 3),
                num(1e6 * garch.loc[ticker, "alpha0"], 3),
                num(garch.loc[ticker, "alpha1"], 3),
                num(garch.loc[ticker, "beta1"], 3),
                num(100 * garch.loc[ticker, "sigma_forecast"], 3),
                num(
                    100
                    * np.sqrt(TRADING_DAYS_PER_YEAR)
                    * garch.loc[ticker, "sigma_forecast"]
                    * scale[i],
                    3,
                ),
                num(fit.spots[i], 3),
            ]
            for i, ticker in enumerate(fit.tickers)
        ],
        bold_header=False,
    )


def write_risk_tables(
    risks: dict[str, dict[str, float]], folder: Path, prefix: str = ""
) -> None:
    """
    Tables of Section 5.2: CFVaR3 of the optimal portfolios, and of neglecting skewness.

    Parameters
    ----------
    risks : dict[str, dict[str, float]]
        For each option type (in the order of the rows), the CFVaR3 of the optimal
        portfolios of :func:`~src.experiments.risk_comparison`.
    folder : Path
        Folder of the files.
    prefix : str, optional
        Prefix of the names of the files, by default none.
    """
    write_latex_tabular(
        folder / f"{prefix}risk_comparison_tabular.tex",
        "lrrrrr",
        [
            "Type",
            stack_cell("Variance", "Optimal Portfolio"),
            stack_cell(r"$\CFVaR_2$", "Optimal Portfolio"),
            stack_cell(r"$\CFVaR_3$", "Optimal Portfolio"),
            stack_cell("Variance", r"\% Incr."),
            stack_cell(r"$\CFVaR_2$", r"\% Incr."),
        ],
        [
            [
                kind.capitalize(),
                *(fmt_num(r[k], 6) for k in ("var", "cf2", "cf3")),
                *(fmt_pct(100 * (r[k] / r["cf3"] - 1)) for k in ("var", "cf2")),
            ]
            for kind, r in risks.items()
        ],
        spanner=(2, 4, r"$\CFVaR_3$ of"),
    )
    write_latex_tabular(
        folder / f"{prefix}neglecting_skewness_tabular.tex",
        "lrrrrr",
        [
            "Type",
            stack_cell(r"Skew-$t$", "Optimal"),
            stack_cell("Student's $t$", "Optimal"),
            stack_cell("Student's $t$ Optimal", r"(Skew-$t$ $\CFVaR_3$)"),
            stack_cell("Normal Optimal", r"(Skew-$t$ $\CFVaR_3$)"),
            stack_cell(r"\% Incr.", "(Skewness)"),
        ],
        [
            [
                kind.capitalize(),
                *(fmt_num(r[k], 6) for k in ("cf3", "t_own", "t", "normal")),
                fmt_pct(100 * (r["t"] / r["cf3"] - 1)),
            ]
            for kind, r in risks.items()
        ],
    )


def _oos_rows(pnl: pd.DataFrame, tail_risk: float) -> list[list[str]]:
    """
    Rows of the out-of-sample table of the models of a sample of profits and losses.

    Parameters
    ----------
    pnl : pd.DataFrame
        Profit and loss of each model (the columns).
    tail_risk : float
        Tail risk ``alpha``.

    Returns
    -------
    list[list[str]]
        For each model, its label, the empirical variance, eCFVaR2 and eCFVaR3, and the
        percentage increase in eCFVaR3 over the first model.
    """
    risks = {label: empirical_risks(pnl[label].to_numpy(), tail_risk) for label in pnl}
    base = risks[MODEL_LABELS["skew_t"]][2]
    return [
        [
            label,
            *(fmt_num(x, 6) for x in r),
            (
                "--"
                if label == MODEL_LABELS["skew_t"]
                else fmt_pct(100 * (r[2] / base - 1))
            ),
        ]
        for label, r in risks.items()
    ]


def write_out_of_sample_table(
    pnl: pd.DataFrame, tail_risk: float, folder: Path, prefix: str = ""
) -> None:
    """
    Table of the out-of-sample test of Section 5.3.

    Parameters
    ----------
    pnl : pd.DataFrame
        Profit and loss of each model (the columns) on each day, from
        :func:`~src.experiments.out_of_sample`.
    tail_risk : float
        Tail risk ``alpha``.
    folder : Path
        Folder of the file.
    prefix : str, optional
        Prefix of the name of the file, by default none.
    """
    write_latex_tabular(
        folder / f"{prefix}out_of_sample_tabular.tex",
        "lrrrr",
        [
            "Model",
            "Emp. Variance",
            rf"$\eCFVaR_2$ ($\alpha = {tail_risk:g}$)",
            rf"$\eCFVaR_3$ ($\alpha = {tail_risk:g}$)",
            stack_cell(r"$\eCFVaR_3$", r"\% Incr."),
        ],
        _oos_rows(pnl, tail_risk),
    )


def write_monte_carlo_table(
    monte_carlo: pd.DataFrame, folder: Path, prefix: str = ""
) -> None:
    """
    Table of the Monte Carlo evaluation of Section 5.4.

    Parameters
    ----------
    monte_carlo : pd.DataFrame
        Result of :func:`~src.experiments.monte_carlo`.
    folder : Path
        Folder of the file.
    prefix : str, optional
        Prefix of the name of the file, by default none.
    """
    write_latex_tabular(
        folder / f"{prefix}monte_carlo_tabular.tex",
        "llrrr",
        ["Type", "Maturity", "MC delta-gamma", "MC full", r"\% Change"],
        [
            [
                kind.capitalize(),
                f"{round(12 * maturity)} mo." if maturity < 1 else f"{maturity:g} y.",
                fmt_num(row["delta_gamma"], 6),
                fmt_num(row["full"], 6),
                fmt_pct(row["pct_change"], 2),
            ]
            for (kind, maturity), row in monte_carlo.iterrows()
        ],
        group_starts=(len(monte_carlo) // 2,),
    )


def write_pricing_table(prices: pd.DataFrame, folder: Path, prefix: str = "") -> None:
    """
    Table of the Monte Carlo check of the option prices of Appendix B.

    Parameters
    ----------
    prices : pd.DataFrame
        Result of :func:`~src.experiments.pricing_check`.
    folder : Path
        Folder of the file.
    prefix : str, optional
        Prefix of the name of the file, by default none.
    """
    write_latex_tabular(
        folder / f"{prefix}monte_carlo_pricing_tabular.tex",
        "llrrrr",
        ["Type", "Ticker", "FFT Price", "MC Price", "MC Std. Error", r"\% Change"],
        [
            [
                kind.capitalize(),
                ticker,
                fmt_num(row["fft"], 6),
                fmt_num(row["monte_carlo"], 6),
                fmt_num(row["std_error"], 6),
                fmt_pct(100 * (row["monte_carlo"] / row["fft"] - 1), 2),
            ]
            for (kind, ticker), row in prices.iterrows()
        ],
        group_starts=(len(prices) // 2,),
    )
