"""
Utilities for converting model fit results to LaTeX tables.
"""

import json

import numpy as np


def fmt(x: object, digits: int = 4) -> str:
    """
    Format a value for LaTeX table output. Handles numbers, None, and other types.
    """
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.{digits}g}"
    return str(x)


def tex_matrix(name: str, mat: object, digits: int = 4) -> list[str]:
    """
    Convert a matrix to LaTeX table rows.
    """
    arr = np.asarray(mat)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    rows = []
    for i in range(arr.shape[0]):
        vals = " & ".join(fmt(v, digits) for v in arr[i])
        rows.append(f"{name}_{i+1} & {vals} \\\\")
    return rows


def json_to_tex_table(json_path: str | object, digits: int = 4) -> str:
    """
    Convert a JSON file containing model fit results to a LaTeX table.
    """
    with open(json_path, "r") as f:  # type: ignore
        obj = json.load(f)

    dp = obj.get("dp", {})
    cp = obj.get("cp", {})
    logL = obj.get("logL", None)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"Quantity & Value \\")
    lines.append(r"\midrule")

    # Direct parameters
    for key in ["mu", "omega", "nu"]:
        if key in dp:
            val = np.asarray(dp[key]).flatten()
            val_tex = ", ".join(fmt(x, digits) for x in val)
            lines.append(rf"$\mathrm{{dp.{key}}}$ & ${val_tex}$ \\")

    if "Sigma" in dp:
        sigma = np.asarray(dp["Sigma"])
        sigma_tex = (
            r"\begin{pmatrix}"
            + r"\\".join(" & ".join(fmt(x, digits) for x in row) for row in sigma)
            + r"\end{pmatrix}"
        )
        lines.append(rf"$\mathrm{{dp.\Sigma}}$ & ${sigma_tex}$ \\")

    # Centred parameters
    for key, val in cp.items():
        arr = np.asarray(val)
        if arr.ndim <= 1:
            val_tex = ", ".join(fmt(x, digits) for x in arr.flatten())
        else:
            val_tex = (
                r"\begin{pmatrix}"
                + r"\\".join(" & ".join(fmt(x, digits) for x in row) for row in arr)
                + r"\end{pmatrix}"
            )
        lines.append(rf"$\mathrm{{cp.{key}}}$ & ${val_tex}$ \\")

    if logL is not None:
        lines.append(rf"$\log L$ & ${fmt(logL, digits)}$ \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Skew-$t$ model fit.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    tex = json_to_tex_table("output.json", digits=5)
    print(tex)
