"""
Helpers for writing the tables of the paper as LaTeX files.
"""

from pathlib import Path


def fmt_num(value: float, digits: int = 3) -> str:
    """
    Format a number with a fixed number of decimal places.

    Parameters
    ----------
    value : float
        Number to format.
    digits : int, optional
        Number of decimal places, by default 3.

    Returns
    -------
    str
        Formatted number, with negative zero printed as zero.
    """
    text = f"{value:.{digits}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{digits}f}"
    return text


def fmt_pct(value: float, digits: int = 1) -> str:
    """
    Format a percentage change with an explicit sign.

    Parameters
    ----------
    value : float
        Percentage to format.
    digits : int, optional
        Number of decimal places, by default 1.

    Returns
    -------
    str
        Formatted percentage, with a ``+`` in front of positive values and no sign
        in front of zero.
    """
    text = fmt_num(value, digits)
    return text if float(text) <= 0.0 else f"+{text}"


def stack_cell(*lines: str) -> str:
    """
    Stack several lines of text in a single table cell, e.g. for long column headers.

    Parameters
    ----------
    *lines : str
        Lines of the cell, from top to bottom.

    Returns
    -------
    str
        Nested LaTeX ``tabular`` that centres the lines within one cell.
    """
    return r"\begin{tabular}[c]{@{}c@{}}" + r" \\ ".join(lines) + r"\end{tabular}"


def write_latex_tabular(
    path: Path,
    col_spec: str,
    headers: list[str],
    rows: list[list[str]],
    group_starts: tuple[int, ...] = (),
    bold_header: bool = True,
    spanner: tuple[int, int, str] | None = None,
) -> None:
    """
    Write a booktabs ``tabular`` environment to a ``.tex`` file.

    The surrounding ``table`` environment, caption, and label are kept in the
    paper so that only the numbers are generated. The header row is set in bold
    unless ``bold_header`` is false.

    Parameters
    ----------
    path : Path
        Output ``.tex`` file.
    col_spec : str
        Column specification, e.g. ``"llrr"``.
    headers : list[str]
        Header cells.
    rows : list[list[str]]
        Body rows, each a list of cells.
    group_starts : tuple[int, ...], optional
        Indices of the body rows that start a new group of rows, which a ``\\midrule``
        separates from the rows before them, by default none.
    bold_header : bool, optional
        Whether to set the header row in bold, by default True. If false, the
        headers are typeset as written, matching the style of the text.
    spanner : tuple[int, int, str], optional
        ``(first, last, text)`` of a header centred over the 1-based columns
        ``first`` to ``last`` in an extra row above the headers, by default none.

    Raises
    ------
    ValueError
        If a row does not have as many cells as there are headers.
    """
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"Row {row!r} does not have {len(headers)} cells")
    prefix = r"\bfseries\boldmath " if bold_header else ""
    lines = [rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule"]
    if spanner is not None:
        first, last, text = spanner
        cells = [""] * (first - 1) + [
            rf"\multicolumn{{{last - first + 1}}}{{c}}{{{prefix}{text}}}"
        ]
        cells += [""] * (len(headers) - last)
        lines += [
            " & ".join(cells) + r" \\",
            rf"\cmidrule(lr){{{first}-{last}}}",
        ]
    lines += [
        " & ".join(f"{prefix}{header}" for header in headers) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i in group_starts and i > 0:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
