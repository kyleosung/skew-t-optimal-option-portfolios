"""
Implement percentile computations.
"""

from scipy.stats import norm


def cornish_fisher_t_value(confidence_level: float, nu: float) -> float:
    """
    Following George and Sivaram [1987, Equation 2.13], return the t-value
    for the Cornish-Fisher expansion given a confidence level.

    Parameters
    ----------
    confidence_level : float
        The desired confidence level (between 0 and 1).
    nu : float
        The degrees of freedom.

    Returns
    -------
    float
        The t-value for the Cornish-Fisher expansion.
    """
    if nu <= 2:
        raise ValueError("Degrees of freedom must be greater than 2.")

    coeff = (nu / (nu - 2)) ** 0.5

    z_alpha = norm.ppf(confidence_level)

    t_alpha = coeff * (
        z_alpha
        + (z_alpha**3 - 3 * z_alpha) / (4 * (nu - 2))
        + (5 * z_alpha**5 - 56 * z_alpha**3 + 75 * z_alpha) / (96 * (nu - 2) ** 2)
        + (3 * z_alpha**7 - 81 * z_alpha**5 + 417 * z_alpha**3 - 315 * z_alpha)
        / (384 * (nu - 2) ** 3)
    )

    return t_alpha
