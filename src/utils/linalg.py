"""
Linear algebra utilities such as ridge regularization and numerical stability
"""

import warnings

import numpy as np

from src.constants.linalg import (
    CONDITION_THRESHOLD,
    ENABLE_TIKHONOV_REGULARIZATION,
    REGULARIZATION_FACTOR_MAX,
    REGULARIZATION_FACTOR_MIN,
)


def apply_tikhonov_regularization(
    matrix: np.ndarray,
    condition_number: float,
    matrix_name: str = "matrix",
) -> np.ndarray:
    """
    Apply Tikhonov (ridge) regularization to an ill-conditioned matrix.
    This adds a small positive value to the diagonal to improve numerical
    stability: M_reg = M + lambda*I, where lambda is adaptively chosen based on the
    condition number and matrix trace.
    The regularization is controlled by the global flag ENABLE_TIKHONOV_REGULARIZATION.
    When disabled, returns the original matrix unchanged.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to regularize
    condition_number : float
        The condition number of the matrix
    matrix_name : str, optional
        Name of the matrix for warning messages, default "matrix"

    Returns
    -------
    np.ndarray
        The regularized matrix (or original if not ill-conditioned or regularization disabled)
    """
    # If regularization is disabled globally, return original matrix
    if not ENABLE_TIKHONOV_REGULARIZATION:
        return matrix

    # Check if matrix is ill-conditioned
    if condition_number > CONDITION_THRESHOLD:
        trace_M = np.trace(matrix)
        M_size = matrix.shape[0]
        regularization_factor = min(
            REGULARIZATION_FACTOR_MIN * np.sqrt(condition_number / CONDITION_THRESHOLD),
            REGULARIZATION_FACTOR_MAX,
        )
        lambda_reg = (abs(trace_M) / M_size) * regularization_factor

        warnings.warn(
            f"{matrix_name} is ill-conditioned (cond={condition_number:.2e}). "
            f"Applying Tikhonov regularization with lambda={lambda_reg:.2e}."
        )

        matrix_reg = matrix + lambda_reg * np.eye(M_size)
        return matrix_reg
    return matrix


def invert_matrix_with_regularization(
    matrix: np.ndarray,
    matrix_name: str = "matrix",
) -> np.ndarray:
    """
    Invert a matrix with automatic regularization and intelligent method selection.
    This function:
    1. Computes the condition number of the input matrix
    2. Applies Tikhonov regularization if the matrix is ill-conditioned
    3. Checks if the regularized matrix is still ill-conditioned
    4. Uses pseudo-inverse for ill-conditioned matrices, regular inverse otherwise

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to invert
    matrix_name : str, optional
        Name of the matrix for warning messages, default "matrix"

    Returns
    -------
    np.ndarray
        The inverted matrix

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> A_inv = invert_matrix_with_regularization(A, "Matrix A")
    """
    cond = np.linalg.cond(matrix)

    if cond > CONDITION_THRESHOLD:
        matrix = apply_tikhonov_regularization(matrix, cond, matrix_name)

    cond_after = np.linalg.cond(matrix)
    if cond_after > CONDITION_THRESHOLD:
        return np.linalg.pinv(matrix)
    else:
        return np.linalg.inv(matrix)
