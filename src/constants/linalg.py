"""
Linear algebra constants.
"""

CONDITION_THRESHOLD = 1e10

# Threshold for treating near-zero negative discriminants as zero
# Used in CFVaR2 epsilon calculation to handle numerical precision errors
DISCRIMINANT_NEGATIVE_THRESHOLD = 1e-15

# Tolerance for comparing epsilon_star solutions
# Relative tolerance for checking if numerical and analytic solutions agree
# Note: This is intentionally loose (10%) because numerical optimization can be
# unstable for ill-conditioned problems, and we want to fall back to the more
# stable analytic solution in such cases
EPSILON_STAR_RTOL = 1e-1
# Absolute tolerance for checking if numerical and analytic solutions agree
EPSILON_STAR_ATOL = 1e-3

# Tikhonov regularization parameters
# Global flag to enable/disable Tikhonov regularization for numerical stability
# When enabled, adds ridge regularization (A + lambda*I) to ill-conditioned matrices
# Default: True (enabled) to prevent extreme weight instability
ENABLE_TIKHONOV_REGULARIZATION = True

# Minimum regularization factor (scales with sqrt of condition number / threshold)
REGULARIZATION_FACTOR_MIN = 1e-6
# Maximum regularization factor (cap to prevent over-regularization)
REGULARIZATION_FACTOR_MAX = 1e-3

# Portfolio weight stability testing thresholds
# Default degrees of freedom for t-distribution in tests
DEFAULT_DEGREES_OF_FREEDOM = 5.87
# Threshold for detecting extreme weights (numerical instability indicator)
EXTREME_WEIGHT_THRESHOLD = 1e6
# Threshold for reasonable portfolio weights in typical optimization
REASONABLE_WEIGHT_THRESHOLD = 10
