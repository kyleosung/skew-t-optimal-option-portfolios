"""
Constants for numerical optimization.
"""

# Number of random initial points to use in multi-start optimization
# to increase chances of finding the global minimum.
# The value 10 is a reasonable trade-off between computational cost and solution quality.
NUM_RANDOM_STARTS = 20

# Tolerance for function value convergence in SLSQP optimization.
# A smaller value provides more precise results but may require more iterations.
OPTIMIZATION_FTOL = 1e-12

# Maximum number of iterations for SLSQP optimization.
# Should be large enough to allow convergence for complex optimization problems.
OPTIMIZATION_MAXITER = 1000

# Absolute threshold for checking if a value is close to zero.
# Used to avoid division by zero or skip invalid initial guesses.
OPTIMIZATION_ABS_THRESHOLD = 1e-12
