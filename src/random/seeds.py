"""
Utils with random seeds.
"""


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
