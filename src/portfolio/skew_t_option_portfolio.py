"""
Backward-compatibility shim.

``SkewTOptionPortfolio`` has been merged into ``OptionPortfolio``.  The
``omega`` parameter (defaulting to the zero vector) now controls skewness;
passing ``omega=None`` or ``omega=0`` recovers the standard Student-t
behaviour.
"""

from src.portfolio.option_portfolio import OptionPortfolio

SkewTOptionPortfolio = OptionPortfolio

__all__ = ["SkewTOptionPortfolio"]
