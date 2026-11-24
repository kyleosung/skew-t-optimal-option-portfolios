"""
Settings of the experiments of the paper.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

TRADING_DAYS_PER_YEAR = 252

# Estimation sample of the five stocks (Hu and Kercheval, 2010, Section 4): the prices
# from SAMPLE_START to SAMPLE_END inclusive are converted to daily log returns and the
# first BURN_IN returns are discarded before any model is fitted, leaving 750 returns
FIVE_STOCKS = ("DIS", "XOM", "PFE", "MO", "INTC")
SAMPLE_START = "2002-07-01"
SAMPLE_END = "2005-08-04"
BURN_IN = 30

# Continuously compounded risk-free rate for two-month options on 4 August 2005: the
# investment-basis constant-maturity Treasury yields were 3.34% (1 month, DGS1MO) and
# 3.47% (3 months), so linear interpolation gives y = 3.405% for two months, and with
# T = 42/252 = 1/6, r_f = ln(1 + y T) / T = 0.0340 (Federal Reserve H.15 release of
# 8 August 2005)
RISK_FREE_RATE = 0.034

TAIL_RISK = 0.01
TIME_PERIOD = 1 / TRADING_DAYS_PER_YEAR

# Options mature after two months, so that the 40-day out-of-sample test fits in their
# life; the Monte Carlo check of Section 5.4 also uses options of one year
TIME_TO_MATURITY = 2 / 12
LONG_TIME_TO_MATURITY = 1.0

# The Student's t fit is used for the strategies of the out-of-sample test only if
# nu >= 6, since CFVaR3 needs the third moment; otherwise it takes the skew-t nu
MIN_T_NU = 6.0

# The normal model is the limit nu -> infinity of the Student's t model
NORMAL_NU = 1000.0

# Out-of-sample test of Section 5.3: trading days after SAMPLE_END
OOS_NUM_DAYS = 40

# Monte Carlo check of the delta-gamma error with these degrees of freedom instead of
# the fitted ones, which are close to 6, where the third moment of the gain diverges
NU_SENSITIVITY = 7.0

# Monte Carlo checks: 20 batches of 100,000 draws, repeated with independent sets of
# draws, whose results are averaged
N_BATCHES = 20
BATCH_SIZE = 100_000
N_SEEDS = 3
