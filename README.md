# Optimal Option Portfolios for Skew-Elliptical t Returns

This repository contains the code for reproducing the results presented in the paper **Optimal Option Portfolios for Skew-Elliptical t Returns**.

## Overview

This paper explores optimal option portfolio allocation when the underlying asset returns follow a skew-t distribution. 
Portfolio risk is assessed using three criteria: variance, mean-volatility (CFVaR2), and mean-volatility-skewness (CFVaR3) risk measures. 
The framework accommodates both heavy tails and asymmetry in financial markets. 
We derive closed-form expressions for the mean, variance, and third central moment of the portfolio gain, together with explicit optimal portfolios under the variance and mean-volatility measures; the mean-volatility-skewness problem is solved numerically. 
The effect of skewness on the optimal allocations is examined by comparing them with those obtained under Student's t-distributed returns. 
The methodology is illustrated with European call and put options.

**Authors:** Kyle Sung and Traian A. Pirvu  
**Institution:** Department of Mathematics and Statistics, McMaster University, Hamilton, ON, Canada  
**Correspondence To:** Kyle Sung, sungk5@mcmaster.ca  
**Paper Link:** https://arxiv.org/abs/2601.07991  

**Keywords:** Options, Portfolio Optimization, Skew-t Returns

## Citation

If you find these results useful in your research, please consider citing our work.

### BibTeX
```bibtex
@misc{sung2026optimaloptionportfoliosskewelliptical,
      title={Optimal Option Portfolios for Skew-Elliptical t Returns}, 
      author={Kyle Sung and Traian A. Pirvu},
      year={2026},
      eprint={2601.07991},
      archivePrefix={arXiv},
      primaryClass={q-fin.PM},
      url={https://arxiv.org/abs/2601.07991}, 
}
```

### APA
Sung, K., & Pirvu, T. A. (2026). Optimal option portfolios for skew-elliptical t returns. arXiv preprint arXiv:2601.07991. https://arxiv.org/abs/2601.07991

### Chicago
Sung, Kyle, and Traian A. Pirvu. "Optimal Option Portfolios for Skew-Elliptical t Returns." arXiv preprint arXiv:2601.07991 (2026). https://arxiv.org/abs/2601.07991


## Setup

### Prerequisites

- Python 3.11.5 (or compatible version)
- R with the `sn` and `jsonlite` packages, and `Rscript` on the PATH (the skew-t and Student's t distributions are fitted with `mst.mple` of `sn`)

### Installation

1. Clone the repository

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r installation/requirements.txt
   ```

4. **Set up Python path**:
   ```bash
   export PYTHONPATH=$(pwd)
   ```


## Acknowledgements

Traian A. Pirvu acknowledges that this work was supported by NSERC grant RGPIN-2019-05397.
