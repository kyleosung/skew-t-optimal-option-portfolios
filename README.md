# Optimal Option Portfolios for Skew-Elliptical t Returns

This repository contains the code for reproducing the results presented in the paper **Optimal Option Portfolios for Skew-Elliptical t Returns**.

## Overview

This paper explores option portfolio optimization when the underlying returns are skew-elliptical t-distributed. 
We use the variance and value at risk (VaR) to measure portfolio risk. 
The novelty of our work is the departure from the traditional normal returns setting, allowing investors to capture both heavy-tailed and skewed market dynamics. 
We provide explicit portfolio weights for the variance and VaR approximation. 
Our second contribution is the numerical representation of portfolio weights, obtained from numerical optimization for better VaR approximations. 
The effect of skewness on the portfolio weights is quantified by comparing our optimal skew t weights with those generated in the Student t setting. 
We also find that, as expected, a better VaR approximation risk measure yields optimal portfolio weights which are more different than the variance optimal weights. 

**Authors:** Kyle Sung and Traian A. Pirvu
**Institution:** Department of Mathematics and Statistics, McMaster University, Hamilton, ON, Canada  
**Correspondence To:** Kyle Sung, sungk5@mcmaster.ca
**Paper Link:** https://arxiv.org/abs/2601.07991

**Keywords:** Options, Optimal Portfolios, Value at Risk, Skew-Elliptical t Returns

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
