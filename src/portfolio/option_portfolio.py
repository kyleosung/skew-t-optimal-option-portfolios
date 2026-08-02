r"""
Option portfolio.

Portfolio optimization class for European options under the multivariate
skew-t distribution \Delta S ~ t_N^skew(\mu, \Sigma, \omega, \nu).

Setting \omega = 0 recovers the standard multivariate Student t-distribution as
a special case; all formulas below reduce to the standard-t results in that
limit.

Proposition (Expectation and Variance of Portfolio Gain for Skew-t Returns)
---------------------------------------------------------------------------
In Setting (Portfolio Optimization in Family of t-Distributions), define
the combined sensitivity vector \alpha = \Gamma\mu + \delta.

**Expectation**:

    E_Q[\Delta V(x)] = u^T x,   where   u := \zeta + c B h + c D^T h

    and \zeta := (\Delta t)\Theta + D^T\mu + \nu/(2(\nu-2)) p + \Xi.

**Variance**:

    Var_Q[\Delta V(x)] = (1/2) x^T Q x

    where Q incorporates skewness corrections from the Proposition.

    When omega = 0: u = \zeta and Q = U (standard t results).

**CFVaR3**:

    CFVaR_3^\alpha[\Delta V(x)]
        = -u^T x - \Phi^{-1}(\alpha) \sqrt{(1/2) x^T Q x}
          - (1/6)([\Phi^{-1}(\alpha)]^2 - 1) \kappa_3(x) / [(1/2) x^T Q x]

    where \kappa_3(x) is the third central moment (standard-t + skew-t
    corrections):

        \kappa_3(x) = [standard-t terms]
            + c * [skew-t first-order corrections]
            + c^2 * [skew-t second-order corrections]
            + 2 c^3 * (\alpha^T h)^3

    The standard-t terms are:
        [2\nu^3 / (\nu-2)^3(\nu-4)(\nu-6)] (x^T p)^3
        + [3\nu^3 / (\nu-2)^2(\nu-4)(\nu-6)] (x^T p)(x^T R x)
        + [3\nu^2 / (\nu-2)^2(\nu-4)] (x^T p)[x^T (D^T+B)^T \Sigma(D+B^T) x]
        + <\mathcal{T}, x \otimes x \otimes x>

    The skew-t corrections involve the portfolio-level sensitivity
    \alpha = (D + B^T) x, the skewness direction h, and the scaling
    constant c = \sqrt{\nu/\pi} \Gamma((\nu-1)/2) / \Gamma(\nu/2).

    When \omega = 0, h = 0 and all skew-t corrections vanish.

    We require \nu > 6 for the third moment to exist.
"""

from functools import cached_property
from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from src.constants.numerics import (
    NUM_RANDOM_STARTS,
    OPTIMIZATION_ABS_THRESHOLD,
    OPTIMIZATION_FTOL,
    OPTIMIZATION_MAXITER,
)
from src.options.european_call import EuropeanCall
from src.options.european_put import EuropeanPut
from src.options.option import Underlying
from src.portfolio.utils import number_of_shares_to_weights, weights_to_number_of_shares
from src.utils.linalg import invert_matrix_with_regularization


class OptionPortfolio:
    r"""
    Unified portfolio class for European options under the skew-t distribution.

    Assumes \Delta S ~ t_N^skew(\mu, \Sigma, \omega, \nu).  Setting
    omega = 0 recovers the standard multivariate Student t-distribution,
    so every property and optimization method works for both settings.
    """

    def __init__(
        self,
        underlyings: list[Underlying],
        options: list[EuropeanCall | EuropeanPut],
        time_period_for_risk_measurements: float,
        scale_matrix: np.ndarray,
        omega: Optional[np.ndarray] = None,
        degrees_of_freedom: float = 5.0,
        returns: Optional[np.ndarray] = None,
    ) -> None:
        r"""
        Initialize the unified option portfolio.

        Parameters
        ----------
        underlyings : list[Underlying]
            The list of underlying assets for the options in the portfolio.
        options : list[EuropeanCall | EuropeanPut]
            The list of options to include in the portfolio.
        time_period_for_risk_measurements : float
            The time period \Delta t for risk measurements.
        scale_matrix : np.ndarray
            The scale matrix \Sigma of the skew-t distribution (N by N).
            For the skew-t with \nu degrees of freedom:
            Cov[\Delta S] = \nu/(\nu-2) \Sigma (in the symmetric case).
        omega : Optional[np.ndarray], optional
            The skewness parameter vector \omega (N-dimensional).
            When omega = 0 (default), the distribution reduces to the
            standard multivariate Student t.
        degrees_of_freedom : float, optional
            The degrees of freedom \nu, default 5.
            Must be > 4 for second moments to exist and > 6 for CFVaR3.
        returns : Optional[np.ndarray], optional
            The ANNUALIZED expected log returns for the underlying assets.
            Converted to absolute price drift: \mu = S * returns * \Delta t.
            Defaults to zero (no drift).
        """
        self.underlyings = underlyings

        # Extract spot prices for drift calculation
        spot_prices = np.array(
            [underlying.spot for underlying in self.underlyings], dtype=float
        ).reshape(-1, 1)

        self.options = options

        self.underlying_map = {
            underlying.name: idx for idx, underlying in enumerate(self.underlyings)
        }

        self.N = len(self.underlyings)
        self.M = len(self.options)

        option_values_list = [option.price for option in self.options]
        self.option_values = np.array(option_values_list, dtype=float).reshape(-1, 1)

        self.delta_per_option_list = np.array(
            [option.delta for option in self.options], dtype=float
        ).reshape(-1, 1)
        self.gammas = np.array(
            [option.gamma for option in self.options], dtype=float
        ).reshape(-1, 1)
        self.thetas = np.array(
            [option.theta for option in self.options], dtype=float
        ).reshape(-1, 1)

        if returns is None:
            self.mu = np.zeros((self.N, 1))  # zero drift returns
        else:
            # mu represents the expected absolute price change E[\Delta S]
            # where \Delta S ~ t_N(mu, \Sigma, \nu) is the change in stock prices
            # For t-distributed returns: E[\Delta S] \approx S * r * \Delta t
            # where S is the spot price, r is the expected log return, and \Delta t is time period
            returns_reshaped = np.asarray(returns, dtype=float).reshape(-1, 1)
            self.mu = spot_prices * returns_reshaped * time_period_for_risk_measurements

        self.time_period_for_risk_measurements = time_period_for_risk_measurements
        self.scale_matrix = np.asarray(scale_matrix, dtype=float)

        if degrees_of_freedom <= 4:
            raise ValueError(
                "Degrees of freedom must be greater than 4 for the second moment to exist."
            )
        self.degrees_of_freedom = degrees_of_freedom

        # Skewness parameter omega (N by 1). Zero vector -> standard t.
        if omega is None:
            self.omega = np.zeros((self.N, 1))
        else:
            omega_array = np.asarray(omega, dtype=float)
            if omega_array.ndim == 0:
                self.omega = np.full((self.N, 1), float(omega_array))
            else:
                omega_array = omega_array.reshape(-1)
                if omega_array.size != self.N:
                    raise ValueError(
                        f"omega must have shape ({self.N},) or ({self.N}, 1); "
                        f"got {np.asarray(omega).shape}."
                    )
                self.omega = omega_array.reshape(-1, 1)

    def delta_vector(self, shares_vector: np.ndarray) -> np.ndarray:
        """
        Compute the portfolio delta vector showing total portfolio sensitivity to each underlying.

        The delta vector is an N-dimensional vector where element n represents:
        d(V)/d(S_n) - the total portfolio sensitivity to underlying asset n.

        This is computed as the sum of all option sensitivities to each underlying.
        When used in portfolio optimization with a shares vector, the portfolio
        delta becomes: portfolio_delta = delta_matrix.T @ shares_vector, where
        shares_vector contains the number of shares for each option.

        Returns
        -------
        np.ndarray
            N-dimensional delta vector where N is the number of underlyings,
            representing the sum of deltas for all options on each underlying
        """
        delta_vec = self.D @ shares_vector
        return delta_vec.reshape(-1, 1)

    def get_expected_portfolio_gain(self, shares_vector: np.ndarray) -> float:
        r"""
        Get the expected portfolio gain under the skew-t distribution.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.

        Returns
        -------
        float
            E_Q[\Delta V(x)] = u^T x

        Notes
        -----
        Uses the unified expected gain vector **u** (= \zeta when \omega = 0).
        """
        return (self.u.T @ shares_vector).item()

    def get_expected_portfolio_loss(self, shares_vector: np.ndarray) -> float:
        """
        Get the expected portfolio loss for a given shares vector.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.

        Returns
        -------
        float
            The expected portfolio loss
        """
        return -self.get_expected_portfolio_gain(shares_vector)

    def get_variance_portfolio_loss(self, shares_vector: np.ndarray) -> float:
        r"""
        Get the variance of the portfolio loss under the skew-t distribution.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.

        Returns
        -------
        float
            Var_Q[\Delta V(x)] = (1/2) x^T Q x

        Notes
        -----
        Uses the unified variance matrix **Q** (= U when \omega = 0).
        """
        return 0.5 * (shares_vector.T @ self.Q @ shares_vector).item()

    def get_CFVaR2_portfolio_loss(
        self, shares_vector: np.ndarray, alpha: float
    ) -> float:
        """
        Get the CFVaR2 (Cornish-Fisher Value at Risk) of the portfolio loss for a given shares vector and tail risk.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.
        alpha : float
            The tail risk for the CFVaR2 calculation.

        Returns
        -------
        float
            The CFVaR2 of the portfolio loss
        """
        return -self.get_expected_portfolio_gain(shares_vector) - np.sqrt(
            self.get_variance_portfolio_loss(shares_vector)
        ) * norm.ppf(alpha)

    def get_CFVaR3_portfolio_loss(
        self, shares_vector: np.ndarray, alpha: float
    ) -> float:
        """
        Get the CFVaR3 (Cornish-Fisher Value at Risk) of the portfolio loss for a given shares vector and tail risk.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.
        alpha : float
            The tail risk for the CFVaR3 calculation.

        Returns
        -------
        float
            The CFVaR3 of the portfolio loss
        """
        cfvar2 = self.get_CFVaR2_portfolio_loss(shares_vector, alpha)

        term3_coeff = -(1 / 6) * (norm.ppf(alpha) ** 2 - 1)
        kappa3 = self.kappa_third_moment(shares_vector)
        variance = self.get_variance_portfolio_loss(shares_vector)

        term3 = term3_coeff.item() * kappa3 / variance

        return cfvar2 + term3

    def get_CFES2_portfolio_loss(
        self, shares_vector: np.ndarray, alpha: float
    ) -> float:
        """
        Get the CFES2 (Cornish-Fisher Expected Shortfall, 2nd order) of the portfolio loss.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.
        alpha : float
            The tail risk (alpha) for the CFES2 calculation.

        Returns
        -------
        float
            The CFES2 of the portfolio loss

        Formula
        -------
        CFES_2^alpha = -zeta^T x + phi(Phi^{-1}(alpha)) / alpha * sqrt(0.5 * x^T U x)
        """
        z_alpha = norm.ppf(alpha)
        phi_z = norm.pdf(z_alpha)
        return -self.get_expected_portfolio_gain(shares_vector) + (
            phi_z / alpha
        ) * np.sqrt(self.get_variance_portfolio_loss(shares_vector))

    def get_CFES3_portfolio_loss(
        self, shares_vector: np.ndarray, alpha: float
    ) -> float:
        """
        Get the CFES3 (Cornish-Fisher Expected Shortfall, 3rd order) of the portfolio loss.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.
        alpha : float
            The tail risk (alpha) for the CFES3 calculation.

        Returns
        -------
        float
            The CFES3 of the portfolio loss

        Formula
        -------
        CFES_3^alpha = CFES_2^alpha
            + (1/6) * (Phi^{-1}(alpha) * phi(Phi^{-1}(alpha)) / alpha)
              * kappa_3(x) / (0.5 * x^T U x)
        """
        cfes2 = self.get_CFES2_portfolio_loss(shares_vector, alpha)

        z_alpha = norm.ppf(alpha)
        phi_z = norm.pdf(z_alpha)
        kappa3 = self.kappa_third_moment(shares_vector)
        variance = self.get_variance_portfolio_loss(shares_vector)

        term3 = (1 / 6) * (z_alpha * phi_z / alpha) * kappa3 / variance

        return (cfes2 + term3).item()

    def get_optimal_variance_weights_lagrange(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal variance number of shares and weights for
        the portfolio using the analytical Lagrange multiplier solution.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        v = self.option_values
        Q_inv = self.Q_inv
        x = 1 / (v.T @ Q_inv @ v) * Q_inv @ v
        w = number_of_shares_to_weights(x, v)
        return x, w

    def get_optimal_CFVaR2_weights_lagrange(
        self, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal CFVaR2 number of shares and weights for
        the portfolio using the analytical Lagrange multiplier solution.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        A_cal = self.A_cal(alpha)
        B_cal = self.B_cal(alpha)
        C_cal = self.C_cal(alpha)

        discriminant = B_cal**2 - 4 * A_cal * C_cal

        if discriminant < 0 and abs(discriminant) < OPTIMIZATION_ABS_THRESHOLD:
            discriminant = 0.0

        epsilon_plus = (-B_cal + np.sqrt(discriminant)) / (2 * A_cal)
        epsilon_minus = (-B_cal - np.sqrt(discriminant)) / (2 * A_cal)

        psi_plus = self.psi(epsilon_plus.item())
        psi_minus = self.psi(epsilon_minus.item())

        x_plus = self.G @ psi_plus
        x_minus = self.G @ psi_minus

        VaR_plus = self.get_CFVaR2_portfolio_loss(x_plus, alpha)
        VaR_minus = self.get_CFVaR2_portfolio_loss(x_minus, alpha)

        x = x_plus if VaR_plus < VaR_minus else x_minus

        w = number_of_shares_to_weights(x, self.option_values)

        return x, w

    def get_optimal_CFES2_weights_lagrange(
        self, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal CFES2 number of shares and weights for
        the portfolio using the analytical Lagrange multiplier solution.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFES2 calculation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        A_cal_ES = self.A_cal_ES(alpha)
        B_cal_ES = self.B_cal_ES(alpha)
        C_cal_ES = self.C_cal_ES(alpha)

        discriminant = B_cal_ES**2 - 4 * A_cal_ES * C_cal_ES

        if discriminant < 0 and abs(discriminant) < OPTIMIZATION_ABS_THRESHOLD:
            discriminant = 0.0

        epsilon_plus = (-B_cal_ES + np.sqrt(discriminant)) / (2 * A_cal_ES)
        epsilon_minus = (-B_cal_ES - np.sqrt(discriminant)) / (2 * A_cal_ES)

        psi_plus = self.psi(epsilon_plus.item())
        psi_minus = self.psi(epsilon_minus.item())

        x_plus = self.G @ psi_plus
        x_minus = self.G @ psi_minus

        ES_plus = self.get_CFES2_portfolio_loss(x_plus, alpha)
        ES_minus = self.get_CFES2_portfolio_loss(x_minus, alpha)

        x = x_plus if ES_plus < ES_minus else x_minus

        w = number_of_shares_to_weights(x, self.option_values)

        return x, w

    ###############################################################################################
    # Numeric optimization helper methods
    ###############################################################################################

    def _process_weight_bounds(
        self,
        lower_bounds: Optional[Union[np.ndarray, float]],
        upper_bounds: Optional[Union[np.ndarray, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process and validate weight bounds for optimization.

        Parameters
        ----------
        lower_bounds : Optional[Union[np.ndarray, float]]
            Lower bounds for the weights. If None, no lower bound is applied (-inf).
            Can be a scalar (applied to all options) or an array of shape (M,).
        upper_bounds : Optional[Union[np.ndarray, float]]
            Upper bounds for the weights. If None, no upper bound is applied (inf).
            Can be a scalar (applied to all options) or an array of shape (M,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Processed lower and upper bounds arrays of shape (M,).

        Raises
        ------
        ValueError
            If bounds have incorrect length or if lower_bounds > upper_bounds for any element.
        """
        if lower_bounds is None:
            lb = np.full(self.M, -np.inf)
        elif np.isscalar(lower_bounds):
            lb = np.full(self.M, lower_bounds)
        else:
            lb = np.asarray(lower_bounds).flatten()

        if upper_bounds is None:
            ub = np.full(self.M, np.inf)
        elif np.isscalar(upper_bounds):
            ub = np.full(self.M, upper_bounds)
        else:
            ub = np.asarray(upper_bounds).flatten()

        if lb.shape[0] != self.M:
            raise ValueError(
                f"lower_bounds must have length {self.M}, got {lb.shape[0]}"
            )
        if ub.shape[0] != self.M:
            raise ValueError(
                f"upper_bounds must have length {self.M}, got {ub.shape[0]}"
            )
        if np.any(lb > ub):
            raise ValueError("lower_bounds must be <= upper_bounds for all elements")

        return lb, ub

    def _run_weight_optimization(
        self,
        objective_fn,
        lower_bounds: Optional[Union[np.ndarray, float]] = None,
        upper_bounds: Optional[Union[np.ndarray, float]] = None,
        warm_start_weights: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run weight-based optimization with optional box constraints.

        This internal method performs the core optimization loop, minimizing the
        given objective function over portfolio weights subject to the constraint
        that weights sum to 1, with optional lower and upper bounds on weights.

        Parameters
        ----------
        objective_fn : callable
            Function that takes weights (np.ndarray of shape (M,)) and returns a scalar.
        lower_bounds : Optional[Union[np.ndarray, float]], optional
            Lower bounds for the weights. If None, no lower bound is applied.
        upper_bounds : Optional[Union[np.ndarray, float]], optional
            Upper bounds for the weights. If None, no upper bound is applied.
        warm_start_weights : Optional[list], optional
            A list of weight vectors (each shape (M,)) to use as additional
            starting points for the optimizer, in addition to the random
            Dirichlet starts.  Useful for seeding with analytically-known
            solutions (e.g., the CFVaR2 Lagrange solution for CFVaR3).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains:
            - The optimal number of shares for the portfolio
            - The optimal weights vector

        Raises
        ------
        RuntimeError
            If numerical optimization fails to converge.
        """
        v = self.option_values

        # Process bounds
        lb, ub = self._process_weight_bounds(lower_bounds, upper_bounds)
        bounds = list(zip(lb, ub))

        # Constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Use multiple random starts to find the global minimum.
        # Each start draws an initial weight vector from Dirichlet(1,...,1), which is
        # the uniform distribution over the M-dimensional probability simplex.  This
        # gives diverse, properly-normalized starting points (all weights >= 0, sum = 1)
        # and avoids the solver converging to a local minimum near a single starting
        # point.  After clipping to the box constraints and re-normalising, each
        # candidate is used as the initial guess for SLSQP.
        best_result = None
        best_objective = float("inf")

        # Collect all starting points: warm starts first, then random Dirichlet
        starting_points = []

        if warm_start_weights is not None:
            for ws in warm_start_weights:
                w0 = np.asarray(ws).flatten()
                w0 = np.clip(w0, lb, ub)
                w_sum = np.sum(w0)
                if abs(w_sum) > OPTIMIZATION_ABS_THRESHOLD:
                    starting_points.append(w0 / w_sum)

        for _ in range(NUM_RANDOM_STARTS):
            # Generate initial guess respecting bounds
            w0 = np.random.dirichlet(np.ones(self.M))
            w0 = np.clip(w0, lb, ub)
            # Normalize to satisfy constraint; skip if sum is zero
            w_sum = np.sum(w0)
            if abs(w_sum) < OPTIMIZATION_ABS_THRESHOLD:
                continue
            starting_points.append(w0 / w_sum)

        for w0 in starting_points:
            result = minimize(
                objective_fn,
                w0,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"ftol": OPTIMIZATION_FTOL, "maxiter": OPTIMIZATION_MAXITER},
            )
            if result.success and result.fun < best_objective:
                best_objective = result.fun
                best_result = result

        if best_result is None:
            raise RuntimeError("Numerical optimization failed to converge")

        w_optimal = best_result.x

        w_optimal = w_optimal.reshape(-1, 1)
        x_optimal = weights_to_number_of_shares(w_optimal, v)
        x_optimal = x_optimal.reshape(-1, 1)

        return x_optimal, w_optimal

    def get_optimal_variance_weights_numeric(
        self,
        lower_bounds: Optional[Union[np.ndarray, float]] = None,
        upper_bounds: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal variance number of shares and weights for
        the portfolio using numerical optimization.

        Uses scipy.optimize.minimize with SLSQP method to find the
        weights that minimize the portfolio variance subject to the
        constraint that weights sum to 1 (i.e., sum(w) = 1).

        Parameters
        ----------
        lower_bounds : Optional[Union[np.ndarray, float]], optional
            Lower bounds for the weights (min percentage per option).
            If None, no lower bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).
        upper_bounds : Optional[Union[np.ndarray, float]], optional
            Upper bounds for the weights (max percentage per option).
            If None, no upper bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains:
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        v = self.option_values

        def objective(w: np.ndarray) -> float:
            """Objective function: minimize variance."""
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            return self.get_variance_portfolio_loss(x_tensor.reshape(-1, 1))

        return self._run_weight_optimization(objective, lower_bounds, upper_bounds)

    def get_optimal_CFVaR2_weights_numeric(
        self,
        alpha: float,
        lower_bounds: Optional[Union[np.ndarray, float]] = None,
        upper_bounds: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal CFVaR2 number of shares and weights for
        the portfolio using numerical optimization.

        Uses scipy.optimize.minimize with SLSQP method to find the
        weights that minimize the CFVaR2 subject to the constraint
        that weights sum to 1 (i.e., sum(w) = 1).

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.
        lower_bounds : Optional[Union[np.ndarray, float]], optional
            Lower bounds for the weights (min percentage per option).
            If None, no lower bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).
        upper_bounds : Optional[Union[np.ndarray, float]], optional
            Upper bounds for the weights (max percentage per option).
            If None, no upper bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains:
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        v = self.option_values

        def objective(w: np.ndarray) -> float:
            """Objective function: minimize CFVaR2."""
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            return self.get_CFVaR2_portfolio_loss(x_tensor.reshape(-1, 1), alpha)

        return self._run_weight_optimization(objective, lower_bounds, upper_bounds)

    def get_optimal_CFVaR3_weights_numeric(
        self,
        alpha: float,
        lower_bounds: Optional[Union[np.ndarray, float]] = None,
        upper_bounds: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal CFVaR3 number of shares and weights for
        the portfolio using numerical optimization.

        Uses scipy.optimize.minimize with SLSQP method to find the
        weights that minimize the CFVaR3 subject to the constraint
        that weights sum to 1 (i.e., sum(w) = 1).

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR3 calculation.
        lower_bounds : Optional[Union[np.ndarray, float]], optional
            Lower bounds for the weights (min percentage per option).
            If None, no lower bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).
        upper_bounds : Optional[Union[np.ndarray, float]], optional
            Upper bounds for the weights (max percentage per option).
            If None, no upper bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains:
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        v = self.option_values

        if self.degrees_of_freedom <= 6:
            raise ValueError(
                "Degrees of freedom must be greater than 6 for the third moment to exist, "
                "which is required for CFVaR3 optimization."
            )

        def objective(w: np.ndarray) -> float:
            """Objective function: minimize CFVaR3."""
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            return self.get_CFVaR3_portfolio_loss(x_tensor.reshape(-1, 1), alpha)

        # Seed the optimizer with the analytical CFVaR2 Lagrange solution,
        # since CFVaR3 = CFVaR2 + third-moment correction and the CFVaR3
        # optimum is typically near the CFVaR2 optimum.
        _, w_cfvar2 = self.get_optimal_CFVaR2_weights_lagrange(alpha)
        warm_starts = [w_cfvar2.flatten()]

        return self._run_weight_optimization(
            objective, lower_bounds, upper_bounds, warm_start_weights=warm_starts
        )

    def get_optimal_CFES2_weights_numeric(
        self,
        alpha: float,
        lower_bounds: Optional[Union[np.ndarray, float]] = None,
        upper_bounds: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal CFES2 number of shares and weights for
        the portfolio using numerical optimization.

        Uses scipy.optimize.minimize with SLSQP method to find the
        weights that minimize the CFES2 subject to the constraint
        that weights sum to 1 (i.e., sum(w) = 1).

        Parameters
        ----------
        alpha : float
            The tail risk for the CFES2 calculation.
        lower_bounds : Optional[Union[np.ndarray, float]], optional
            Lower bounds for the weights (min percentage per option).
            If None, no lower bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).
        upper_bounds : Optional[Union[np.ndarray, float]], optional
            Upper bounds for the weights (max percentage per option).
            If None, no upper bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains:
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        v = self.option_values

        def objective(w: np.ndarray) -> float:
            """Objective function: minimize CFES2."""
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            return self.get_CFES2_portfolio_loss(x_tensor.reshape(-1, 1), alpha)

        return self._run_weight_optimization(objective, lower_bounds, upper_bounds)

    def get_optimal_CFES3_weights_numeric(
        self,
        alpha: float,
        lower_bounds: Optional[Union[np.ndarray, float]] = None,
        upper_bounds: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal CFES3 number of shares and weights for
        the portfolio using numerical optimization.

        Uses scipy.optimize.minimize with SLSQP method to find the
        weights that minimize the CFES3 subject to the constraint
        that weights sum to 1 (i.e., sum(w) = 1).

        Parameters
        ----------
        alpha : float
            The tail risk for the CFES3 calculation.
        lower_bounds : Optional[Union[np.ndarray, float]], optional
            Lower bounds for the weights (min percentage per option).
            If None, no lower bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).
        upper_bounds : Optional[Union[np.ndarray, float]], optional
            Upper bounds for the weights (max percentage per option).
            If None, no upper bound is applied. Default is None.
            Can be a scalar (applied to all options) or an array of shape (M,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains:
            - The optimal number of shares for the portfolio
            - The optimal weights vector scaled by the option values
        """
        v = self.option_values

        if self.degrees_of_freedom <= 6:
            raise ValueError(
                "Degrees of freedom must be greater than 6 for the third moment to exist, "
                "which is required for CFES3 optimization."
            )

        def objective(w: np.ndarray) -> float:
            """Objective function: minimize CFES3."""
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            return self.get_CFES3_portfolio_loss(x_tensor.reshape(-1, 1), alpha)

        return self._run_weight_optimization(objective, lower_bounds, upper_bounds)

    ###############################################################################################
    # Sharpe Ratio optimization methods
    ###############################################################################################

    def get_sharpe_ratio(
        self,
        shares_vector: np.ndarray,
        annual_risk_free_rate: float,
    ) -> float:
        """
        Compute the Sharpe ratio of the portfolio.
        The Sharpe ratio is defined as:
        SR = (E[\\Delta V(x)] - r_f * \\Delta t) / sqrt(Var[\\Delta V(x)])
        where E[\\Delta V(x)] is the expected portfolio return (negative of expected loss),
        r_f * \\Delta t is the risk-free rate over the time period, and
        Var[\\Delta V(x)] is the portfolio variance.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.
        annual_risk_free_rate : float
            The annualized risk-free interest rate. It will be scaled by
            time_period_for_risk_measurements (\\Delta t) internally.

        Returns
        -------
        float
            The Sharpe ratio of the portfolio
        """
        shares_vector = np.asarray(shares_vector, dtype=float)

        expected_return = self.get_expected_portfolio_gain(shares_vector)

        variance = self.get_variance_portfolio_loss(shares_vector)
        std_dev = np.sqrt(variance)

        risk_free_rate_period = (
            annual_risk_free_rate * self.time_period_for_risk_measurements
        )
        sharpe_ratio = (expected_return - risk_free_rate_period) / std_dev

        return float(sharpe_ratio)

    def get_optimal_sharpe_ratio_weights_lagrange(
        self,
        annual_risk_free_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal weights for maximum Sharpe ratio using analytical solution.
        The analytical solution is:
        x = U^{-1}(zeta - r_f * \\Delta t * v) / (v^T @ U^{-1} @ (zeta - r_f * \\Delta t * v))
        where U is the variance matrix with drift, zeta is the expected loss vector with drift,
        v is the option values vector, and r_f * \\Delta t is the risk-free rate over the period.

        Parameters
        ----------
        annual_risk_free_rate : float
            The annualized risk-free interest rate. It will be scaled by
            time_period_for_risk_measurements (\\Delta t) internally.

        Returns
        -------
        np.ndarray
            The optimal number of shares for maximum Sharpe ratio
        np.ndarray
            The weights vector scaled by the option values
        """
        v = self.option_values
        u = self.u
        Q_inv = self.Q_inv

        # Compute u - r_f * v
        risk_free_rate_period = (
            annual_risk_free_rate * self.time_period_for_risk_measurements
        )
        u_minus_rf_v = u - risk_free_rate_period * v

        # Compute Q^{-1} @ (u - r_f * v)
        numerator = Q_inv @ u_minus_rf_v

        # Compute v^T @ Q^{-1} @ (u - r_f * v)
        denominator = v.T @ numerator

        x = numerator / denominator
        weights = number_of_shares_to_weights(x, v)

        return x, weights

    def get_optimal_sharpe_ratio_weights_numeric(
        self,
        annual_risk_free_rate: float,
        lower_bounds: Optional[Union[float, np.ndarray]] = 0.0,
        upper_bounds: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal weights for maximum Sharpe ratio using numerical optimization.
        Note: We maximize the Sharpe ratio (not minimize), so we negate the objective function.

        Parameters
        ----------
        annual_risk_free_rate : float
            The annualized risk-free interest rate. It will be scaled by
            time_period_for_risk_measurements (\\Delta t) internally.
        lower_bounds : Optional[Union[float, np.ndarray]], optional
            Lower bounds for the weights. Defaults to 0.0 (no short selling).
        upper_bounds : Optional[Union[float, np.ndarray]], optional
            Upper bounds for the weights. Defaults to None (no upper bound).

        Returns
        -------
        np.ndarray
            The optimal number of shares for maximum Sharpe ratio
        np.ndarray
            The weights vector scaled by the option values
        """
        v = self.option_values

        def objective(w: np.ndarray) -> float:
            """
            Objective function: maximize Sharpe ratio (minimize negative Sharpe ratio).

            Parameters
            ----------
            w : np.ndarray
                Weights vector of shape (M,)

            Returns
            -------
            float
                Negative Sharpe ratio to be minimized
            """
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            sharpe_ratio = self.get_sharpe_ratio(
                x_tensor.reshape(-1, 1), annual_risk_free_rate
            )
            # Return negative because we want to maximize Sharpe ratio
            return -sharpe_ratio

        return self._run_weight_optimization(objective, lower_bounds, upper_bounds)

    def get_returnVaR_ratio(
        self, shares_vector: np.ndarray, alpha: float, annual_risk_free_rate: float
    ) -> float:
        """
        Compute the return-to-VaR ratio of the portfolio.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.
        alpha : float
            The tail risk for the CFVaR2 calculation.
        annual_risk_free_rate : float
            The annualized risk-free interest rate. It will be scaled by
            time_period_for_risk_measurements (\\Delta t) internally.

        Returns
        -------
        float
            The return-to-VaR ratio of the portfolio
        """
        shares_vector = np.asarray(shares_vector, dtype=float)

        expected_return = -self.get_expected_portfolio_loss(shares_vector)

        cfvar2 = self.get_CFVaR2_portfolio_loss(shares_vector, alpha=alpha)

        risk_free_rate_period = (
            annual_risk_free_rate * self.time_period_for_risk_measurements
        )
        returnVaR_ratio = (expected_return - risk_free_rate_period) / cfvar2

        return float(returnVaR_ratio)

    def get_optimal_returnVaR_ratio_lagrange(
        self, alpha: float, annual_risk_free_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal weights for maximum return-to-VAR ratio using analytical solution.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.
        annual_risk_free_rate : float
            The annualized risk-free interest rate. It will be scaled by
            time_period_for_risk_measurements (\\Delta t) internally.

        Returns
        -------
        np.ndarray
            The optimal number of shares for maximum return-to-VAR ratio
        np.ndarray
            The weights vector scaled by the option values
        """
        v = self.option_values
        Q_inv = self.Q_inv
        u = self.u

        alpha = alpha
        r = annual_risk_free_rate * self.time_period_for_risk_measurements

        A_var_sharpe = u.T @ Q_inv @ u - norm.ppf(alpha) ** 2 / 2
        B_var_sharpe = 2 * r * u.T @ Q_inv @ v - 2 * u.T @ Q_inv @ u
        C_var_sharpe = (
            u.T @ Q_inv @ u - 2 * r * u.T @ Q_inv @ v + r**2 * v.T @ Q_inv @ v
        )

        discriminant = B_var_sharpe**2 - 4 * A_var_sharpe * C_var_sharpe
        discriminant = float(discriminant)

        if discriminant < 0 and abs(discriminant) < OPTIMIZATION_ABS_THRESHOLD:
            discriminant = 0.0

        lambda_plus = (-B_var_sharpe + np.sqrt(discriminant)) / (2 * A_var_sharpe)

        lambda_minus = (-B_var_sharpe - np.sqrt(discriminant)) / (2 * A_var_sharpe)

        numerator_plus = Q_inv @ ((1 - lambda_plus) * u - r * v)
        denominator_plus = v.T @ numerator_plus
        x_plus = numerator_plus / denominator_plus

        numerator_minus = Q_inv @ ((1 - lambda_minus) * u - r * v)
        denominator_minus = v.T @ numerator_minus
        x_minus = numerator_minus / denominator_minus

        returnVaR_plus = self.get_returnVaR_ratio(x_plus, alpha, annual_risk_free_rate)
        returnVaR_minus = self.get_returnVaR_ratio(
            x_minus, alpha, annual_risk_free_rate
        )

        x = x_plus if returnVaR_plus > returnVaR_minus else x_minus
        weights = number_of_shares_to_weights(x, v)

        return x, weights

    def get_optimal_returnVaR_ratio_numeric(
        self,
        alpha: float,
        annual_risk_free_rate: float,
        lower_bounds: Optional[Union[float, np.ndarray]] = 0.0,
        upper_bounds: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimal weights for maximum return-to-VaR ratio using numerical optimization.
        Note: We maximize the return-to-VaR ratio (not minimize), so we negate the objective function.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.
        annual_risk_free_rate : float
            The annualized risk-free interest rate. It will be scaled by
            time_period_for_risk_measurements (\\Delta t) internally.
        lower_bounds : Optional[Union[float, np.ndarray]], optional
            Lower bounds for the weights. Defaults to 0.0 (no short selling).
        upper_bounds : Optional[Union[float, np.ndarray]], optional
            Upper bounds for the weights. Defaults to None (no upper bound).

        Returns
        -------
        np.ndarray
            The optimal number of shares for maximum return-to-VaR ratio
        np.ndarray
            The weights vector scaled by the option values
        """
        v = self.option_values

        def objective(w: np.ndarray) -> float:
            """
            Objective function: maximize return-to-VaR ratio (minimize negative return-to-VaR ratio).

            Parameters
            ----------
            w : np.ndarray
                Weights vector of shape (M,)

            Returns
            -------
            float
                Negative return-to-VaR ratio to be minimized
            """
            w_tensor = w.reshape(-1, 1)
            x_tensor = weights_to_number_of_shares(w_tensor, v)
            returnVaR_ratio = self.get_returnVaR_ratio(
                x_tensor.reshape(-1, 1), alpha, annual_risk_free_rate
            )
            # Return negative because we want to maximize return-to-VaR ratio
            return -returnVaR_ratio

        return self._run_weight_optimization(objective, lower_bounds, upper_bounds)

    ###############################################################################################
    # Helper functions to compute components of variance and CFVaR2
    ###############################################################################################

    def kappa_third_moment(self, shares_vector: np.ndarray) -> float:
        r"""
        Compute the third central moment \kappa_3(x) of the portfolio gain
        under the skew-t distribution.

        Parameters
        ----------
        shares_vector : np.ndarray
            The vector of number of shares of the options in the portfolio.

        Returns
        -------
        float
            The third central moment \kappa_3(x).

        Raises
        ------
        ValueError
            If degrees_of_freedom <= 6 (third moment does not exist).

        Formula
        -------
        The standard-t terms are:

        \kappa_3(x) =
            [2\nu^3 / (\nu-2)^3(\nu-4)(\nu-6)]  (x^T p)^3
          + [3\nu^3 / (\nu-2)^2(\nu-4)(\nu-6)]  (x^T p)(x^T R x)
          + [3\nu^2 / (\nu-2)^2(\nu-4)]          (x^T p)[x^T (D^T+B)^T\Sigma(D+B^T) x]
          + <\mathcal{T}, x \otimes x \otimes x>

        The skew-t corrections add c*[...] + c^2*[...] + 2c^3*(...) terms
        involving the skewness direction vector h and portfolio-level
        sensitivity vector \alpha = (D + B^T) x.  When \omega = 0, h = 0
        and all corrections vanish, recovering the standard-t result.
        """
        nu = self.degrees_of_freedom
        if nu <= 6:
            raise ValueError(
                f"kappa_third_moment requires degrees_of_freedom > 6; got {nu}."
            )
        x = shares_vector.reshape(-1)  # required to pass to einsum
        T = self.T_tensor  # combined tensor matching \mathcal{T} of paper1
        p = self.p
        R = self.R
        D = self.D
        B = self.B
        SIGMA = self.scale_matrix

        coeff1 = (2 * nu**3) / ((nu - 2) ** 3 * (nu - 4) * (nu - 6))
        coeff3 = (3 * nu**3) / ((nu - 2) ** 2 * (nu - 4) * (nu - 6))
        coeff5 = (3 * nu**2) / ((nu - 2) ** 2 * (nu - 4))

        term1 = coeff1 * (shares_vector.T @ p) ** 3
        term3 = coeff3 * (x.T @ p) * (x.T @ R @ x)
        term5 = coeff5 * (x.T @ p) * (x.T @ ((D.T + B) @ SIGMA @ (D.T + B).T) @ x)
        term_T = np.einsum("ijk,i,j,k->", T, x, x, x, optimize=True)

        kappa3_std = term1 + term3 + term5 + term_T

        # ------------------------------------------------------------------
        # Skew-t correction terms (vanish when omega = 0 since h = 0)
        # ------------------------------------------------------------------
        h = self.h_skew  # N x 1
        h_flat = h.ravel()

        # Fast path: if h is exactly 0 (standard t), skip all correction algebra.
        if np.dot(h_flat, h_flat) == 0.0:
            return np.asarray(kappa3_std).item()

        c = self.c_skew

        # Portfolio-level Gamma_port diagonal via precomputed gamma diagonals.
        # Since each Gamma^[m] is diagonal, Gamma_port = sum_m x_m Gamma^[m]
        # is also diagonal.  We represent it as a vector g of length N.
        g = x @ self._gamma_diags  # (N,) diagonal of Gamma_port

        # Portfolio sensitivity vector: alpha = (D + B^T) x  (N x 1)
        x_col = x.reshape(-1, 1)
        alpha = (D + B.T) @ x_col

        # Precompute common vectors exploiting diagonal Gamma_port:
        #   Gamma_port @ v  =  g * v   (element-wise)
        alpha_flat = alpha.ravel()
        Gh = g * h_flat  # Gamma_port @ h  (N,)
        S_Gh = SIGMA @ Gh  # Sigma @ Gamma_port @ h  (N,)
        S_alpha = SIGMA @ alpha_flat  # Sigma @ alpha  (N,)

        # Scalar intermediate quantities (no full NxN matrix ops)
        a_h = alpha_flat @ h_flat  # alpha^T h
        trGS = float((x_col.T @ p).item())  # tr[Gamma Sigma] = x^T p
        trGSGS = float(x @ R @ x)  # tr[Gamma Sigma Gamma Sigma] = x^T R x
        a_Sa = float(alpha_flat @ S_alpha)  # alpha^T Sigma alpha
        a_SGh = float(alpha_flat @ S_Gh)  # alpha^T Sigma Gamma h
        hGh = float(h_flat @ Gh)  # h^T Gamma h
        # h^T Gamma_port Sigma Gamma_port h = (Gh)^T Sigma (Gh)

        hGSGh = float(Gh @ S_Gh)
        # alpha^T Sigma Gamma_port Sigma Gamma_port h
        #   = alpha^T Sigma (g * (Sigma (g * h)))
        a_SGSGh = float(alpha_flat @ (SIGMA @ (g * S_Gh)))

        # c * [...] block
        sc1 = (6 * nu**2 / ((nu - 3) * (nu - 5))) * a_SGSGh
        sc2 = (-3 * nu**2 / ((nu - 3) * (nu - 5))) * (a_SGh * hGh + a_h * hGSGh)
        sc3 = (9 * nu**2 / (4 * (nu - 3) * (nu - 5))) * a_h * hGh**2
        sc4 = (9 * nu**2 / ((nu - 2) * (nu - 3) * (nu - 5))) * a_SGh * trGS
        sc5 = (-nu / (nu - 3)) * a_h**3
        sc6 = (-9 * nu**2 / (2 * (nu - 2) * (nu - 3) * (nu - 5))) * (a_h * hGh * trGS)
        sc7 = (
            (3 * nu**2 * (2 * nu - 7) / (2 * (nu - 2) * (nu - 3) * (nu - 4) * (nu - 5)))
            * a_h
            * trGSGS
        )
        sc8 = (
            (
                3
                * nu**2
                * (7 * nu - 26)
                / (4 * (nu - 2) ** 2 * (nu - 3) * (nu - 4) * (nu - 5))
            )
            * a_h
            * trGS**2
        )
        sc9 = (3 * nu / ((nu - 2) * (nu - 3))) * a_h * a_Sa

        c_block = c * (sc1 + sc2 + sc3 + sc4 + sc5 + sc6 + sc7 + sc8 + sc9)

        # c^2 * [...] block
        sc2_1 = (3 * nu / (nu - 3)) * (a_h**2 * hGh - 2 * a_h * a_SGh)
        sc2_2 = (-3 * nu / ((nu - 2) * (nu - 3))) * a_h**2 * trGS

        c2_block = c**2 * (sc2_1 + sc2_2)

        # c^3 term
        c3_block = 2 * c**3 * a_h**3

        kappa3 = kappa3_std + c_block + c2_block + c3_block

        return np.asarray(kappa3).item()

    @cached_property
    def R(self) -> np.ndarray:
        """
        Compute the R matrix used in variance and CFVaR2 calculations.

        Returns
        -------
        np.ndarray
            The R matrix

        Formula
        -------
        $\tr \\!
            \\left[
                \\Gamma^{[i]} \\SIGMA \\Gamma^{[j]} \\SIGMA
            \\right]_{i=1, \\ldots, M}^{j=1, \\ldots, M}$
        """
        scale_matrix = self.scale_matrix

        R = np.zeros((self.M, self.M))

        for i in range(self.M):
            for j in range(self.M):
                Gamma_i = self.__get_gamma_k(i)
                Gamma_j = self.__get_gamma_k(j)

                matrix_product = Gamma_i @ scale_matrix @ Gamma_j @ scale_matrix

                R[i, j] = np.trace(matrix_product)
        return R

    @cached_property
    def p(self) -> np.ndarray:
        """
        Compute the p vector used in variance and CFVaR2 calculations.

        Returns
        -------
        np.ndarray
            The p vector

        Formula
        -------
        $\\coloneq (p_1,\\ldots,p_M)$, \\quad  $p_m\\coloneq \\tr
            \\left[
                 \\Gamma^{[m]}
                \\SIGMA
            \\right]$
        """
        scale_matrix = self.scale_matrix

        p = np.zeros((self.M, 1))
        for m in range(self.M):
            gamma_m = self.__get_gamma_k(m)
            matrix_product = gamma_m @ scale_matrix
            p[m, 0] = np.trace(matrix_product)

        return p

    @cached_property
    def B(self) -> np.ndarray:
        """
        B is an M x N matrix where each row m contains the product of the drift vector (mu)
        with the gamma matrix for option m, i.e., B[m, :] = mu.T @ Gamma^[m].
        Each row m corresponds to option m, not underlying m.
        This means that row m contains the drift vector multiplied by the gamma matrix for option m.

        Returns
        -------
        np.ndarray
            The B matrix.

        Formula
        -------
        $\\begin{bmatrix}
                --- & \\mu^\\top \\Gamma^{[1]} & \\ ---
                \\
                --- & ... & \\ ---
                \\
                --- & \\mu^\\top \\Gamma^{[M]} & \\ ---
            \\end{bmatrix} \\in \\R^{M \\times N}$
        """
        mu = self.mu

        B = np.zeros((self.M, self.N))

        for m in range(self.M):
            row_i = mu.T @ (self.__get_gamma_k(m))
            B[m, :] = row_i  # type: ignore

        return B

    @cached_property
    def U(self) -> np.ndarray:
        """
        Compute the matrix U which is the variance matrix of the portfolio loss.

        Returns
        -------
        np.ndarray
            The matrix U representing the variance of the portfolio loss.

        Formula
        -------
        U &\\coloneq
        \\left[\\dfrac{2\\nu}{\\nu -2} ((D^\\top+B)\\SIGMA(D^\\top + B)^\\top)
        + \\dfrac{\\nu^2}{(\\nu - 2) (\\nu -4)} R
        + \\dfrac{\\nu^2}{(\\nu-2)^2(\\nu-4)}\\p\\p^\\top\\right]
        """
        SIGMA = self.scale_matrix
        B = self.B
        D = self.D
        p = self.p
        R = self.R
        nu = self.degrees_of_freedom

        term1 = (2 * nu) / (nu - 2) * ((D.T + B) @ SIGMA @ (D.T + B).T)
        term2 = (nu**2) / ((nu - 2) * (nu - 4)) * R
        term3 = (nu**2) / ((nu - 2) ** 2 * (nu - 4)) * (p @ p.T)

        U = term1 + term2 + term3

        U = (U + U.T) / 2  # Ensure symmetry

        return U

    @cached_property
    def U_inv(self) -> np.ndarray:
        """
        Compute the inverse of the matrix U which is the variance of the portfolio loss.

        Returns
        -------
        np.ndarray
            The inverse of the matrix U representing the variance of the portfolio loss.
        """
        U = self.U
        U_inv = invert_matrix_with_regularization(U, "Variance quadratic form matrix U")
        return U_inv

    @cached_property
    def D(self) -> np.ndarray:
        """
        Compute the D matrix used in variance and CFVaR2 calculations.

        Returns
        -------
        np.ndarray
            The D matrix (N x M)

        Formula
        -----
        $\\coloneq \\left[\\pdv{V_m}{S_n}\right]_{n=1,\\ldots,N}^{m=1,\\ldots,M} \\in \\R^{N \\times M}$
        """
        delta_matrix = np.zeros((self.M, self.N))

        for m, option in enumerate(self.options):
            underlying_index = self.underlying_map[option.underlying.name]
            delta_matrix[m, underlying_index] = option.delta

        return delta_matrix.T

    def __get_gamma_k(self, k: int) -> np.ndarray:
        """
        Compute the Gamma_k matrix used in variance and CFVaR2 calculations.

        Parameters
        ----------
        k : int
            The index of the option.

        Returns
        -------
        np.ndarray
            The Gamma_k matrix

        Formula
        -------
        $\\coloneq \\left[\\gamma_{i,j}^{[m]}\\right]_{i=1,\\ldots,N}^{j=1,\\ldots,N} \\in \\R^{N\\times N},
        \\qquad \\displaystyle \\gamma_{i,j}^{[m]} \\coloneq \\pdv{V_m( \\S,t)}{S_i,S_j}$
        """
        Gamma_k = np.zeros((self.N, self.N))

        option = self.options[k]
        underlying_index = self.underlying_map[option.underlying.name]
        Gamma_k[underlying_index, underlying_index] = option.gamma

        return Gamma_k

    @cached_property
    def _gamma_diags(self) -> np.ndarray:
        """
        Precomputed gamma diagonals for all options, shape (M, N).

        ``_gamma_diags[m, n]`` is the gamma of option *m* with respect to
        underlying *n* (nonzero only at the option's own underlying index).
        Used for fast vectorized computation of portfolio-level Gamma_port
        diagonal: ``diag(Gamma_port) = x @ _gamma_diags``.
        """
        G = np.zeros((self.M, self.N))
        for m, option in enumerate(self.options):
            idx = self.underlying_map[option.underlying.name]
            G[m, idx] = option.gamma
        return G

    @cached_property
    def gamma_matrix(self) -> np.ndarray:
        """
        Compute the Gamma matrix used in variance and CFVaR2 calculations.

        Returns
        -------
        np.ndarray
            The Gamma matrix

        Formula
        -------
        $\\coloneq [\\gamma_{i,j}]_{i=1,\\ldots,N}^{j=1,\\ldots,N} \\in \\R^{N\times N}, \\qquad \\displaystyle
        \\gamma_{i,j} \\coloneq \\pdv{V (\\x; \\S,t)}{S_i,S_j}$
        """
        Gamma_matrix = np.zeros((self.N, self.N))

        for _, option in enumerate(self.options):
            underlying_index = self.underlying_map[option.underlying.name]
            Gamma_matrix[underlying_index, underlying_index] += option.gamma

        return Gamma_matrix

    @cached_property
    def J(self) -> np.ndarray:
        r"""
        Constraint matrix J used in Lagrange portfolio optimization.

        The matrix J is constructed by vertically stacking the transposed
        unified expected gain vector **u** and the transposed option values.

        Returns
        -------
        np.ndarray
            A 2 x M matrix with first row u^T and second row v^T.

        Formula
        -------
        $J = \begin{bmatrix} \mathbf{u}^\top \\ \mathbf{v}^\top \end{bmatrix}$

        Notes
        -----
        When \omega = 0, u = \zeta, recovering the standard-t constraint matrix.
        """
        return np.vstack([self.u.T, self.option_values.T])

    def __compute_xi_m(self, m: int) -> float:
        """
        Compute xi_m for a single option m using vectorized operations.

        xi_m = (1/2) * sum_{i=1}^N sum_{j=1}^N mu_i * mu_j * gamma_tilde_m^[i,j]
             = (1/2) * mu^T @ Gamma^[m] @ mu

        Parameters
        ----------
        m : int
            The index of the option.

        Returns
        -------
        float
            The xi_m value for option m

        Formula
        -------
        $\\xi_m = \\frac{1}{2} \\sum\\limits_{i=1}^N \\sum\\limits_{j=1}^N \\mu_i \\mu_j \\tilde{\\gamma}^{[i,j]}_m$
        """
        mu = self.mu
        Gamma_m = self.__get_gamma_k(m)

        result = 0.5 * (mu.T @ Gamma_m @ mu)

        return result.item()  # type: ignore

    @cached_property
    def xi(self) -> np.ndarray:
        """
        Returns the xi vector used in portfolio optimization.

        xi = (xi_1, ..., xi_M) where
        xi_m = (1/2) * sum_{i=1}^N sum_{j=1}^N mu_i * mu_j * gamma_tilde_m^[i,j]

        Returns
        -------
        np.ndarray
            The xi vector of shape (M, 1)
        """
        xi = np.zeros((self.M, 1))

        for m in range(self.M):
            xi[m, 0] = self.__compute_xi_m(m)

        return xi

    @cached_property
    def zeta(self) -> np.ndarray:
        """
        Returns the zeta vector used in portfolio optimization.

        zeta = u + D^T @ mu - xi

        Returns
        -------
        np.ndarray
            The zeta vector of shape (M, 1)

        Formula
        -------
        $\\ZETA \\coloneq
        \\left[ (\\Delta t) \\THETA + D^\\top \\MU + \\dfrac{\\nu}{2(\\nu - 2)} \\p + \\XI  \right]$
        """
        return (
            self.time_period_for_risk_measurements * self.thetas
            + self.D.T @ self.mu
            + (self.degrees_of_freedom / (2 * (self.degrees_of_freedom - 2))) * self.p
            + self.xi
        )

    ###########################################################################
    # Skew-t intermediate variables
    ###########################################################################

    @cached_property
    def c_skew(self) -> float:
        r"""
        Skewness scaling constant c.

        Returns
        -------
        float
            $c \coloneq \sqrt{\nu/\pi}\;\Gamma((\nu-1)/2)\,/\,\Gamma(\nu/2)$

        Notes
        -----
        Computed in log-space via ``gammaln`` to avoid overflow for large
        degrees of freedom (the direct ``gamma`` ratio overflows for nu >= 344).
        As nu -> inf the value converges to ``sqrt(2/pi) ~= 0.7979``.
        """
        nu = self.degrees_of_freedom
        return float(
            # np.sqrt(nu / np.pi) * gamma_func((nu - 1) / 2) / gamma_func(nu / 2)
            np.exp(0.5 * np.log(nu / np.pi) + gammaln((nu - 1) / 2) - gammaln(nu / 2))
        )

    @cached_property
    def h_skew(self) -> np.ndarray:
        r"""
        Skewness direction vector h (N by 1).

        Returns
        -------
        np.ndarray
            $h \coloneq \Sigma\omega / \sqrt{1 + \omega^\top\Sigma\omega}$

        Notes
        -----
        h = 0 when omega = 0, which ensures all skew corrections vanish.
        """
        SIGMA = self.scale_matrix
        omega = self.omega
        denominator = np.sqrt(1 + (omega.T @ SIGMA @ omega).item())
        return SIGMA @ omega / denominator

    @cached_property
    def Bh(self) -> np.ndarray:
        r"""
        Drift-gamma-skewness vector Bh (M by 1).

        Returns
        -------
        np.ndarray
            $(Bh)_m = \mu^\top\Gamma^{[m]} h$
        """
        return self.B @ self.h_skew

    @cached_property
    def Dh(self) -> np.ndarray:
        r"""
        Delta-skewness vector D^T h (M by 1).

        Returns
        -------
        np.ndarray
            $(D^\top h)_m = \sum_n D_{nm}\,h_n$
        """
        return self.D.T @ self.h_skew

    @cached_property
    def H_skew(self) -> np.ndarray:
        r"""
        Drift-gamma-scale-skewness matrix H (M by M).

        Returns
        -------
        np.ndarray
            $H_{i,j} = \mu^\top\Gamma^{[i]}\Sigma\Gamma^{[j]} h$
        """
        H = np.zeros((self.M, self.M))
        SIGMA = self.scale_matrix
        h = self.h_skew
        mu = self.mu

        for i in range(self.M):
            Gamma_i = self.__get_gamma_k(i)
            for j in range(self.M):
                Gamma_j = self.__get_gamma_k(j)
                H[i, j] = (mu.T @ Gamma_i @ SIGMA @ Gamma_j @ h).item()

        return H

    @cached_property
    def q_skew(self) -> np.ndarray:
        r"""
        Skewness-gamma quadratic vector q (M by 1).

        Returns
        -------
        np.ndarray
            $q_m = h^\top\Gamma^{[m]} h$
        """
        h = self.h_skew
        q = np.zeros((self.M, 1))

        for m in range(self.M):
            Gamma_m = self.__get_gamma_k(m)
            q[m, 0] = (h.T @ Gamma_m @ h).item()

        return q

    @cached_property
    def _delta_sigma_gamma_skewness_matrix(self) -> np.ndarray:
        r"""
        Delta-sigma-gamma-skewness interaction matrix E (M by M).

        Returns
        -------
        np.ndarray
            $E_{k,j} = D_{\cdot k}^\top\Sigma\Gamma^{[j]} h$
        """
        E = np.zeros((self.M, self.M))
        SIGMA = self.scale_matrix
        h = self.h_skew
        D_mat = self.D

        for k in range(self.M):
            D_k = D_mat[:, k].reshape(-1, 1)
            for j in range(self.M):
                Gamma_j = self.__get_gamma_k(j)
                E[k, j] = (D_k.T @ SIGMA @ Gamma_j @ h).item()

        return E

    @cached_property
    def _shifted_location_skewness_vector(self) -> np.ndarray:
        r"""
        Combined shifted-location skewness vector f (M by 1).

        Returns
        -------
        np.ndarray
            $f = Bh + D^\top h$, so that $\alpha^\top h = f^\top x$.
        """
        return self.Bh + self.Dh

    ###########################################################################
    # Unified expected gain vector u and variance matrix Q
    ###########################################################################

    @cached_property
    def u(self) -> np.ndarray:
        r"""
        Unified expected gain vector u (M by 1).

        Returns
        -------
        np.ndarray
            $\mathbf{u} \coloneq \zeta + c\,B h + c\,D^\top h$

        Notes
        -----
        E_Q[\Delta V(x)] = u^T x.
        When \omega = 0: h = 0, so u = \zeta (standard t result).
        """
        return self.zeta + self.c_skew * self.Bh + self.c_skew * self.Dh

    @cached_property
    def _Q_tilde(self) -> np.ndarray:
        r"""
        Skew-t variance matrix \tilde{Q} before symmetrization (M by M).

        Returns
        -------
        np.ndarray
            $\tilde{Q} = U
            + \frac{4c\nu}{\nu-3}(H+E)
            + \frac{2c\nu}{(\nu-2)(\nu-3)} f p^\top
            - \frac{2c\nu}{\nu-3} f q^\top
            - 2c^2 f f^\top$
        """
        nu = self.degrees_of_freedom
        c = self.c_skew
        p_vec = self.p
        f = self._shifted_location_skewness_vector
        q = self.q_skew
        H = self.H_skew
        E = self._delta_sigma_gamma_skewness_matrix
        U_std = self.U

        correction1 = (4 * c * nu / (nu - 3)) * (H + E)
        correction2 = (2 * c * nu / ((nu - 2) * (nu - 3))) * (f @ p_vec.T)
        correction3 = -(2 * c * nu / (nu - 3)) * (f @ q.T)
        correction4 = -(2 * c**2) * (f @ f.T)

        return U_std + correction1 + correction2 + correction3 + correction4

    @cached_property
    def Q(self) -> np.ndarray:
        r"""
        Unified variance matrix Q (M by M).

        Returns
        -------
        np.ndarray
            $Q \coloneq \tfrac{1}{2}(\tilde{Q} + \tilde{Q}^\top)$

        Notes
        -----
        Var_Q[\Delta V(x)] = (1/2) x^T Q x.
        When \omega = 0: Q = U (standard t result).
        """
        Q_tilde = self._Q_tilde
        return 0.5 * (Q_tilde + Q_tilde.T)

    @cached_property
    def Q_inv(self) -> np.ndarray:
        r"""
        Inverse of the unified variance matrix Q (M by M).

        Returns
        -------
        np.ndarray
            $Q^{-1}$
        """
        return invert_matrix_with_regularization(
            self.Q, "Unified variance quadratic form matrix Q"
        )

    def psi(self, epsilon: float) -> np.ndarray:
        """
        Returns the PSI vector used in portfolio optimization.

        PSI = [epsilon]
                 [ 1 ]

        Parameters
        ----------
        epsilon : float
            The epsilon parameter.

        Returns
        -------
        np.ndarray
            The PSI vector of shape (2, 1)
        """
        return np.array([epsilon, 1.0]).reshape(-1, 1)

    @cached_property
    def G(self) -> np.ndarray:
        r"""
        Constraint-to-asset mapping matrix G (M by 2).

        Returns
        -------
        np.ndarray
            The G matrix used in Lagrange optimization. Shape (M, 2).

        Formula
        -------
        $G = Q^{-1} J^\top (J Q^{-1} J^\top)^{-1}$

        Notes
        -----
        When \omega = 0, Q = U and J uses \zeta, recovering the standard-t G.
        """
        Q_inv = self.Q_inv
        J = self.J

        J_Q_inv_J_T_inv = self.J_U_inv_J_T_inv

        G = Q_inv @ J.T @ J_Q_inv_J_T_inv
        return G

    @cached_property
    def J_U_inv_J_T_inv(self) -> np.ndarray:
        r"""
        Inverse of J Q^{-1} J^T (2 by 2).

        Returns
        -------
        np.ndarray
            The 2 x 2 inverse matrix $(J Q^{-1} J^\top)^{-1}$.

        Notes
        -----
        Uses the unified variance matrix Q (= U when \omega = 0).
        The name is kept for backward compatibility.
        """
        Q_inv = self.Q_inv
        J = self.J

        J_Q_inv_J_T = J @ Q_inv @ J.T

        J_Q_inv_J_T_inv = invert_matrix_with_regularization(J_Q_inv_J_T)

        return J_Q_inv_J_T_inv

    @cached_property
    def T_tensor(self) -> np.ndarray:
        r"""
        Combined third-order tensor for CFVaR3 computation.

        This tensor corresponds to the paper's :math:`\mathcal{T}` with entries

        .. math::

            \tau_{i,j,k}
            = \frac{3\nu^2}{(\nu-2)(\nu-4)}
              (D_{(:,i)} + \Gamma^{[i]}\mu)^\top
              (\Sigma\Gamma^{[k]}\Sigma)
              (D_{(:,j)} + \Gamma^{[j]}\mu)
            + \frac{\nu^3}{(\nu-2)(\nu-4)(\nu-6)}
              \operatorname{tr}\!\left[
                  \Gamma^{[i]}\Sigma\Gamma^{[j]}\Sigma\Gamma^{[k]}\Sigma
              \right].

        Returns
        -------
        np.ndarray
            3-D array of shape (M, M, M).

        Raises
        ------
        ValueError
            If degrees_of_freedom <= 6 (third moment does not exist).
        """
        nu = self.degrees_of_freedom
        if nu <= 6:
            raise ValueError(f"T_tensor requires degrees_of_freedom > 6; got {nu}.")

        coeff_q = (3 * nu**2) / ((nu - 2) * (nu - 4))
        coeff_t = nu**3 / ((nu - 2) * (nu - 4) * (nu - 6))

        T = np.zeros((self.M, self.M, self.M))

        for i in range(self.M):
            for j in range(self.M):
                for k in range(self.M):
                    Gamma_i = self.__get_gamma_k(i)
                    Gamma_j = self.__get_gamma_k(j)
                    Gamma_k = self.__get_gamma_k(k)

                    D_i = self.D[:, i].reshape(-1, 1)
                    D_j = self.D[:, j].reshape(-1, 1)

                    # Delta-gamma cross term q_{ijk}
                    q_ijk = np.trace(
                        (D_i + Gamma_i @ self.mu).T
                        @ (self.scale_matrix @ Gamma_k @ self.scale_matrix)
                        @ (D_j + Gamma_j @ self.mu)
                    )

                    # Pure gamma trace term t_{ijk}
                    t_ijk = np.trace(
                        Gamma_i
                        @ self.scale_matrix
                        @ Gamma_j
                        @ self.scale_matrix
                        @ Gamma_k
                        @ self.scale_matrix
                    )

                    T[i, j, k] = coeff_q * q_ijk + coeff_t * t_ijk

        return T

    @cached_property
    def A_script(self) -> float:
        r"""
        Intermediate variable in optimal number of shares under CFVaR2 minimization.

        Returns
        -------
        float
            $\mathscr{A} = \tfrac{1}{2} G_{[\bullet,1]}^\top Q\, G_{[\bullet,1]}$

        Notes
        -----
        Uses the unified variance matrix Q (= U when \omega = 0).
        """
        G = self.G
        Q = self.Q
        A_script = 0.5 * G[:, 0].T @ Q @ G[:, 0]
        return A_script.item()  # type: ignore

    @cached_property
    def B_script(self) -> float:
        r"""
        Intermediate variable in optimal number of shares under CFVaR2 minimization.

        Returns
        -------
        float
            $\mathscr{B} = G_{[\bullet,2]}^\top Q\, G_{[\bullet,1]}$

        Notes
        -----
        Uses the unified variance matrix Q (= U when \omega = 0).
        """
        G = self.G
        Q = self.Q
        B_script = G[:, 1].T @ Q @ G[:, 0]
        return B_script.item()  # type: ignore

    @cached_property
    def C_script(self) -> float:
        r"""
        Intermediate variable in optimal number of shares under CFVaR2 minimization.

        Returns
        -------
        float
            $\mathscr{C} = \tfrac{1}{2} G_{[\bullet,2]}^\top Q\, G_{[\bullet,2]}$

        Notes
        -----
        Uses the unified variance matrix Q (= U when \omega = 0).
        """
        G = self.G
        Q = self.Q
        C_script = 0.5 * G[:, 1].T @ Q @ G[:, 1]
        return C_script.item()  # type: ignore

    def A_cal(self, alpha: float) -> np.ndarray:
        """
        Coefficient for epsilon^2 in the quadratic equation for CFVaR2 optimization.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.

        Returns
        -------
        np.ndarray
            The coefficient A in the quadratic equation.

        Formula
        -------
        $\\mathcal{A} \\coloneq  4\\mathscr{A}^2 (\\normal^{-1}(\alpha))^2 - 4\\mathscr{A}$
        """
        A_script = self.A_script

        A_cal = 4 * (A_script**2) * norm.ppf(alpha) ** 2 - 4 * A_script
        return A_cal

    def B_cal(self, alpha: float) -> np.ndarray:
        """
        Coefficient for epsilon in the quadratic equation for CFVaR2 optimization.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.

        Returns
        -------
        np.ndarray
            The coefficient B in the quadratic equation.

        Formula
        -------
        $\\mathcal{B} \\coloneq  4\\mathscr{A}\\mathscr{B} (\\normal^{-1}(\\alpha))^2 - 4\\mathscr{B}$
        """
        A_script = self.A_script

        B_script = self.B_script
        B_cal = 4 * A_script * B_script * norm.ppf(alpha) ** 2 - 4 * B_script
        return B_cal

    def C_cal(self, alpha: float) -> np.ndarray:
        """
        Constant in the quadratic equation for CFVaR2 optimization.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFVaR2 calculation.

        Returns
        -------
        np.ndarray
            The constant term C in the quadratic equation.

        Formula
        -------
        $\\mathcal{C} \\coloneq \\mathscr{B}^2 (\\normal^{-1}(\\alpha))^2 - 4\\mathscr{C}$
        """
        B_script = self.B_script

        C_script = self.C_script
        C_cal = B_script**2 * norm.ppf(alpha) ** 2 - 4 * C_script
        return C_cal

    def A_cal_ES(self, alpha: float) -> np.ndarray:
        """
        Coefficient for epsilon^2 in the quadratic equation for CFES2 optimization.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFES2 calculation.

        Returns
        -------
        np.ndarray
            The coefficient $\\hat{\\mathcal{A}}$ in the quadratic equation.

        Formula
        -------
        $\\hat{\\mathcal{A}} \\coloneq  4\\mathscr{A}^2
        \\left(\\dfrac{\\varphi(\\normal^{-1}(\\alpha))}{\\alpha}\\right)^2 - 4\\mathscr{A}$
        """
        A_script = self.A_script
        z_alpha = norm.ppf(alpha)
        phi_over_alpha = norm.pdf(z_alpha) / alpha

        A_cal_ES = 4 * (A_script**2) * phi_over_alpha**2 - 4 * A_script
        return A_cal_ES

    def B_cal_ES(self, alpha: float) -> np.ndarray:
        """
        Coefficient for epsilon in the quadratic equation for CFES2 optimization.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFES2 calculation.

        Returns
        -------
        np.ndarray
            The coefficient $\\hat{\\mathcal{B}}$ in the quadratic equation.

        Formula
        -------
        $\\hat{\\mathcal{B}} \\coloneq  4\\mathscr{A}\\mathscr{B}
        \\left(\\dfrac{\\varphi(\\normal^{-1}(\\alpha))}{\\alpha}\\right)^2 - 4\\mathscr{B}$
        """
        A_script = self.A_script
        B_script = self.B_script
        z_alpha = norm.ppf(alpha)
        phi_over_alpha = norm.pdf(z_alpha) / alpha

        B_cal_ES = 4 * A_script * B_script * phi_over_alpha**2 - 4 * B_script
        return B_cal_ES

    def C_cal_ES(self, alpha: float) -> np.ndarray:
        """
        Constant in the quadratic equation for CFES2 optimization.

        Parameters
        ----------
        alpha : float
            The tail risk for the CFES2 calculation.

        Returns
        -------
        np.ndarray
            The constant term $\\hat{\\mathcal{C}}$ in the quadratic equation.

        Formula
        -------
        $\\hat{\\mathcal{C}} \\coloneq \\mathscr{B}^2
        \\left(\\dfrac{\\varphi(\\normal^{-1}(\\alpha))}{\\alpha}\\right)^2 - 4\\mathscr{C}$
        """
        B_script = self.B_script
        C_script = self.C_script
        z_alpha = norm.ppf(alpha)
        phi_over_alpha = norm.pdf(z_alpha) / alpha

        C_cal_ES = B_script**2 * phi_over_alpha**2 - 4 * C_script
        return C_cal_ES
