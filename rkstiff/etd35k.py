r"""
Adaptive-Step Fifth-Order ETD Integrator with Krylov Subspace Methods
======================================================================

Implements the ETD(3,5) exponential time-differencing scheme with embedded
error estimation using Krylov subspace methods (KIOPS) for computing
phi-function actions on large sparse or matrix-free operators.

The solver combines:

- Fifth-order ETD Runge-Kutta integration
- Third-order embedded error estimation for adaptive stepping
- KIOPS for efficient phi-function evaluation

The governing equation is:

.. math::

    \frac{\partial \mathbf{U}}{\partial t}
      = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U})

References
----------
Whalen, P., Brio, M., & Moloney, J. V. (2015).
*Exponential time-differencing with embedded Runge-Kutta adaptive step control.*
Journal of Computational Physics, 280, 579-601.

Gaudreault, S., Rainwater, G., & Tokman, M. (2018).
*KIOPS: A fast adaptive Krylov subspace solver for exponential integrators.*
Journal of Computational Physics, 372, 236-255.
"""

import logging
from typing import Callable, Union, Literal, Tuple
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

from .etd import ETDAS, ETDConfig, SolverConfig
from .krylov import KrylovConfig, kiops


class _ETD35K:  # pylint: disable=too-few-public-methods
    r"""
    ETD(3,5) solver using Krylov subspace methods.

    Implements the seven-stage ETD(3,5) scheme of Whalen et al. (2015)
    using KIOPS for phi-function evaluations. Provides both the fifth-order
    solution and a third-order embedded error estimate for adaptive stepping.

    Parameters
    ----------
    lin_op : array, LinearOperator, or callable
        Linear operator L (sparse matrix, LinearOperator, or matvec function).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function N(U).
    krylov_config : KrylovConfig
        Configuration for Krylov subspace methods.
    logger : logging.Logger, optional
        Logger instance for diagnostic output.
    """

    def __init__(
        self,
        lin_op: Union[np.ndarray, LinearOperator, Callable[[np.ndarray], np.ndarray]],
        nl_func: Callable[[np.ndarray], np.ndarray],
        krylov_config: KrylovConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.lin_op = lin_op
        self.nl_func = nl_func
        self.krylov_config = krylov_config
        self.logger = logger

        # Determine problem size
        if isinstance(lin_op, np.ndarray):
            self._n = lin_op.shape[0] if lin_op.ndim > 1 else lin_op.shape[0]
        elif issparse(lin_op) or isinstance(lin_op, LinearOperator):
            self._n = lin_op.shape[0]
        else:
            self._n = None

        # Pre-allocate storage for nonlinear evaluations
        self._NL1 = None
        self._NL2 = None
        self._NL3 = None
        self._NL4 = None
        self._NL5 = None
        self._NL6 = None
        self._k = None
        self._err = None

        self._h_cached = None

    def stage_init(self, u: np.ndarray) -> None:
        """
        Initialize the first nonlinear evaluation for a new step.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)
        if self._n is None:
            self._n = len(u)

    def update_coeffs(self, h: float) -> None:
        """
        Update cached step size.

        For Krylov methods, coefficients are computed on-the-fly.

        Parameters
        ----------
        h : float
            Time step size.
        """
        self._h_cached = h

    def update_stages(self, u: np.ndarray, accept: bool) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Perform one ETD(3,5) stage pass using Krylov methods.

        The ETD(3,5) scheme has 7 stages with coefficients from Whalen et al.
        The scheme uses nodes at t + h/4, t + h/4, t + h/2, t + 3h/4, t + h, t + h.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.

        Notes
        -----
        The ETD(3,5) coefficients are complex combinations of psi-functions
        evaluated at different fractions of hL. Using KIOPS, we express each
        stage as a linear combination of phi-function actions.
        """
        h = self._h_cached

        if accept:
            self.stage_init(u)

        # Stage 2: k2 = exp(hL/4)*u + a21*NL1
        # a21 = (h/4)*psi1(hL/4) = (h/4)*phi_1(hL/4)
        tau_14 = h / 4
        B2 = np.column_stack([u, self._NL1])
        self._k, _ = kiops(tau_14, self.lin_op, B2, self.krylov_config)
        self._NL2 = self.nl_func(self._k)

        # Stage 3: k3 = exp(hL/4)*u + a31*NL1 + a32*NL2
        # a31 = (h/4)*(psi1 - psi2/2) = (h/4)*(phi_1 - phi_2/2)
        # a32 = (h/4)*(psi2/2) = (h/8)*phi_2
        # Total: exp*u + (h/4)*phi_1*NL1 + (h/4)*phi_2*(NL2/2 - NL1/2)
        #      = exp*u + (h/4)*phi_1*NL1 + (h/8)*phi_2*(NL2 - NL1)
        vec_phi1_3 = self._NL1
        vec_phi2_3 = 0.5 * (self._NL2 - self._NL1)
        B3 = np.column_stack([u, vec_phi1_3, vec_phi2_3])
        self._k, _ = kiops(tau_14, self.lin_op, B3, self.krylov_config)
        self._NL3 = self.nl_func(self._k)

        # Stage 4: k4 = exp(hL/2)*u + a41*NL1 + a43*NL3
        # a41 = (h/2)*(psi1 - psi2) at hL/2
        # a43 = (h/2)*psi2 at hL/2
        # Total: exp*u + (h/2)*phi_1*NL1 + (h/2)*phi_2*(NL3 - NL1)
        tau_12 = h / 2
        vec_phi1_4 = self._NL1
        vec_phi2_4 = self._NL3 - self._NL1
        B4 = np.column_stack([u, vec_phi1_4, vec_phi2_4])
        self._k, _ = kiops(tau_12, self.lin_op, B4, self.krylov_config)
        self._NL4 = self.nl_func(self._k)

        # Stage 5: k5 = exp(3hL/4)*u + a51*NL1 + a52*(NL2-NL3) + a54*NL4
        # a51 = (3h/4)*(psi1 - (3/4)*psi2) at 3hL/4
        # a52 = -(3h/8)*psi1 at 3hL/4
        # a54 = (9h/16)*psi2 at 3hL/4
        # Express in terms of phi: need to collect phi_1 and phi_2 contributions
        # phi_1 terms: (3h/4)*NL1 - (3h/8)*(NL2-NL3) = (3h/4)*NL1 - (3h/8)*NL2 + (3h/8)*NL3
        # phi_2 terms: -(9h/16)*NL1 + (9h/16)*NL4
        tau_34 = 3 * h / 4
        vec_phi1_5 = self._NL1 - 0.5 * (self._NL2 - self._NL3)
        vec_phi2_5 = -(3.0 / 4.0) * self._NL1 + (3.0 / 4.0) * self._NL4
        B5 = np.column_stack([u, vec_phi1_5, vec_phi2_5])
        self._k, _ = kiops(tau_34, self.lin_op, B5, self.krylov_config)
        self._NL5 = self.nl_func(self._k)

        # Stage 6: k6 = exp(hL)*u + a61*NL1 + a62*(NL2 - 3*NL4/2) + a63*NL3 + a65*NL5
        # a61 = h*(-77*psi1 + 59*psi2)/42
        # a62 = h*8*psi1/7
        # a63 = h*(111*psi1 - 87*psi2)/28
        # a65 = h*(-47*psi1 + 143*psi2)/84
        #
        # Collecting phi_1 terms:
        # (-77/42)*NL1 + (8/7)*(NL2 - 3*NL4/2) + (111/28)*NL3 + (-47/84)*NL5
        # Collecting phi_2 terms:
        # (59/42)*NL1 + (-87/28)*NL3 + (143/84)*NL5
        vec_phi1_6 = (
            (-77.0 / 42.0) * self._NL1
            + (8.0 / 7.0) * (self._NL2 - 1.5 * self._NL4)
            + (111.0 / 28.0) * self._NL3
            + (-47.0 / 84.0) * self._NL5
        )
        vec_phi2_6 = (59.0 / 42.0) * self._NL1 + (-87.0 / 28.0) * self._NL3 + (143.0 / 84.0) * self._NL5
        B6 = np.column_stack([u, vec_phi1_6, vec_phi2_6])
        self._k, _ = kiops(h, self.lin_op, B6, self.krylov_config)
        self._NL6 = self.nl_func(self._k)

        # Stage 7 (final): k7 = exp(hL)*u + a71*NL1 + a73*NL3 + a74*NL4 + a75*NL5 + a76*NL6
        # a71 = h*7*(257*psi1 - 497*psi2 + 270*psi3)/2700
        # a73 = h*(1097*psi1 - 467*psi2 - 150*psi3)/1350
        # a74 = h*2*(-49*psi1 + 199*psi2 - 135*psi3)/225
        # a75 = h*(-313*psi1 + 883*psi2 - 90*psi3)/1350
        # a76 = h*(509*psi1 - 2129*psi2 + 1830*psi3)/2700
        #
        # Collecting phi_1, phi_2, phi_3 terms:
        c1_phi1 = 7.0 * 257.0 / 2700.0
        c3_phi1 = 1097.0 / 1350.0
        c4_phi1 = 2.0 * (-49.0) / 225.0
        c5_phi1 = -313.0 / 1350.0
        c6_phi1 = 509.0 / 2700.0

        c1_phi2 = 7.0 * (-497.0) / 2700.0
        c3_phi2 = -467.0 / 1350.0
        c4_phi2 = 2.0 * 199.0 / 225.0
        c5_phi2 = 883.0 / 1350.0
        c6_phi2 = -2129.0 / 2700.0

        c1_phi3 = 7.0 * 270.0 / 2700.0
        c3_phi3 = -150.0 / 1350.0
        c4_phi3 = 2.0 * (-135.0) / 225.0
        c5_phi3 = -90.0 / 1350.0
        c6_phi3 = 1830.0 / 2700.0

        vec_phi1_7 = (
            c1_phi1 * self._NL1 + c3_phi1 * self._NL3 + c4_phi1 * self._NL4 + c5_phi1 * self._NL5 + c6_phi1 * self._NL6
        )
        vec_phi2_7 = (
            c1_phi2 * self._NL1 + c3_phi2 * self._NL3 + c4_phi2 * self._NL4 + c5_phi2 * self._NL5 + c6_phi2 * self._NL6
        )
        vec_phi3_7 = (
            c1_phi3 * self._NL1 + c3_phi3 * self._NL3 + c4_phi3 * self._NL4 + c5_phi3 * self._NL5 + c6_phi3 * self._NL6
        )

        B7 = np.column_stack([u, vec_phi1_7, vec_phi2_7, vec_phi3_7])
        self._k, _ = kiops(h, self.lin_op, B7, self.krylov_config)

        # Error estimate: err = a75 * (-NL1 + 4*NL3 - 6*NL4 + 4*NL5 - NL6)
        # a75 = h*(-313*psi1 + 883*psi2 - 90*psi3)/1350
        # This represents the difference between 5th and 3rd order solutions
        err_vec = -self._NL1 + 4 * self._NL3 - 6 * self._NL4 + 4 * self._NL5 - self._NL6

        # Check if error vector is too small (pure linear problem case)
        err_norm = np.linalg.norm(err_vec)
        if err_norm < 1e-15:
            # For pure linear problems, use a small error based on solution magnitude
            # This prevents division by zero in adaptive stepping
            self._err = np.full_like(self._k, 1e-15 * (1.0 + np.linalg.norm(self._k)))
        else:
            # Apply a75 scaling using KIOPS
            B_err = np.column_stack(
                [
                    np.zeros_like(u),  # No exp term
                    c5_phi1 * err_vec,
                    c5_phi2 * err_vec,
                    c5_phi3 * err_vec,
                ]
            )
            self._err, _ = kiops(h, self.lin_op, B_err, self.krylov_config)

        return self._k, self._err


class ETD35K(ETDAS):
    r"""
    Adaptive fifth-order ETD solver using Krylov subspace methods.

    Solves stiff systems of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U})

    using the ETD(3,5) scheme with KIOPS for phi-function evaluation and
    embedded third-order error estimation for adaptive time stepping.

    This solver is designed for problems where:

    - The linear operator is large and sparse
    - Only matrix-vector products A*v are available (matrix-free)
    - Adaptive time stepping is needed for efficiency

    For diagonal operators or small dense matrices, use the standard
    :class:`ETD35` solver instead.

    Parameters
    ----------
    lin_op : array, LinearOperator, or callable
        Linear operator L in dU/dt = L*U + N(U). Can be:
        - A sparse matrix (scipy.sparse)
        - A LinearOperator
        - A callable that computes A*v
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function N(U).
    config : SolverConfig, optional
        General solver configuration for adaptivity.
    krylov_config : KrylovConfig, optional
        Configuration for Krylov subspace methods.
    etd_config : ETDConfig, optional
        ETD configuration (not used for Krylov, kept for API compatibility).
    loglevel : str or int, optional
        Logging verbosity level.

    Examples
    --------
    Using with a sparse matrix:

    >>> from scipy.sparse import diags
    >>> L = diags([-1, 2, -1], [-1, 0, 1], shape=(1000, 1000))
    >>> def N(u): return -u**3
    >>> config = SolverConfig(epsilon=1e-6)
    >>> solver = ETD35K(L, N, config=config)
    >>> u_final = solver.evolve(u0, t0=0, tf=10.0)

    Using matrix-free with a callable:

    >>> def matvec(v):
    ...     # Custom operator application
    ...     return some_operation(v)
    >>> solver = ETD35K(matvec, N)

    Notes
    -----
    The adaptive stepping combines the Krylov approximation error with
    the RK truncation error estimate. The step size is adjusted to keep
    the local error below the specified tolerance.

    See Also
    --------
    ETD35 : Standard ETD35 for diagonal/dense operators
    ETD4K : Constant-step ETD4 with Krylov
    KrylovConfig : Configuration options for Krylov methods
    SolverConfig : Configuration for adaptive stepping
    """

    def __init__(
        self,
        lin_op: Union[np.ndarray, LinearOperator, Callable[[np.ndarray], np.ndarray]],
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        krylov_config: KrylovConfig = KrylovConfig(),
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        # Initialize base class
        # Note: ETDAS expects a numpy array for lin_op, but we support more types
        if isinstance(lin_op, np.ndarray):
            super().__init__(lin_op, nl_func, config=config, etd_config=etd_config, loglevel=loglevel)
        else:
            # Create dummy array for base class initialization
            if isinstance(lin_op, LinearOperator):
                n = lin_op.shape[0]
            else:
                n = 1
            dummy_lin_op = np.zeros(n)
            super().__init__(dummy_lin_op, nl_func, config=config, etd_config=etd_config, loglevel=loglevel)

        self.krylov_config = krylov_config
        self._method = _ETD35K(lin_op, nl_func, krylov_config, self.logger)
        self._stages_init = False
        self._accept = False

    def _reset(self) -> None:
        """Reset internal solver state."""
        self._stages_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """
        Update coefficients for new step size.

        For Krylov methods, this just caches the step size.

        Parameters
        ----------
        h : float
            Time step size.
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD35K step size updated to h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute next-step state using one ETD35 Krylov pass.

        Parameters
        ----------
        u : np.ndarray
            Current state vector.
        h : float
            Time step size.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The tuple (u_next, err_estimate) with the next-step value and
            the local truncation error estimate.
        """
        self._update_coeffs(h)
        if not self._stages_init:
            self._method.stage_init(u)
            self._stages_init = True
        return self._method.update_stages(u, self._accept)

    def _q(self) -> int:
        """
        Order variable for computing suggested step size.

        Returns
        -------
        int
            Effective order (4 for ETD(3,5)), used by the adaptive controller.
        """
        return 4
