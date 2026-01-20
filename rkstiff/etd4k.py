r"""
Constant-Step Fourth-Order ETD Integrator with Krylov Subspace Methods
========================================================================

Implements a fourth-order exponential time-differencing (ETD4) solver
using Krylov subspace methods (KIOPS) for computing phi-function actions
on large sparse or matrix-free operators.

This solver is designed for systems where the linear operator cannot be
efficiently diagonalized or stored as a dense matrix, such as:

- Large sparse matrices from finite difference/element discretizations
- Matrix-free operators where only A*v products are available
- High-dimensional problems where O(n^2) storage is prohibitive

The governing equation is:

.. math::

    \frac{\partial \mathbf{U}}{\partial t}
      = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U})

References
----------
Krogstad, S. (2005).
*Generalized integrating factor methods for stiff PDEs.*
Journal of Computational Physics, 203(1), 72-88.

Niesen, J., & Wright, W. M. (2012).
*Algorithm 919: A Krylov subspace algorithm for evaluating the
phi-functions appearing in exponential integrators.*
ACM Trans. Math. Softw., 38(3).
"""

import logging
from typing import Callable, Union, Literal
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

from .etd import ETDCS, ETDConfig
from .krylov import KrylovConfig, kiops, phi_functions


class _ETD4K:  # pylint: disable=too-few-public-methods
    r"""
    ETD4 Krogstad solver using Krylov subspace methods.

    Computes the four-stage ETD4 Runge-Kutta update using KIOPS for
    phi-function evaluations, enabling application to large sparse or
    matrix-free operators.

    The scheme uses the Krogstad (2005) coefficients with stages:

    .. math::

        k_2 &= e^{hL/2}u + \tfrac{h}{2}\varphi_1(hL/2) N_1 \\
        k_3 &= e^{hL/2}u + \tfrac{h}{2}(\varphi_1(hL/2) - \varphi_2(hL/2)) N_1
             + \tfrac{h}{2}\varphi_2(hL/2) N_2 \\
        k_4 &= e^{hL}u + h(\varphi_1(hL) - \varphi_2(hL)) N_1
             + h\varphi_2(hL) N_3 \\
        u_{n+1} &= e^{hL}u + h b_1 N_1 + h b_2 (N_2 + N_3) + h b_4 N_4

    where :math:`b_1, b_2, b_4` are specific combinations of
    :math:`\varphi_1, \varphi_2, \varphi_3`.

    Parameters
    ----------
    lin_op : array, LinearOperator, or callable
        Linear operator L (sparse matrix, LinearOperator, or matvec function).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function N(U).
    krylov_config : KrylovConfig
        Configuration for Krylov subspace methods.
    """

    def __init__(
        self,
        lin_op: Union[np.ndarray, LinearOperator, Callable[[np.ndarray], np.ndarray]],
        nl_func: Callable[[np.ndarray], np.ndarray],
        krylov_config: KrylovConfig,
    ) -> None:
        self.lin_op = lin_op
        self.nl_func = nl_func
        self.krylov_config = krylov_config

        # Determine problem size
        if isinstance(lin_op, np.ndarray):
            self._n = lin_op.shape[0]
        elif issparse(lin_op) or isinstance(lin_op, LinearOperator):
            self._n = lin_op.shape[0]
        else:
            # For callable, we'll determine size from first call
            self._n = None

        # Pre-allocate storage for stage values
        self._NL1 = None
        self._NL2 = None
        self._NL3 = None
        self._NL4 = None
        self._k = None

        # Cache step size for coefficient reuse
        self._h_cached = None

    def _apply_phi_linear_combination(
        self,
        h: float,
        u: np.ndarray,
        nl_vectors: list,
        coeffs: list,
        h_scale: float = 1.0,
    ) -> np.ndarray:
        r"""
        Compute a linear combination of phi-functions using KIOPS.

        Computes:

        .. math::

            e^{h \cdot h\_scale \cdot L} u
            + h \sum_k c_k \varphi_k(h \cdot h\_scale \cdot L) b_k

        Parameters
        ----------
        h : float
            Base time step.
        u : np.ndarray
            Current solution vector.
        nl_vectors : list of np.ndarray
            Vectors to apply phi-functions to.
        coeffs : list of float
            Coefficients for each phi-function term.
        h_scale : float
            Scaling factor for h (e.g., 0.5 for half-step).

        Returns
        -------
        np.ndarray
            Result of the linear combination.
        """
        tau = h * h_scale

        # Build the B matrix for KIOPS
        # Column 0: u (for phi_0 = exp)
        # Column k: scaled nonlinear vector (for phi_k)
        num_terms = len(nl_vectors) + 1
        n = len(u)

        B = np.zeros((n, num_terms), dtype=u.dtype)
        B[:, 0] = u

        for k, (vec, coeff) in enumerate(zip(nl_vectors, coeffs)):
            if vec is not None and coeff != 0:
                B[:, k + 1] = coeff * vec

        # Use KIOPS to compute the linear combination
        w, _ = kiops(tau, self.lin_op, B, self.krylov_config)

        return w

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize the first nonlinear evaluation.

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
        Update cached step size (no precomputation for Krylov).

        Parameters
        ----------
        h : float
            Time step size.
        """
        self._h_cached = h

    def update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        r"""
        Advance solution by one time step using ETD4 Krogstad scheme.

        The ETD4 scheme with Krylov computes each stage by calling KIOPS
        to evaluate the necessary phi-function linear combinations.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        h : float
            Time step size.

        Returns
        -------
        np.ndarray
            Updated solution vector.

        Notes
        -----
        The Krogstad coefficients are:

        - Stage 2: :math:`e^{hL/2}u + \frac{h}{2}\varphi_1(hL/2) N_1`
        - Stage 3: :math:`e^{hL/2}u + h a_{31} N_1 + h a_{32} N_2`
        - Stage 4: :math:`e^{hL}u + h a_{41} N_1 + h a_{43} N_3`
        - Final: :math:`e^{hL}u + h b_1 N_1 + h b_2 (N_2+N_3) + h b_4 N_4`

        where the coefficients are combinations of psi-functions.
        """
        n = len(u)

        # Stage 2: k2 = exp(hL/2)*u + (h/2)*psi1(hL/2)*NL1
        # a21 = (h/2) * psi1(hL/2) = (h/2) * (exp(hL/2) - 1)/(hL/2)
        # This is phi_1 scaled: (h/2) * phi_1(hL/2) where phi_1 = psi_1
        tau_half = h / 2
        B2 = np.column_stack([u, self._NL1])
        self._k, _ = kiops(tau_half, self.lin_op, B2, self.krylov_config)
        self._NL2 = self.nl_func(self._k)

        # Stage 3: k3 = exp(hL/2)*u + a31*NL1 + a32*NL2
        # a31 = (h/2)(psi1 - psi2/2) at hL/2
        # a32 = (h/2)(psi2/2) at hL/2
        # In terms of phi: psi_k = phi_k, and we need:
        # exp(hL/2)*u + (h/2)*(phi_1 - phi_2)*NL1 + (h/2)*phi_2*NL2
        # = exp(z/2)*u + (h/2)*phi_1(z/2)*(NL1 - NL2) + (h/2)*phi_1(z/2)*NL2 + correction
        # Actually simpler: use KIOPS directly with the coefficients
        # B = [u, c1*NL1 + c2*NL2] where we combine appropriately

        # For stage 3, we need:
        # exp(hL/2)*u + h*(a31/h)*NL1 + h*(a32/h)*NL2
        # Build combined vector for phi_1 and phi_2 terms
        # Using psi functions: a31 = (h/2)*(psi1 - psi2) = (h/2)*phi_1 - (h/2)*phi_2
        # a32 = (h/2)*phi_2
        # Total: exp*u + [(h/2)*phi_1 - (h/2)*phi_2]*NL1 + (h/2)*phi_2*NL2
        #      = exp*u + (h/2)*phi_1*NL1 + (h/2)*phi_2*(NL2 - NL1)
        vec_phi1 = self._NL1
        vec_phi2 = self._NL2 - self._NL1
        B3 = np.column_stack([u, vec_phi1, vec_phi2])
        self._k, _ = kiops(tau_half, self.lin_op, B3, self.krylov_config)
        self._NL3 = self.nl_func(self._k)

        # Stage 4: k4 = exp(hL)*u + a41*NL1 + a43*NL3
        # a41 = h*(psi1 - psi2) = h*phi_1 - h*phi_2 at hL
        # a43 = h*psi2 = h*phi_2 at hL
        # Total: exp*u + h*phi_1*NL1 + h*phi_2*(NL3 - NL1)
        vec_phi1 = self._NL1
        vec_phi2 = self._NL3 - self._NL1
        B4 = np.column_stack([u, vec_phi1, vec_phi2])
        self._k, _ = kiops(h, self.lin_op, B4, self.krylov_config)
        self._NL4 = self.nl_func(self._k)

        # Final step: u_new = exp(hL)*u + b1*NL1 + b2*(NL2+NL3) + b4*NL4
        # b1 = h*(psi1 - (3/2)*psi2 + (2/3)*psi3)
        # b2 = h*(psi2 - (2/3)*psi3)
        # b4 = h*(-(1/2)*psi2 + (2/3)*psi3)
        #
        # Rewrite in terms of phi_1, phi_2, phi_3:
        # b1 = h*phi_1 - (3/2)*h*phi_2 + (2/3)*h*phi_3
        # b2 = h*phi_2 - (2/3)*h*phi_3
        # b4 = -(1/2)*h*phi_2 + (2/3)*h*phi_3
        #
        # Collect coefficients:
        # phi_1: NL1 (coefficient h)
        # phi_2: -(3/2)*NL1 + (NL2+NL3) - (1/2)*NL4 (coefficient h)
        # phi_3: (2/3)*NL1 - (2/3)*(NL2+NL3) + (2/3)*NL4 (coefficient h)
        #      = (2/3)*(NL1 - NL2 - NL3 + NL4)
        vec_phi1 = self._NL1
        vec_phi2 = -1.5 * self._NL1 + (self._NL2 + self._NL3) - 0.5 * self._NL4
        vec_phi3 = (2.0 / 3.0) * (self._NL1 - self._NL2 - self._NL3 + self._NL4)

        B_final = np.column_stack([u, vec_phi1, vec_phi2, vec_phi3])
        self._k, _ = kiops(h, self.lin_op, B_final, self.krylov_config)

        # FSAL: store NL1 for next step
        self._NL1 = self.nl_func(self._k)

        return self._k


class ETD4K(ETDCS):
    r"""
    Fourth-order ETD solver using Krylov subspace methods.

    Integrates stiff PDEs of the form

    .. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U})

    using the ETD4 Krogstad scheme with KIOPS for phi-function evaluation.

    This solver is designed for problems where:

    - The linear operator is large and sparse
    - Only matrix-vector products A*v are available (matrix-free)
    - The operator cannot be efficiently diagonalized

    For diagonal operators or small dense matrices, use the standard
    :class:`ETD4` solver instead, which has lower overhead.

    Parameters
    ----------
    lin_op : array, LinearOperator, or callable
        Linear operator L in dU/dt = L*U + N(U). Can be:
        - A sparse matrix (scipy.sparse)
        - A LinearOperator
        - A callable that computes A*v
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function N(U).
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
    >>> def N(u): return -u**2
    >>> solver = ETD4K(L, N)
    >>> u_final = solver.evolve(u0, t0=0, tf=1.0, h=0.01)

    Using matrix-free with a callable:

    >>> def matvec(v):
    ...     # Custom operator application
    ...     return some_operation(v)
    >>> solver = ETD4K(matvec, N)

    See Also
    --------
    ETD4 : Standard ETD4 for diagonal/dense operators
    KrylovConfig : Configuration options for Krylov methods
    """

    def __init__(
        self,
        lin_op: Union[np.ndarray, LinearOperator, Callable[[np.ndarray], np.ndarray]],
        nl_func: Callable[[np.ndarray], np.ndarray],
        krylov_config: KrylovConfig = KrylovConfig(),
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        # Initialize base class (we override most functionality)
        # Note: ETDCS expects a numpy array for lin_op, but we support more types
        # We pass a dummy array if lin_op is not an array
        if isinstance(lin_op, np.ndarray):
            super().__init__(lin_op, nl_func, etd_config=etd_config, loglevel=loglevel)
        else:
            # Create dummy array for base class initialization
            # Actual operator is stored separately
            if isinstance(lin_op, LinearOperator):
                n = lin_op.shape[0]
            else:
                # For callable, we need to determine size later
                n = 1
            dummy_lin_op = np.zeros(n)
            super().__init__(dummy_lin_op, nl_func, etd_config=etd_config, loglevel=loglevel)

        self.krylov_config = krylov_config
        self._method = _ETD4K(lin_op, nl_func, krylov_config)
        self.__n1_init = False

    def _reset(self) -> None:
        """Reset solver to initial state."""
        self.__n1_init = False
        self._h_coeff = None

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
        self.logger.debug("ETD4K step size updated to h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Advance solution by one step using ETD4 Krylov scheme.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        h : float
            Time step size.

        Returns
        -------
        np.ndarray
            Updated solution vector.
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, h)
