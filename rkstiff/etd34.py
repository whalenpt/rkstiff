r"""
Adaptive-Step Fourth Order (Third Order Embedding) Exponential Time-Differencing Integrator
============================================================================================

Implements the **Exponential Time Differencing (ETD)** 3/4 adaptive solver.

This solver integrates stiff systems of the form:

.. math::

        \frac{\partial \mathbf{U}}{\partial t}
        = \mathcal{L}\mathbf{U}
        + \mathcal{N}(\mathbf{U})

using an exponential Runge–Kutta method of order four with embedded
third-order error estimation for adaptive step control.
"""

import logging
from typing import Callable, Union, Literal
import numpy as np
from scipy.linalg import expm
from .etd import ETDAS, ETDConfig, psi1, psi2, psi3, SolverConfig


class _Etd34Diagonal:  # pylint: disable=too-few-public-methods
    """
    ETD34 diagonal system strategy for ETD34 solver.

    Implements the ETD(3,4) method for diagonal linear operators.

    Parameters
    ----------
    lin_op : np.ndarray
        1D array representing the diagonal linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function that maps the solution vector to its nonlinear contribution.
    etd_config : ETDConfig
        Configuration object containing modecutoff, contour_points, contour_radius.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initialize ETD34 diagonal system strategy.

        Parameters
        ----------
        lin_op : np.ndarray
            Diagonal linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration object.
        logger : logging.Logger, optional
            Logger instance for this solver.
        """
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config
        self.logger = logger

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(n, dtype=np.complex128) for _ in range(2)]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a43,
            self._a51,
            self._a52,
            self._a54,
        ) = [np.zeros(n, dtype=np.complex128) for _ in range(8)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """
        Update internal ETD coefficients for step size h.

        Parameters
        ----------
        h : float
            Step size.
        """
        z = h * self.lin_op
        self._update_coeffs_diagonal(h, z)

    def _update_coeffs_diagonal(self, h: float, z: np.ndarray) -> None:
        """
        Compute elementwise ETD coefficients for diagonal systems.

        Parameters
        ----------
        h : float
            Step size.
        z : np.ndarray
            Elementwise scaled linear operator (z = h * L).
        """
        self._EL = np.exp(z)
        self._EL2 = np.exp(z / 2)

        smallmode_idx = np.abs(z) < self.etd_config.modecutoff
        zb = z[~smallmode_idx]  # z big
        # compute big mode coeffs
        psi1_12 = h * psi1(zb / 2)
        psi2_12 = h * psi2(zb / 2)
        psi1_1 = h * psi1(zb)
        psi2_1 = h * psi2(zb)
        psi3_1 = h * psi3(zb)

        self._a21[~smallmode_idx] = 0.5 * psi1_12
        self._a31[~smallmode_idx] = 0.5 * (psi1_12 - psi2_12)
        self._a32[~smallmode_idx] = 0.5 * psi2_12
        self._a41[~smallmode_idx] = psi1_1 - psi2_1
        self._a43[~smallmode_idx] = psi2_1
        self._a51[~smallmode_idx] = psi1_1 - (3.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1
        self._a52[~smallmode_idx] = psi2_1 - (2.0 / 3) * psi3_1
        self._a54[~smallmode_idx] = -(1.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1

        # compute small mode coeffs
        zs = z[smallmode_idx]  # z small
        r = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )
        rr, zz = np.meshgrid(r, zs)
        Z = zz + rr

        psi1_12 = h * np.sum(psi1(Z / 2), axis=1) / self.etd_config.contour_points
        psi2_12 = h * np.sum(psi2(Z / 2), axis=1) / self.etd_config.contour_points
        psi1_1 = h * np.sum(psi1(Z), axis=1) / self.etd_config.contour_points
        psi2_1 = h * np.sum(psi2(Z), axis=1) / self.etd_config.contour_points
        psi3_1 = h * np.sum(psi3(Z), axis=1) / self.etd_config.contour_points

        self._a21[smallmode_idx] = 0.5 * psi1_12
        self._a31[smallmode_idx] = 0.5 * (psi1_12 - psi2_12)
        self._a32[smallmode_idx] = 0.5 * psi2_12
        self._a41[smallmode_idx] = psi1_1 - psi2_1
        self._a43[smallmode_idx] = psi2_1
        self._a51[smallmode_idx] = psi1_1 - (3.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1
        self._a52[smallmode_idx] = psi2_1 - (2.0 / 3) * psi3_1
        self._a54[smallmode_idx] = -(1.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize stored nonlinear evaluation N1 = nl_func(u).

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the RK stage updates and return (k, error_estimate).

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()
        # If not accept, then step failed, reuse previously computed N1
        self._k = self._EL2 * u + self._a21 * self._NL1
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2 * u + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL * u + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = self._EL * u + self._a51 * self._NL1 + self._a52 * (self._NL2 + self._NL3) + self._a54 * self._NL4
        self._NL5 = self.nl_func(self._k)
        self._err = self._a54 * (self._NL4 - self._NL5)
        return self._k, self._err


class _Etd34Diagonalized(_Etd34Diagonal):
    """
    ETD34 non-diagonal system with eigenvector diagonalization strategy.

    Performs eigen-decomposition of L and operates in the diagonal basis.

    Parameters
    ----------
    lin_op : np.ndarray
        Full matrix linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function.
    etd_config : ETDConfig
        ETD configuration object.
    logger : logging.Logger, optional
        Logger instance for this solver.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initialize diagonalized strategy; computes eigen-decomposition of lin_op.

        Parameters
        ----------
        lin_op : np.ndarray
            Full matrix linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration object.
        logger : logging.Logger, optional
            Logger instance for this solver.
        """
        super().__init__(lin_op, nl_func, etd_config, logger)
        if len(lin_op.shape) == 1:
            raise ValueError("cannot diagonalize a 1D system")
        lin_op_cond = np.linalg.cond(lin_op)
        if lin_op_cond > 1e16:
            raise ValueError("cannot diagonalize a non-invertible linear operator L")
        if lin_op_cond > 1000:
            # Provide a friendly, single-line warning.
            self.logger.warning(
                f"Linear matrix array has a large condition number of {lin_op_cond:.2f}, method may be unstable"
            )
        self._eig_vals, self._S = np.linalg.eig(lin_op)
        self._Sinv = np.linalg.inv(self._S)
        self._v = np.zeros(lin_op.shape[0])

    def update_coeffs(self, h: float) -> None:
        """
        Update coefficients for diagonalized eigenvalues.

        Parameters
        ----------
        h : float
            Step size.
        """
        z = h * self._eig_vals
        self._update_coeffs_diagonal(h, z)

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize stored nonlinear evaluation and transformed state v = S^{-1} u.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self._Sinv.dot(self.nl_func(u))
        self._v = self._Sinv.dot(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform stage updates in the diagonalized basis and return (u_next, err).

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()
            self._v = self._Sinv.dot(u)

        self._k = self._EL2 * self._v + self._a21 * self._NL1
        self._NL2 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL2 * self._v + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = self._EL * self._v + self._a41 * self._NL1 + self._a43 * self._NL3
        self._NL4 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._k = (
            self._EL * self._v + self._a51 * self._NL1 + self._a52 * (self._NL2 + self._NL3) + self._a54 * self._NL4
        )
        self._NL5 = self._Sinv.dot(self.nl_func(self._S.dot(self._k)))
        self._err = self._a54 * (self._NL4 - self._NL5)
        return self._S.dot(self._k), self._err


class _Etd34NonDiagonal:
    """
    ETD34 non-diagonal system strategy for ETD34 solver.

    Implements the ETD(3,4) method for full matrix linear operators.

    Parameters
    ----------
    lin_op : np.ndarray
        Full matrix linear operator.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function.
    etd_config : ETDConfig
        ETD configuration object.
    """

    def __init__(self, lin_op: np.ndarray, nl_func: Callable[[np.ndarray], np.ndarray], etd_config: ETDConfig):
        """
        Initialize the non-diagonal strategy.

        Parameters
        ----------
        lin_op : np.ndarray
            Full matrix linear operator.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        etd_config : ETDConfig
            ETD configuration object.
        """
        self.lin_op = lin_op.astype(np.complex128, copy=False)
        self.nl_func = nl_func
        self.etd_config = etd_config

        n = lin_op.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(2)]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a43,
            self._a51,
            self._a52,
            self._a54,
        ) = [np.zeros(shape=lin_op.shape, dtype=np.complex128) for _ in range(8)]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5 = [np.zeros(n, dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(n, dtype=np.complex128)
        self._err = np.zeros(n, dtype=np.complex128)

    def update_coeffs(self, h: float) -> None:
        """
        Update matrix-valued ETD coefficients for step size h.

        Parameters
        ----------
        h : float
            Step size.
        """
        z = h * self.lin_op
        self._EL = expm(z)
        self._EL2 = expm(z / 2)

        contour_points = self.etd_config.contour_radius * np.exp(
            2j * np.pi * np.arange(0.5, self.etd_config.contour_points) / self.etd_config.contour_points
        )

        psi1_12, psi2_12, psi1_1, psi2_1, psi3_1 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        for point in contour_points:
            Q = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z)
            Q2 = np.linalg.inv(point * np.eye(*self.lin_op.shape) - z / 2)
            psi1_12 += point * psi1(point) * Q2 / self.etd_config.contour_points
            psi2_12 += point * psi2(point) * Q2 / self.etd_config.contour_points
            psi1_1 += point * psi1(point) * Q / self.etd_config.contour_points
            psi2_1 += point * psi2(point) * Q / self.etd_config.contour_points
            psi3_1 += point * psi3(point) * Q / self.etd_config.contour_points

        self._a21 = 0.5 * h * psi1_12
        self._a31 = 0.5 * h * (psi1_12 - psi2_12)
        self._a32 = 0.5 * h * psi2_12
        self._a41 = h * (psi1_1 - psi2_1)
        self._a43 = h * psi2_1
        self._a51 = h * (psi1_1 - (3.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1)
        self._a52 = h * (psi2_1 - (2.0 / 3) * psi3_1)
        self._a54 = h * (-(1.0 / 2) * psi2_1 + (2.0 / 3) * psi3_1)

    def n1_init(self, u: np.ndarray) -> None:
        """
        Initialize stored nonlinear evaluation N1 = nl_func(u).

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        """
        self._NL1 = self.nl_func(u)

    def update_stages(self, u: np.ndarray, accept: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform stage updates for the full matrix strategy and return (k, err).

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        accept : bool
            Whether the previous step was accepted (FSAL principle).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()

        self._k = self._EL2.dot(u) + self._a21.dot(self._NL1)
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL2.dot(u) + self._a31.dot(self._NL1) + self._a32.dot(self._NL2)
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL.dot(u) + self._a41.dot(self._NL1) + self._a43.dot(self._NL3)
        self._NL4 = self.nl_func(self._k)
        self._k = (
            self._EL.dot(u) + self._a51.dot(self._NL1) + self._a52.dot(self._NL2 + self._NL3) + self._a54.dot(self._NL4)
        )
        self._NL5 = self.nl_func(self._k)
        self._err = self._a54.dot(self._NL4 - self._NL5)
        return self._k, self._err


class ETD34(ETDAS):
    r"""
    Fourth-order Exponential Time Differencing (ETD) integrator with adaptive stepping.

    Implements the **ETD(3,4)** scheme, a fourth-order exponential integrator that
    automatically adjusts the timestep based on embedded error estimation.
    It wraps the per-strategy implementations (diagonal, diagonalized, non-diagonal)
    and uses the adaptive controller from :class:`rkstiff.etd.ETDAS`.

    The governing equation is assumed to be of the form

    .. math::

            \frac{\partial \mathbf{U}}{\partial t}
            = \mathcal{L}\mathbf{U}
            + \mathcal{N}(\mathbf{U}),

    where :math:`\mathcal{L}` is the linear operator and :math:`\mathcal{N}(\mathbf{U})`
    is the nonlinear function.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator :math:`\mathcal{L}` in the system
        :math:`\dot{\mathbf{U}} = \mathcal{L}\mathbf{U} + \mathcal{N}(\mathbf{U})`.

        Can be one of the following:

        - **2D matrix** – for general full linear operators.
        - **1D array** – for diagonal (elementwise) systems.

        Both real-valued and complex-valued operators are supported.
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function :math:`\mathcal{N}(\mathbf{U})`.
    config : SolverConfig, optional
        General solver configuration controlling adaptivity thresholds,
        safety factors, and other integration parameters.
    etd_config : ETDConfig, optional
        Configuration for ETD-specific parameters such as contour integration
        settings and spectral radius estimation.
    diagonalize : bool, optional
        If ``True``, the solver diagonalizes the linear operator :math:`\mathcal{L}`
        before integration. This can improve performance for some sparse systems.
    loglevel : Union[str, int], optional
        Logging level.

    Notes
    -----
    Configuration parameters for contour integration, adaptivity, and safety
    factors are inherited from :class:`rkstiff.etd.ETDAS` and
    :class:`rkstiff.solver.BaseSolverAS`.

    Specifically:

    - From **ETDAS**:

    - ``modecutoff`` — threshold for switching to contour integration
    - ``contour_points`` — number of contour quadrature points
    - ``contour_radius`` — radius for contour integration in the complex plane

    - From **BaseSolverAS**:

    - ``epsilon``, ``incr_f``, ``decr_f`` — adaptive step control constants
    - ``safety_f`` — safety factor
    - ``adapt_cutoff``, ``minh`` — adaptive and minimum step limits
    """

    _method: Union[_Etd34Diagonal, _Etd34Diagonalized, _Etd34NonDiagonal]

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        etd_config: ETDConfig = ETDConfig(),
        diagonalize: bool = False,
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ):
        """
        Initialize the ETD(3,4) solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Linear operator L in the system dU/dt = L U + N(U). May be
            either a 2D NumPy matrix or a 1D array representing a diagonal system.
            Supports both real and complex-valued operators.
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function N(U) in the system dU/dt = L U + N(U).
        config : SolverConfig, optional
            General solver configuration controlling adaptivity thresholds,
            safety factors, and other integration parameters.
        etd_config : ETDConfig, optional
            Configuration for ETD-specific parameters, such as contour integration
            settings and spectral radius estimation.
        diagonalize : bool, optional
            If True, the solver diagonalizes the linear operator L before integration.
            This can improve performance for certain non-diagonalizable but sparse systems.
        loglevel : Union[str, int], optional
            Logging level.

        Notes
        -----
        The following parameters are inherited from parent classes:
        - From ETDAS: modecutoff, contour_points, contour_radius
        - From BaseSolverAS: epsilon, incr_f, decr_f, safety_f, adapt_cutoff, and minh
        """
        super().__init__(lin_op, nl_func, config=config, etd_config=etd_config, loglevel=loglevel)
        if self._diag:
            self._method = _Etd34Diagonal(lin_op, nl_func, etd_config, self.logger)
        else:
            if diagonalize:
                self._method = _Etd34Diagonalized(lin_op, nl_func, etd_config, self.logger)
            else:
                self._method = _Etd34NonDiagonal(lin_op, nl_func, etd_config)
        self.__n1_init = False
        self._accept = False

    def _reset(self) -> None:
        """
        Reset internal solver state.

        Resets adaptive-step control flags and cached coefficients.
        Called when the solver starts or restarts an integration sequence.
        """
        # Resets solver to its initial state
        self.__n1_init = False
        self._h_coeff = None
        self._accept = False

    def _update_coeffs(self, h: float) -> None:
        """
        Update ETD coefficients for a new time step.

        Parameters
        ----------
        h : float
            Current time step size.

        Notes
        -----
        The coefficient update is skipped if the time step h has not changed
        since the last update.
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method.update_coeffs(h)
        self.logger.debug("ETD34 coefficients updated for step size h=%s", h)

    def _update_stages(self, u: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute u_{n+1} (and an error estimate) from u_n through one RK pass.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        h : float
            Current time step size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution and error estimate.
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._method.n1_init(u)
            self.__n1_init = True
        return self._method.update_stages(u, self._accept)

    def _q(self) -> int:
        """
        Return the solver order used for adaptive step control.

        Returns
        -------
        int
            Effective order (4 for ETD(3,4)), used by the adaptive controller
            for error-based time step adjustments.
        """
        return 4
