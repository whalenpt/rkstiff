"""Integrating factor adaptive step solver of 5th order with 4rd order embedding."""

from typing import Callable, Literal, Union
import numpy as np
from .solveras import SolverConfig, BaseSolverAS


class IF45DP(BaseSolverAS):
    """
    Fifth-order Integrating Factor solver with adaptive stepping (Dormand-Prince).

    Implements the IF(4,5) scheme based on the Dormand-Prince Runge-Kutta method
    with an embedded fourth-order method for error estimation. Designed for
    diagonal stiff systems of the form dU/dt = L*U + NL(U).

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator L (must be 1D array for diagonal systems).
    nl_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear function nl_func(U).
    config : SolverConfig, optional
        Solver configuration for adaptive stepping parameters.
    loglevel : str or int, optional
        Logging level.

    Attributes
    ----------
    t : np.ndarray
        Time values from most recent call to evolve().
    u : np.ndarray
        Solution array from most recent call to evolve().
    logs : list
        Log messages recording solver operations.

    Raises
    ------
    ValueError
        If lin_op is not a 1D array (non-diagonal system).

    Notes
    -----
    This solver only supports diagonal linear operators. For non-diagonal systems,
    use IF34, ETD34, or ETD35 instead.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        """
        Initialize the IF45DP adaptive solver.

        Parameters
        ----------
        lin_op : np.ndarray
            Diagonal linear operator (1D array).
        nl_func : Callable[[np.ndarray], np.ndarray]
            Nonlinear function.
        config : SolverConfig, optional
            Solver configuration.
        loglevel : str or int, optional
            Logging level.
        """
        super().__init__(lin_op, nl_func, config=config, loglevel=loglevel)
        if len(lin_op.shape) > 1:
            raise ValueError("IF45DP only handles 1D linear operators (diagonal systems): try IF34,ETD34, or ETD35")
        self._EL15, self._EL310, self._EL45, self._EL89, self._EL = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        self._NL1, self._NL2, self._NL3, self._NL4, self._NL5, self._NL6, self._NL7 = [
            np.zeros(self.lin_op.shape[0], dtype=np.complex128) for _ in range(7)
        ]
        (
            self._a21,
            self._a31,
            self._a32,
            self._a41,
            self._a42,
            self._a43,
            self._a51,
            self._a52,
            self._a53,
            self._a54,
        ) = [np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(10)]
        self._a61, self._a62, self._a63, self._a64, self._a65 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(5)
        ]
        self._a71, self._a73, self._a74, self._a75 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(4)
        ]
        self._a76 = 0.0
        self._r1, self._r3, self._r4, self._r5 = [
            np.zeros(shape=self.lin_op.shape, dtype=np.complex128) for _ in range(4)
        ]
        self._r6, self._r7 = 0.0, 0.0
        self._k = np.zeros(self.lin_op.shape[0], dtype=np.complex128)
        self._err = np.zeros(self.lin_op.shape[0], dtype=np.complex128)
        self._h_coeff = None
        self.__n1_init = False

    def _reset(self) -> None:
        """
        Reset solver to its initial state.
        """
        self._h_coeff = None
        self.__n1_init = False

    def _update_stages(self, u: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute next state and error estimate using the 7-stage Dormand-Prince scheme.

        Parameters
        ----------
        u : np.ndarray
            Current solution vector.
        h : float
            Time step size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated solution vector and error estimate.

        Notes
        -----
        Uses the "First Same As Last" (FSAL) principle where the last stage
        evaluation becomes the first evaluation of the next step.
        """
        self._update_coeffs(h)
        if not self.__n1_init:
            self._NL1 = self.nl_func(u)
            self.__n1_init = True
        elif self._accept:
            self._NL1 = self._NL7.copy()

        self._k = self._EL15 * u + self._a21 * self._NL1
        self._NL2 = self.nl_func(self._k)
        self._k = self._EL310 * u + self._a31 * self._NL1 + self._a32 * self._NL2
        self._NL3 = self.nl_func(self._k)
        self._k = self._EL45 * u + self._a41 * self._NL1 + self._a42 * self._NL2 + self._a43 * self._NL3
        self._NL4 = self.nl_func(self._k)
        self._k = (
            self._EL89 * u
            + self._a51 * self._NL1
            + self._a52 * self._NL2
            + self._a53 * self._NL3
            + self._a54 * self._NL4
        )
        self._NL5 = self.nl_func(self._k)
        self._k = (
            self._EL * u
            + self._a61 * self._NL1
            + self._a62 * self._NL2
            + self._a63 * self._NL3
            + self._a64 * self._NL4
            + self._a65 * self._NL5
        )
        self._NL6 = self.nl_func(self._k)
        self._k = (
            self._EL * u
            + self._a71 * self._NL1
            + self._a73 * self._NL3
            + self._a74 * self._NL4
            + self._a75 * self._NL5
            + self._a76 * self._NL6
        )
        self._NL7 = self.nl_func(self._k)
        self._err = (
            self._r1 * self._NL1
            + self._r3 * self._NL3
            + self._r4 * self._NL4
            + self._r5 * self._NL5
            + self._r6 * self._NL6
            + self._r7 * self._NL7
        )

        return self._k, self._err

    def _update_coeffs(self, h: float) -> None:
        """
        Update IF45DP coefficients based on step size h.

        Computes all exponential coefficients and Runge-Kutta weights for the
        Dormand-Prince integrating factor method. Coefficients are cached and
        only recomputed when the step size changes.

        Parameters
        ----------
        h : float
            Time step size.

        Notes
        -----
        The method uses 7 stages with specific fractional exponentials
        (1/5, 3/10, 4/5, 8/9, etc.) based on the Dormand-Prince tableau.
        """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        z = h * self.lin_op
        self._EL15 = np.exp(z / 5)
        self._EL310 = np.exp(3 * z / 10)
        self._EL45 = np.exp(4 * z / 5)
        self._EL89 = np.exp(8 * z / 9)
        self._EL = np.exp(z)
        EL710 = np.exp(7 * z / 10)
        EL19 = np.exp(z / 9)
        self._a21 = h * self._EL15 / 5.0
        self._a31 = 3 * h * self._EL310 / 40.0
        self._a32 = 9 * h * np.exp(z / 10) / 40.0
        self._a41 = 44 * h * self._EL45 / 45.0
        self._a42 = -56 * h * np.exp(3 * z / 5) / 15.0
        self._a43 = 32 * h * np.exp(z / 2) / 9.0
        self._a51 = 19372 * h * self._EL89 / 6561.0
        self._a52 = -25360 * h * np.exp(31 * z / 45) / 2187.0
        self._a53 = 64448.0 * h * np.exp(53 * z / 90) / 6561.0
        self._a54 = -212 * h * np.exp(4 * z / 45) / 729.0
        self._a61 = 9017 * h * self._EL / 3168.0
        self._a62 = -355 * h * self._EL45 / 33.0
        self._a63 = 46732 * h * EL710 / 5247.0
        self._a64 = 49 * h * self._EL15 / 176.0
        self._a65 = -5103 * h * EL19 / 18656.0
        self._a71 = 35 * h * self._EL / 384.0
        self._a73 = 500 * h * EL710 / 1113.0
        self._a75 = -2187 * h * EL19 / 6784.0
        self._a74 = 125 * h * self._EL15 / 192.0
        self._a76 = 11 * h / 84.0
        self._r1 = h * 71 * self._EL / 57600.0
        self._r3 = -71 * h * EL710 / 16695.0
        self._r4 = 17 * h * self._EL15 / 1920.0
        self._r5 = -17253 * h * EL19 / 339200.0
        self._r6 = 22 * h / 525.0
        self._r7 = -h / 40.0
        self.logger.debug("IF45 coefficients updated for step size h=%s", h)

    def _q(self) -> int:
        """
        Return order for computing suggested step size.

        Returns
        -------
        int
            Method order (5 for this fifth-order method).
        """
        return 5
