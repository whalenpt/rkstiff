"""Exponential Time Differencing Runge-Kutta Methods.

This module provides classes and functions for implementing Exponential Time
Differencing (ETD) Runge-Kutta methods for solving stiff ordinary differential
equations. ETD methods are particularly effective for problems with a linear
stiff component and a nonlinear non-stiff component.

The module includes:
    - Configuration classes for ETD parameters
    - Phi functions (φ₁, φ₂, φ₃) used in ETD schemes
    - Adaptive-step and constant-step ETD solvers

Example
-------
::

    import numpy as np
    from rkstiff.etd import ETDAS, ETDConfig
    from rkstiff.solver import SolverConfig

    # Define linear operator and nonlinear function
    lin_op = np.array([[-1.0, 0.0], [0.0, -2.0]])
    nl_func = lambda u: np.sin(u)

    # Create solver with custom configuration
    etd_config = ETDConfig(modecutoff=0.01, contour_points=32)
    solver_config = SolverConfig()
    solver = ETDAS(lin_op, nl_func, solver_config, etd_config)
"""

from typing import Union, Literal, Callable
import numpy as np
from rkstiff.solver import StiffSolverAS, StiffSolverCS, SolverConfig


class ETDConfig:
    """Configuration parameters for ETD-based solvers.

    This class manages the numerical parameters used in Exponential Time
    Differencing methods, including cutoff values for the phi functions and
    parameters for contour integral computations.

    Parameters
    ----------
    modecutoff : float, optional
        Numerical inaccuracy cutoff for psi/phi functions. Values of the linear
        operator below this threshold use Taylor series approximations to avoid
        numerical instabilities. Default is 0.01. Must be between 0.0 and 1.0.
    contour_points : int, optional
        Number of points used for contour integral evaluation in computing the
        phi functions. More points provide higher accuracy but increase
        computational cost. Default is 32. Must be greater than 1.
    contour_radius : float, optional
        Radius of the circular contour used for contour integral evaluation.
        Default is 1.0. Must be greater than 0.

    Raises
    ------
    ValueError
        If modecutoff is not between 0.0 and 1.0, if contour_points is not
        greater than 1, or if contour_radius is not greater than 0.
    TypeError
        If contour_points is not an integer.

    Examples
    --------
    Create a configuration with default values:

    >>> config = ETDConfig()
    >>> config.modecutoff
    0.01

    Create a configuration with custom values:

    >>> config = ETDConfig(modecutoff=0.005, contour_points=64, contour_radius=2.0)
    >>> config.contour_points
    64

    Notes
    -----
    The modecutoff parameter is critical for numerical stability. When eigenvalues
    of the linear operator have magnitudes close to zero, direct evaluation of
    phi functions can lead to division by small numbers. The cutoff determines
    when to switch to Taylor series approximations.
    """

    def __init__(self, modecutoff: float = 0.01, contour_points: int = 32, contour_radius: float = 1.0) -> None:
        self._modecutoff: float = 0.01
        self._contour_points: int = 32
        self._contour_radius: float = 1.0
        self.modecutoff = modecutoff
        self.contour_points = contour_points
        self.contour_radius = contour_radius

    @property
    def modecutoff(self) -> float:
        """Numerical inaccuracy cutoff for psi/phi functions.

        Must be between 0.0 and 1.0.
        """
        return self._modecutoff

    @modecutoff.setter
    def modecutoff(self, value: float) -> None:
        if (value > 1.0) or (value <= 0):
            raise ValueError(f"modecutoff must be between 0.0 and 1.0 but is {value}")
        self._modecutoff = value

    @property
    def contour_points(self) -> int:
        """Number of points for the contour integral.

        Must be an integer greater than 1.
        """
        return self._contour_points

    @contour_points.setter
    def contour_points(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"contour_points must be an integer but is {value}")
        if value <= 1:
            raise ValueError(f"contour_points must be an integer greater than 1 but is {value}")
        self._contour_points = value

    @property
    def contour_radius(self) -> float:
        """Radius of the circular contour integral.

        Must be greater than 0.
        """
        return self._contour_radius

    @contour_radius.setter
    def contour_radius(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"contour_radius must greater than 0 but is {value}")
        self._contour_radius = value


def phi1(z: np.ndarray) -> np.ndarray:
    """
    Compute the first-order RKETD psi-function.

    Parameters
    ----------
    z : float, complex, or array-like
        Real or complex input array.

    Returns
    -------
    float, complex, or array-like
        1! * (exp(z) - 1) / z

    Examples
    --------
    >>> import numpy as np
    >>> from rkstiff.etd import phi1
    >>> phi1(np.array([0.1, 0.5]))
    array([0.950166, 1.297442])
    """
    return (np.exp(z) - 1) / z


def phi2(z: np.ndarray) -> np.ndarray:
    """
    Compute the second-order RKETD psi-function.

    Parameters
    ----------
    z : float, complex, or array-like
        Real or complex input array.

    Returns
    -------
    float, complex, or array-like
        2! * (exp(z) - 1 - z) / z**2
    """
    return 2 * (np.exp(z) - 1 - z) / z**2


def phi3(z: np.ndarray) -> np.ndarray:
    """
    Compute the third-order RKETD psi-function.

    Parameters
    ----------
    z : float, complex, or array-like
        Real or complex input array.

    Returns
    -------
    float, complex, or array-like
        3! * (exp(z) - 1 - z - z**2/2) / z**3
    """
    return 6 * (np.exp(z) - 1 - z - z**2 / 2) / z**3


class ETDAS(StiffSolverAS):
    """Adaptive-step Exponential Time Differencing solver.

    This class implements an adaptive time-stepping ETD Runge-Kutta method for
    solving stiff ordinary differential equations of the form:

        du/dt = L*u + N(u)

    where L is a linear operator and N(u) is a nonlinear function. The adaptive
    step size is controlled based on error estimates to balance accuracy and
    computational efficiency.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator matrix L. Should be a square matrix with shape (n, n).
    nl_func : callable
        Nonlinear function N(u). Should accept a numpy array and return a numpy
        array of the same shape. Signature: ``nl_func(u: np.ndarray) -> np.ndarray``
    config : SolverConfig, optional
        General solver configuration including tolerances, step size bounds, and
        other numerical parameters. Default is ``SolverConfig()``.
    etd_config : ETDConfig, optional
        ETD-specific configuration including modecutoff and contour integral
        parameters. Default is ``ETDConfig()``.
    loglevel : str or int, optional
        Logging level for solver output. Can be "DEBUG", "INFO", "WARNING",
        "ERROR", "CRITICAL", or an integer logging level. Default is "WARNING".

    Attributes
    ----------
    etd_config : ETDConfig
        The ETD-specific configuration parameters.

    See Also
    --------
    ETDCS : Constant-step ETD solver
    ETDConfig : Configuration for ETD methods
    StiffSolverAS : Base class for adaptive-step solvers

    Examples
    --------
    Solve a simple stiff ODE:

    >>> import numpy as np
    >>> from rkstiff.etd import ETDAS, ETDConfig
    >>> from rkstiff.solver import SolverConfig
    >>>
    >>> # Linear operator (stiff)
    >>> lin_op = np.array([[-100.0, 0.0], [0.0, -0.1]])
    >>>
    >>> # Nonlinear function (non-stiff)
    >>> def nl_func(u):
    ...     return np.array([0.1 * u[1], -0.1 * u[0]])
    >>>
    >>> # Configure solver
    >>> solver_config = SolverConfig()
    >>> etd_config = ETDConfig(modecutoff=0.01)
    >>> solver = ETDAS(lin_op, nl_func, solver_config, etd_config)
    >>>
    >>> # Solve from t=0 to t=1 with initial condition u0
    >>> u0 = np.array([1.0, 0.0])
    >>> t_span = (0.0, 1.0)
    >>> result = solver.solve(u0, t_span)

    Notes
    -----
    The ETD methods are particularly effective when the linear component is stiff
    (has large negative eigenvalues) while the nonlinear component varies on
    slower time scales. The adaptive stepping ensures efficiency while maintaining
    accuracy.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        config: SolverConfig = SolverConfig(),
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        super().__init__(lin_op, nl_func, config, loglevel=loglevel)
        self.etd_config: ETDConfig = etd_config
        self._h_coeff = None


class ETDCS(StiffSolverCS):
    """Constant-step Exponential Time Differencing solver.

    This class implements a constant time-stepping ETD Runge-Kutta method for
    solving stiff ordinary differential equations of the form:

        du/dt = L*u + N(u)

    where L is a linear operator and N(u) is a nonlinear function. Unlike the
    adaptive version, this solver uses a fixed step size throughout the
    integration, which can be more efficient when the solution behavior is
    well-understood.

    Parameters
    ----------
    lin_op : np.ndarray
        Linear operator matrix L. Should be a square matrix with shape (n, n).
    nl_func : callable
        Nonlinear function N(u). Should accept a numpy array and return a numpy
        array of the same shape. Signature: ``nl_func(u: np.ndarray) -> np.ndarray``
    etd_config : ETDConfig, optional
        ETD-specific configuration including modecutoff and contour integral
        parameters. Default is ``ETDConfig()``.
    loglevel : str or int, optional
        Logging level for solver output. Can be "DEBUG", "INFO", "WARNING",
        "ERROR", "CRITICAL", or an integer logging level. Default is "WARNING".

    Attributes
    ----------
    etd_config : ETDConfig
        The ETD-specific configuration parameters.

    See Also
    --------
    ETDAS : Adaptive-step ETD solver
    ETDConfig : Configuration for ETD methods
    StiffSolverCS : Base class for constant-step solvers

    Examples
    --------
    Solve with a fixed time step:

    >>> import numpy as np
    >>> from rkstiff.etd import ETDCS, ETDConfig
    >>>
    >>> # Linear operator
    >>> lin_op = np.array([[-100.0, 0.0], [0.0, -0.1]])
    >>>
    >>> # Nonlinear function
    >>> def nl_func(u):
    ...     return np.array([0.1 * u[1], -0.1 * u[0]])
    >>>
    >>> # Configure solver
    >>> etd_config = ETDConfig(modecutoff=0.01)
    >>> solver = ETDCS(lin_op, nl_func, etd_config)
    >>>
    >>> # Solve with fixed step size
    >>> u0 = np.array([1.0, 0.0])
    >>> t_span = (0.0, 1.0)
    >>> dt = 0.01
    >>> result = solver.solve(u0, t_span, dt)

    Notes
    -----
    Constant-step solvers are appropriate when:

    - The step size required for stability is known a priori
    - The solution varies smoothly without rapid transients
    - Maximum computational efficiency is desired
    - Output is needed at regular time intervals

    For problems with unknown smoothness or transient behavior, consider using
    the adaptive-step solver :class:`ETDAS` instead.
    """

    def __init__(
        self,
        lin_op: np.ndarray,
        nl_func: Callable[[np.ndarray], np.ndarray],
        etd_config: ETDConfig = ETDConfig(),
        loglevel: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "WARNING",
    ) -> None:
        super().__init__(lin_op, nl_func, loglevel=loglevel)
        self.etd_config: ETDConfig = etd_config
        self._h_coeff = None
