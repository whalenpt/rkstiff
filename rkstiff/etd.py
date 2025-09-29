"""rkstiff.etd
Provides classes and functions for Exponential Time Differencing Runge-Kutta methods
"""

import numpy as np
from rkstiff.solver import StiffSolverAS, StiffSolverCS, SolverConfig


class ETDConfig:
    """
    Configuration parameters for the ETD-based solvers.

    Attributes
    ----------
    modecutoff : float
        Numerical inaccuracy cutoff for psi functions (default 0.01).
        Must be between 0.0 and 1.0.
    contour_points : int
        Number of points for the contour integral (default 32).
        Must be an integer greater than 1.
    contour_radius : float
        Radius of the circular contour integral (default 1.0).
        Must be greater than 0.
    """

    def __init__(self, modecutoff=0.01, contour_points=32, contour_radius=1.0):
        """Initialize ETDConfig with validated parameters."""
        self._modecutoff = None
        self._contour_points = None
        self._contour_radius = None

        # Use setters for validation
        self.modecutoff = modecutoff
        self.contour_points = contour_points
        self.contour_radius = contour_radius

    @property
    def modecutoff(self):
        """Get the modecutoff value."""
        return self._modecutoff

    @modecutoff.setter
    def modecutoff(self, value):
        """Set the modecutoff value with validation."""
        if (value > 1.0) or (value <= 0):
            raise ValueError(f"modecutoff must be between 0.0 and 1.0 but is {value}")
        self._modecutoff = value

    @property
    def contour_points(self):
        """Get the contour_points value."""
        return self._contour_points

    @contour_points.setter
    def contour_points(self, value):
        """Set the contour_points value with validation."""
        if not isinstance(value, int):
            raise TypeError(f"contour_points must be an integer but is {value}")
        if value <= 1:
            raise ValueError(f"contour_points must be an integer greater than 1 but is {value}")
        self._contour_points = value

    @property
    def contour_radius(self):
        """Get the contour_radius value."""
        return self._contour_radius

    @contour_radius.setter
    def contour_radius(self, value):
        """Set the contour_radius value with validation."""
        if value <= 0:
            raise ValueError(f"contour_radius must greater than 0 but is {value}")
        self._contour_radius = value


def phi1(z):
    """compute_s RKETD psi-function of the first order.
    INPUTS
        z - real or complex-valued input array
    OUTPUT
        return (exp(z)-1)/z  -> real or complex-valued RKETD function of first order
    """
    return (np.exp(z) - 1) / z


def phi2(z):
    """compute_s RKETD psi-function of the second order.
    INPUTS
        z - real or complex-valued input array
    OUTPUT
        return 2!*(exp(z)-1-z/2)/z^2  -> real or complex-valued RKETD function of second order
    """
    return 2 * (np.exp(z) - 1 - z) / z**2


def phi3(z):
    """compute_s RKETD psi-function of the third order.
    INPUTS
        z - real or complex-valued input array
    OUTPUT
        return 3!*(exp(z)-1-z/2-z^3/6)/z^3  -> real or complex-valued RKETD function of third order
    """
    return 6 * (np.exp(z) - 1 - z - z**2 / 2) / z**3


class ETDAS(StiffSolverAS):
    """
    Class template for an RKETD adaptive-step solver. Adds several ETD specific parameters
    to the StiffSolverAS class.

    ATTRIBUTES
    ----------

    modecutoff : float
       psi RKETD functions are numerically inaccurate near zero when computed directly; when |z| < modecutoff use a
       contour integral (more computationally expensive method) to compute these values
    contour_points : int
        number of points to use in contour integral when evaluating small input values to the psi RKETD functions
    contour_radius : float
        radius size of a circular contour integral to be used to evaluate small input values to the psi RKETD functions

    """

    def __init__(self, lin_op, nl_func, config=SolverConfig(), etd_config=ETDConfig()):
        super().__init__(lin_op, nl_func, config)
        self.etd_config = etd_config
        self._h_coeff = None


class ETDCS(StiffSolverCS):
    """
    Class template for an RKETD constant-step solver. Adds several ETD specific parameters
    to the StiffSolverCS class.

    ATTRIBUTES
    ----------

    modecutoff : float
       psi RKETD functions are numerically inaccurate near zero when computed directly; when |z| < modecutoff use a
       contour integral (more computationally expensive method) to compute these values
    contour_points : int
        number of points to use in contour integral when evaluating small input values to the psi RKETD functions
    contour_radius : float
        radius size of a circular contour integral to be used to evaluate small input values to the psi RKETD functions

    """

    def __init__(self, lin_op, nl_func, etd_config=ETDConfig()):
        super().__init__(lin_op, nl_func)
        self.etd_config = etd_config
        self._h_coeff = None
