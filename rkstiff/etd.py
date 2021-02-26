
import numpy as np
from rkstiff.solver import StiffSolverAS

def phi1(z):
    """ Computes RKETD psi-function of the first order. 
    INPUTS
        z - real or complex-valued input array 
    OUTPUT
        return (exp(z)-1)/z  -> real or complex-valued RKETD function of first order
    """
    return (np.exp(z)-1)/z

def phi2(z):
    """ Computes RKETD psi-function of the second order. 
    INPUTS
        z - real or complex-valued input array 
    OUTPUT
        return 2!*(exp(z)-1-z/2)/z^2  -> real or complex-valued RKETD function of second order
    """
    return 2*(np.exp(z)-1-z)/z**2

def phi3(z):
    """ Computes RKETD psi-function of the third order. 
    INPUTS
        z - real or complex-valued input array 
    OUTPUT
        return 3!*(exp(z)-1-z/2-z^3/6)/z^3  -> real or complex-valued RKETD function of third order
    """
    return 6*(np.exp(z)-1-z-z**2/2)/z**3

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

    def __init__(self,linop,NLfunc,**kwargs):
        super().__init__(linop,NLfunc,**kwargs)
        self.modecutoff = 0.01
        if 'modecutoff' in kwargs:
            self.modecutoff = kwargs['modecutoff']
            if (self.modecutoff > 1) or (self.modecutoff <= 0):
                raise ValueError('modecutoff must be between 0 and 1 but is {}'.format(self.modecutoff))
        self.contour_points = 32
        if 'contour_points' in kwargs:
            self.contour_points = kwargs['contour_points']
            if not isinstance(self.contour_points,int):
                raise TypeError('contour_points must be an integer but is {}'.format(self.contour_points))
            if self.contour_points <= 1:
                raise ValueError('contour_points must be an integer greater than 1 but is {}'.format(self.contour_points))
        self.contour_radius = 1.0
        if 'contour_radius' in kwargs:
            self.contour_radius = kwargs['contour_radius']
            if self.contour_radius <= 0:
                raise ValueError('contour_radius must greater than 0 but is {}'.format(self.contour_radius))
        self._h_coeff = None
        






