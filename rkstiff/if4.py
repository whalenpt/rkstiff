
from rkstiff.solver import StiffSolverCS
import numpy as np
from typing import Callable
from scipy.linalg import expm

class _IF4_Diagonal:
    """ IF4 diagonal system strategy for IF4 solver """
    def __init__(self,linop,NLfunc):
        self.linop = linop
        self.NLfunc = NLfunc
        
        N = linop.shape[0]
        self._EL, self._EL2 = [np.zeros(N,dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3,self._NL4 = [np.zeros(N,dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(N,dtype=np.complex128) 
        
    def _updateCoeffs(self,h):
        z = h*self.linop
        self._EL = np.exp(z)
        self._EL2 = np.exp(z/2)
    
    def _N1_init(self,u):
        """ Need to initialize N1 before first updateStage call """
        self._NL1 = self.NLfunc(u)
        
    def _updateStages(self,u,h):        
        """ One RK step """
        self._k = self._EL2*u + h*self._EL2*self._NL1/2.0
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL2*u + h*self._NL2/2.
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL*u + h*self._EL2*self._NL3
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL*u + h*(self._EL*self._NL1/6.0 + self._EL2*self._NL2/3.0 \
                + self._EL2*self._NL3/3.0 + self._NL4/6.0)
        self._NL1 = self.NLfunc(self._k) # FSAL principle
        return self._k

class _IF4_NonDiagonal:
    """ IF4 non-diagonal system strategy for IF4 solver """
    def __init__(self,linop,NLfunc):
        self.linop = linop
        self.NLfunc = NLfunc
        
        N = linop.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3,self._NL4 = [np.zeros(N,dtype=np.complex128) for _ in range(4)]
        self._k = np.zeros(N,dtype=np.complex128) 
        
    def _updateCoeffs(self,h):   
        z = h*self.linop
        self._EL = expm(z)
        self._EL2 = expm(z/2)
        
    def _N1_init(self,u):
        self._NL1 = self.NLfunc(u)
        
    def _updateStages(self,u,h):        
        self._k = self._EL2.dot(u) + h*self._EL2.dot(self._NL1/2.0)
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL2.dot(u) + h*self._NL2/2.
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL.dot(u) + h*self._EL2.dot(self._NL3)
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL.dot(u) + h*(self._EL.dot(self._NL1/6.0) + self._EL2.dot(self._NL2/3.0) \
                + self._EL2.dot(self._NL3/3.0) + self._NL4/6.0)
        self._NL1 = self.NLfunc(self._k) # FSAL principle
        return self._k

class IF4(StiffSolverCS):
    """
    Integrating factor constant step solver of 4th order 

    ATTRIBUTES
    __________
    linop : np.array
    NLfunc : function
    t : time-array stored with evolve function call 
    u : output-array stored with evolve function call 
    logs : array of info stored related to the solver

    StiffSolverAS Parameters (see StiffSolverAS class in solver module)
    ________________________
    epsilon : float
    incrF : float
    decrF : float 
    safetyF : float
    adapt_cutoff : float
    minh : float 
    """

    def __init__(self,linop : np.ndarray,NLfunc : Callable[[np.ndarray],np.ndarray]):
        """
        INPUTS
        ______

        linop : np.array
            Linear operator (L) in the system dtU = LU + NL(U). Can be either a 2D numpy array (matrix)
            or a 1D array (diagonal system). L can be either real-valued or complex-valued.

        NLfunc : function 
            Nonlinear function (NL(U)) in the system dtU = LU + NL(U). Can be a complex or real-valued function.

        """

        super().__init__(linop,NLfunc)
        self._method = None
        if self._diag:
            self._method = _IF4_Diagonal(linop,NLfunc)
        else:
            self._method = _IF4_NonDiagonal(linop,NLfunc)
        self._reset()

    def _reset(self):
        """ Resets solver to its initial state  """
        self.__N1_init = False
        self._h_coeff = None
        
    def _updateCoeffs(self,h):
        """ Update coefficients if step size h changed """
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method._updateCoeffs(h)
        self.logs.append("IF4 coefficients updated")
        
    def _updateStages(self,u,h):
        """ Computes u_{n+1} from u_{n} through one RK passthrough """
        self._updateCoeffs(h)
        if not self.__N1_init:
            self._method._N1_init(u)
            self.__N1_init = True
        return self._method._updateStages(u,h)
    
