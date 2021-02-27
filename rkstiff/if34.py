
from rkstiff.solver import StiffSolverAS
import numpy as np
from scipy.linalg import expm

class _IF34_Diagonal:
    """
    IF34 diagonal system strategy for IF34 solver 
    """
    def __init__(self,linop,NLfunc):
        self.linop = linop
        self.NLfunc = NLfunc
        
        N = linop.shape[0]
        self._EL, self._EL2 = [np.zeros(N,dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3,self._NL4, self._NL5 = [np.zeros(N,dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(N,dtype=np.complex128) 
        self._err = np.zeros(N,dtype=np.complex128)
        
    def _updateCoeffs(self,h):
        z = h*self.linop
        self._updateCoeffsDiagonal(h,z)
        
    def _updateCoeffsDiagonal(self,h,z):   
        self._EL = np.exp(z)
        self._EL2 = np.exp(z/2)
    
    def _N1_init(self,u):
        self._NL1 = self.NLfunc(u)
        
    def _updateStages(self,u,h,accept):        
        if accept:
            self._NL1 = self._NL5.copy()
        
        self._k = self._EL2*u + h*self._EL2*self._NL1/2.0
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL2*u + h*self._NL2/2.
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL*u + h*self._EL2*self._NL3
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL*u + h*(self._EL*self._NL1/6.0 + self._EL2*self._NL2/3.0 \
                + self._EL2*self._NL3/3.0 + self._NL4/6.0)
        self._NL5 = self.NLfunc(self._k)
        self._err = h*(self._NL4-self._NL5)/6.
        return self._k, self._err

class _IF34_Diagonalized(_IF34_Diagonal):
    """
    IF34 non-diagonal system with eigenvector diagonalization strategy for IF34 solver 
    """
    def __init__(self,linop,NLfunc):
        super().__init__(linop,NLfunc)
        if len(linop.shape) == 1:
            raise Exception('Cannot diagonalize a 1D system')
        linop_cond = np.linalg.cond(linop)
        if linop_cond > 1e16:
            raise Exception('Cannot diagonalize a non-invertible linear operator L')
        if linop_cond > 1000:
            print('''Warning: linear matrix array has a large condition number of {:.2f}, 
            method may be unstable'''.format(linop_cond))
        self._eig_vals, self._S = np.linalg.eig(linop)
        self._Sinv = np.linalg.inv(self._S)
        self._v = np.zeros(linop.shape[0])
        
    def _updateCoeffs(self,h):
        z = h*self._eig_vals
        self._updateCoeffsDiagonal(h,z)        
        
    def _N1_init(self,u):
        self._NL1 = self._Sinv.dot(self.NLfunc(u))
        self._v = self._Sinv.dot(u)
    
    def _updateStages(self,u,h,accept):        
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()    
            self._v = self._Sinv.dot(u)

        self._k = self._EL2*self._v + h*self._EL2*self._NL1/2.0
        self._NL2 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._k = self._EL2*self._v + h*self._NL2/2.
        self._NL3 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._k = self._EL*self._v + h*self._EL2*self._NL3
        self._NL4 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._k = self._EL*self._v + h*(self._EL*self._NL1/6.0 + self._EL2*self._NL2/3.0 \
                + self._EL2*self._NL3/3.0 + self._NL4/6.0)
        self._NL5 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._err = h*(self._NL4-self._NL5)/6.
        return self._S.dot(self._k), self._err

class _IF34_NonDiagonal:
    """
    IF34 non-diagonal system strategy for IF34 solver 
    """
    def __init__(self,linop,NLfunc):
        self.linop = linop
        self.NLfunc = NLfunc
        
        N = linop.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(2)]
        self._NL1, self._NL2, self._NL3,self._NL4, self._NL5 = [np.zeros(N,dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(N,dtype=np.complex128) 
        self._err = np.zeros(N,dtype=np.complex128)
        
    def _updateCoeffs(self,h):   
        z = h*self.linop
        self._EL = expm(z)
        self._EL2 = expm(z/2)
        
    def _N1_init(self,u):
        self._NL1 = self.NLfunc(u)
        
    def _updateStages(self,u,h,accept):        
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()    

        self._k = self._EL2.dot(u) + h*self._EL2.dot(self._NL1/2.0)
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL2.dot(u) + h*self._NL2/2.
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL.dot(u) + h*self._EL2.dot(self._NL3)
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL.dot(u) + h*(self._EL.dot(self._NL1/6.0) + self._EL2.dot(self._NL2/3.0) \
                + self._EL2.dot(self._NL3/3.0) + self._NL4/6.0)
        self._NL5 = self.NLfunc(self._k)
        self._err = h*(self._NL4-self._NL5)/6.
        return self._k, self._err

class IF34(StiffSolverAS):
    """
    Integrating factor adaptive step solver of 4th order with 3rd order embedding

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

    def __init__(self,linop,NLfunc,**kwargs):
        """
        INPUTS
        ______

        linop : np.array
            Linear operator (L) in the system dtU = LU + NL(U). Can be either a 2D numpy array (matrix)
            or a 1D array (diagonal system). L can be either real-valued or complex-valued.

        NLfunc : function 
            Nonlinear function (NL(U)) in the system dtU = LU + NL(U). Can be a complex or real-valued function.

        diagonalize : bool, optional
            Diagonalize the linear operator (matrix) and solve the diagonalized system 

        StiffSolverAS variables: epsilon, incrF, decrF, safetyF, adapt_cutoff, minh 
                        (see StiffSolverAS documentation from solver module)
        """

        super().__init__(linop,NLfunc,**kwargs)
        self._method = None
        if self._diag:
            self._method = _IF34_Diagonal(linop,NLfunc)
        else:
            if 'diagonalize' in kwargs and kwargs['diagonalize']:
                self._method = _IF34_Diagonalized(linop,NLfunc)
            else:
                self._method = _IF34_NonDiagonal(linop,NLfunc)
        self._reset()

    def _reset(self):
        #Resets solver to its initial state 
        self.__N1_init = False
        self._h_coeff = None
        self._accept = False
        
    def _updateCoeffs(self,h):
        # Update coefficients if step size h changed
        if h == self._h_coeff:
            return
        self._h_coeff = h
        self._method._updateCoeffs(h)
        self.logs.append("IF34 coefficients updated")
        
    def _updateStages(self,u,h):
        # Computes u_{n+1} from u_{n} through one RK passthrough
        self._updateCoeffs(h)
        if not self.__N1_init:
            self._method._N1_init(u)
            self.__N1_init = True
        return self._method._updateStages(u,h,self._accept)
    
    def _q(self):
        # Order variable for computing suggested step size (embedded order + 1)
        return 4        



