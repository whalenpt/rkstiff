
import numpy as np
from rkstiff.solver import StiffSolverAS
from scipy.linalg import expm

class IF45DP(StiffSolverAS):
    """
    Integrating factor adaptive step solver of 5th order with 4rd order embedding.
    Based on underlying Dormand-Prince method.

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
        if len(linop.shape) > 1:
            raise Exception('IF45DP only handles 1D linear operators (diagonal systems): try IF34,ETD34, or ETD35')
        self._EL15, self._EL310, self._EL45, self._EL89,self._EL = \
                [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(5)]
        self._NL1, self._NL2, self._NL3, self._NL4,self._NL5,self._NL6,\
                self._NL7 = [np.zeros(self.linop.shape[0],dtype=np.complex128) for _ in range(7)]
        self._a21, self._a31, self._a32, self._a41, self._a42, self._a43,\
                self._a51, self._a52, self._a53, self._a54 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(10)]
        self._a61, self._a62, self._a63, self._a64, self._a65 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(5)]
        self._a71, self._a73, self._a74, self._a75 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(4)]
        self._a76 = 0.0
        self._r1, self._r3, self._r5, self._r5 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(4)]
        self._r6, self._r7 = 0.0, 0.0
        self._k = np.zeros(self.linop.shape[0],dtype=np.complex128)
        self._err = np.zeros(self.linop.shape[0],dtype=np.complex128)
        self._h_coeff = None
        self.__N1_init = False
        
    def _reset(self):
        self._h_coeff = None
        self.__N1_init = False
        
    def _updateStages(self,u,h):
        self._updateCoeffs(h)
        # First is same as last principle
        if not self.__N1_init:
            self._NL1 = self.NLfunc(u)
            self.__N1_init = True
        elif self._accept:
            self._NL1 = self._NL7.copy()
        
        self._k = self._EL15*u + self._a21*self._NL1
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL310*u + self._a31*self._NL1 + self._a32*self._NL2
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL45*u + self._a41*self._NL1 + self._a42*self._NL2 + self._a43*self._NL3
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL89*u + self._a51*self._NL1 + self._a52*self._NL2 + self._a53*self._NL3 \
                + self._a54*self._NL4
        self._NL5 = self.NLfunc(self._k)
        self._k = self._EL*u + self._a61*self._NL1 + self._a62*self._NL2 + self._a63*self._NL3 \
                + self._a64*self._NL4 + self._a65*self._NL5
        self._NL6 = self.NLfunc(self._k)  
        self._k = self._EL*u + self._a71*self._NL1 + self._a73*self._NL3 + self._a74*self._NL4 \
                + self._a75*self._NL5 + self._a76*self._NL6
        self._NL7 = self.NLfunc(self._k)
        self._err = self._r1*self._NL1 + self._r3*self._NL3 + self._r4*self._NL4 \
                  + self._r5*self._NL5  + self._r6*self._NL6 + self._r7*self._NL7

        return self._k, self._err
            
    def _updateCoeffs(self,h):
        # Update coefficients if step size h changed
        if h == self._h_coeff:
            return
        self._h_coeff = h
        z = h*self.linop
        self._EL15 = np.exp(z/5)
        self._EL310 = np.exp(3*z/10)
        self._EL45 = np.exp(4*z/5)
        self._EL89 = np.exp(8*z/9)
        self._EL = np.exp(z)
        EL710 = np.exp(7*z/10)
        EL19 = np.exp(z/9)
        self._a21 = h*self._EL15/5.0
        self._a31 = 3*h*self._EL310/40.0
        self._a32 = 9*h*np.exp(z/10)/40.0
        self._a41 = 44*h*self._EL45/45.0
        self._a42 = -56*h*np.exp(3*z/5)/15.0
        self._a43 = 32*h*np.exp(z/2)/9.0
        self._a51 = 19372*h*self._EL89/6561.0
        self._a52 = -25360*h*np.exp(31*z/45)/2187.0
        self._a53 = 64448.0*h*np.exp(53*z/90)/6561.0
        self._a54 = -212*h*np.exp(4*z/45)/729.0
        self._a61 = 9017*h*self._EL/3168.0
        self._a62 = -355*h*self._EL45/33.0
        self._a63 = 46732*h*EL710/5247.0
        self._a64 = 49*h*self._EL15/176.0
        self._a65 = -5103*h*EL19/18656.0
        self._a71 = 35*h*self._EL/384.0
        self._a73 = 500*h*EL710/1113.0
        self._a75 = -2187*h*EL19/6784.0
        self._a74 = 125*h*self._EL15/192.0
        self._a76 = 11*h/84.0
        self._r1 = h*71*self._EL/57600.0
        self._r3 = -71*h*EL710/16695.0
        self._r4 = 17*h*self._EL15/1920.0
        self._r5 = -17253*h*EL19/339200.0
        self._r6 = 22*h/525.0
        self._r7 = -h/40.0
           
        self.logs.append("IF45DP coefficients updated")
        
    def _q(self):
        return 5



