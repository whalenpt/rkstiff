
from rkstiff.etd import ETDAS,phi1,phi2,phi3
import numpy as np
from typing import Callable
from scipy.linalg import expm

class _ETD34_Diagonal:
    """
    ETD34 diagonal system strategy for ETD34 solver 
    """

    def __init__(self,linop,NLfunc,contourM,contourR,modecutoff):
        self.linop = linop
        self.NLfunc = NLfunc
        self.M = contourM
        self.R = contourR
        self.modecutoff = modecutoff
        
        N = linop.shape[0]
        self._EL, self._EL2 = [np.zeros(N,dtype=np.complex128) for _ in range(2)]
        self._a21, self._a31, self._a32, self._a41, self._a43, self._a51,\
                self._a52, self._a54 = [np.zeros(N,dtype=np.complex128) for _ in range(8)]   
        self._NL1, self._NL2, self._NL3,self._NL4, self._NL5 = [np.zeros(N,dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(N,dtype=np.complex128)
        self._err = np.zeros(N,dtype=np.complex128)
        
    def _updateCoeffs(self,h):
        z = h*self.linop
        self._updateCoeffsDiagonal(h,z)
        
    def _updateCoeffsDiagonal(self,h,z):   
        self._EL = np.exp(z)
        self._EL2 = np.exp(z/2)
      
        smallmode_idx = np.abs(z) < self.modecutoff
        zb = z[~smallmode_idx] # z big
        # compute big mode coeffs
        phi1_12 = h*phi1(zb/2)
        phi2_12 = h*phi2(zb/2)
        phi1_1 = h*phi1(zb)
        phi2_1 = h*phi2(zb)
        phi3_1 = h*phi3(zb)
        
        self._a21[~smallmode_idx] = 0.5*phi1_12
        self._a31[~smallmode_idx] = 0.5*(phi1_12-phi2_12)
        self._a32[~smallmode_idx] = 0.5*phi2_12
        self._a41[~smallmode_idx] = phi1_1 - phi2_1
        self._a43[~smallmode_idx] = phi2_1
        self._a51[~smallmode_idx] = phi1_1 - (3.0/2)*phi2_1+(2.0/3)*phi3_1
        self._a52[~smallmode_idx] = phi2_1 - (2.0/3)*phi3_1
        self._a54[~smallmode_idx] = -(1.0/2)*phi2_1 + (2.0/3)*phi3_1
        
        # compute small mode coeffs
        zs = z[smallmode_idx] # z small
        r = self.R*np.exp(2j*np.pi*np.arange(0.5,self.M)/self.M)
        rr, zz = np.meshgrid(r,zs)
        Z = zz+rr
        
        phi1_12 = h*np.sum(phi1(Z/2),axis=1)/self.M
        phi2_12 = h*np.sum(phi2(Z/2),axis=1)/self.M
        phi1_1 = h*np.sum(phi1(Z),axis=1)/self.M
        phi2_1 = h*np.sum(phi2(Z),axis=1)/self.M
        phi3_1 = h*np.sum(phi3(Z),axis=1)/self.M
        
        self._a21[smallmode_idx] = 0.5*phi1_12
        self._a31[smallmode_idx] = 0.5*(phi1_12-phi2_12)
        self._a32[smallmode_idx] = 0.5*phi2_12
        self._a41[smallmode_idx] = phi1_1 - phi2_1
        self._a43[smallmode_idx] = phi2_1
        self._a51[smallmode_idx] = phi1_1 - (3.0/2)*phi2_1+(2.0/3)*phi3_1
        self._a52[smallmode_idx] = phi2_1 - (2.0/3)*phi3_1
        self._a54[smallmode_idx] = -(1.0/2)*phi2_1 + (2.0/3)*phi3_1
    
    def _N1_init(self,u):
        self._NL1 = self.NLfunc(u)
        
    def _updateStages(self,u,h,accept):        
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()    
            
        self._k = self._EL2*u + self._a21*self._NL1
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL2*u + self._a31*self._NL1 + self._a32*self._NL2
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL*u + self._a41*self._NL1 + self._a43*self._NL3
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL*u + self._a51*self._NL1 + self._a52*(self._NL2+self._NL3) \
                + self._a54*self._NL4
        self._NL5 = self.NLfunc(self._k)
        self._err = self._a54*(self._NL4-self._NL5)
        return self._k, self._err

class _ETD34_Diagonalized(_ETD34_Diagonal):
    """
    ETD34 non-diagonal system with eigenvector diagonalization strategy for ETD34 solver 
    """
    def __init__(self,linop,NLfunc,contourM,contourR,modecutoff):
        super().__init__(linop,NLfunc,contourM,contourR,modecutoff)
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
        
        self._k = self._EL2*self._v + self._a21*self._NL1
        self._NL2 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._k = self._EL2*self._v + self._a31*self._NL1 + self._a32*self._NL2
        self._NL3 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._k = self._EL*self._v + self._a41*self._NL1 + self._a43*self._NL3
        self._NL4 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._k = self._EL*self._v + self._a51*self._NL1 + self._a52*(self._NL2+self._NL3) \
                + self._a54*self._NL4
        self._NL5 = self._Sinv.dot(self.NLfunc(self._S.dot(self._k)))
        self._err = self._a54*(self._NL4-self._NL5)
        return self._S.dot(self._k), self._err


class _ETD34_NonDiagonal:
    """
    ETD34 non-diagonal system strategy for ETD34 solver 
    """
    def __init__(self,linop,NLfunc,contourM,contourR):
        self.linop = linop
        self.NLfunc = NLfunc
        self.M = contourM
        self.R = contourR
        
        N = linop.shape[0]
        self._EL, self._EL2 = [np.zeros(shape=linop.shape,dtype=np.complex128) for _ in range(2)]
        self._a21, self._a31, self._a32, self._a41, self._a43, self._a51,\
                self._a52, self._a54 = [np.zeros(shape=linop.shape,dtype=np.complex128) for _ in range(8)]   
        self._NL1, self._NL2, self._NL3,self._NL4, self._NL5 = [np.zeros(N,dtype=np.complex128) for _ in range(5)]
        self._k = np.zeros(N,dtype=np.complex128)
        self._err = np.zeros(N,dtype=np.complex128)
        
    def _updateCoeffs(self,h):   
        z = h*self.linop
        self._EL = expm(z)
        self._EL2 = expm(z/2)

        contour_points = self.R*np.exp(2j*np.pi*np.arange(0.5,self.M)/self.M)
        
        phi1_12, phi2_12,phi1_1,phi2_1,phi3_1 = [np.zeros(shape=self.linop.shape,dtype=np.complex128) for _ in range(5)]
        for point in contour_points:
            Q = np.linalg.inv(point*np.eye(*self.linop.shape)-z)
            Q2 = np.linalg.inv(point*np.eye(*self.linop.shape)-z/2)
            phi1_12 += point*phi1(point)*Q2/self.M
            phi2_12 += point*phi2(point)*Q2/self.M
            phi1_1 += point*phi1(point)*Q/self.M
            phi2_1 += point*phi2(point)*Q/self.M
            phi3_1 += point*phi3(point)*Q/self.M  
        
        self._a21 = 0.5*h*phi1_12
        self._a31 = 0.5*h*(phi1_12-phi2_12)
        self._a32 = 0.5*h*phi2_12
        self._a41 = h*(phi1_1 - phi2_1)
        self._a43 = h*phi2_1
        self._a51 = h*(phi1_1 - (3.0/2)*phi2_1+(2.0/3)*phi3_1)
        self._a52 = h*(phi2_1 - (2.0/3)*phi3_1)
        self._a54 = h*(-(1.0/2)*phi2_1 + (2.0/3)*phi3_1)
        
    def _N1_init(self,u):
        self._NL1 = self.NLfunc(u)
        
    def _updateStages(self,u,h,accept):        
        # Use First is same as last principle (FSAL) -> k5 stage is input u for next step
        if accept:
            self._NL1 = self._NL5.copy()    
            
        self._k = self._EL2.dot(u) + self._a21.dot(self._NL1)
        self._NL2 = self.NLfunc(self._k)
        self._k = self._EL2.dot(u) + self._a31.dot(self._NL1) + self._a32.dot(self._NL2)
        self._NL3 = self.NLfunc(self._k)
        self._k = self._EL.dot(u) + self._a41.dot(self._NL1) + self._a43.dot(self._NL3)
        self._NL4 = self.NLfunc(self._k)
        self._k = self._EL.dot(u) + self._a51.dot(self._NL1) + self._a52.dot(self._NL2+self._NL3) \
                + self._a54.dot(self._NL4)
        self._NL5 = self.NLfunc(self._k)
        self._err = self._a54.dot(self._NL4-self._NL5)
        return self._k, self._err

class ETD34(ETDAS):
    """
    Exponential time-differencing adaptive step solver of 4th order with 3rd order embedding

    ATTRIBUTES
    __________

    linop : np.array
    NLfunc : function
    t : time-array stored with evolve function call 
    u : output-array stored with evolve function call 
    logs : array of info stored related to the solver

    ETD Parameters (see ETDAS class in etd module)
    ______________
    modecutoff : float
    contour_points :  int
    contour_radius : float 

    StiffSolverAS Parameters (see StiffSolverAS class in solver module)
    ________________________
    epsilon : float
    incrF : float
    decrF : float 
    safetyF : float
    adapt_cutoff : float
    minh : float 
    """
    
    def __init__(self,linop : np.ndarray,NLfunc : Callable[[np.ndarray],np.ndarray],epsilon : float = 1e-4,incrF : float = 1.25,\
            decrF : float = 0.85, safetyF : float = 0.8, adapt_cutoff : float = 0.01,\
            minh : float = 1e-16, modecutoff : float = 0.01, contour_points : int = 32,
            contour_radius : float = 1.0, diagonalize : bool = False):
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

        ETDAS variables: modecutoff, contour_points, contour_radius (see ETDAS documentation from etd module)

        StiffSolverAS variables: epsilon, incrF, decrF, safetyF, adapt_cutoff, minh 
                        (see StiffSolverAS documentation from solver module)

        """
        super().__init__(linop,NLfunc,epsilon=epsilon,incrF=incrF,decrF=decrF,\
                safetyF=safetyF,adapt_cutoff=adapt_cutoff,minh=minh,\
                modecutoff=modecutoff,contour_points=contour_points,\
                contour_radius=contour_radius)
        self._method = None
        if self._diag:
            self._method = _ETD34_Diagonal(linop,NLfunc,self.contour_points,\
                    self.contour_radius,self.modecutoff)
        else:
            if diagonalize:
                self._method = _ETD34_Diagonalized(linop,NLfunc,self.contour_points,\
                        self.contour_radius,self.modecutoff)
            else:
                self._method = _ETD34_NonDiagonal(linop,NLfunc,self.contour_points,\
                        self.contour_radius)
        self.__N1_init = False
        self._accept = False

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
        self.logs.append("ETD34 coefficients updated")
        
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

        
