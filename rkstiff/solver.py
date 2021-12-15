
import numpy as np
from typing import Tuple,Optional


class StiffSolverAS:
    """
    Base class for an adaptive-step Runge-Kutta solver for stiff systems of the type dtU = LU + NL(U),
    where L is a linear operator and NL is a non-linear function.

    ATTRIBUTES
    __________

    linop : np.array
        Linear operator (L) in the system dtU = LU + NL(U). Can be either a 2D numpy array (matrix)
        or a 1D array (diagonal system). L can be either real-valued or complex-valued.

    NLfunc : function 
        Nonlinear function (NL(U)) in the system dtU = LU + NL(U). Can be a complex or real-valued function.

    u : list
        List of np.arrays corresponding to the propagated U in the system dtU = LU + NL(U) using the evolve function

    t : list
        List of times corresponding to the propagated U in the system dtU = LU + NL(U) using the evolve function

    logs : list
        List of log messages

    incrF : float, > 1.0
        Increment factor for increasing the step size utilized in propagating a system. After each step in the propagation
        a 'optimal' step size of h_opt is computed and compared vs the current step size h_current. If h_opt > incrF*h_current,
        then the solver will suggest h_opt as the next step size, otherwise it will continue using the current step size. This
        parameter is used such that very small changes to the step size are avoided and hence potentially expensive evaluations
        of the RK coefficients are avoided. 

    decrF : float, < 1.0
        Decrement factor for decreasing the step size utilized in propagating a system. After each step in the propagation
        a 'optimal' step size of h_opt is computed and compared vs the current step size h_current. If h_opt > decrF*h_current and
        h_opt < h_current, then the solver will suggest h_new = decrF*h_current as the next step size. This
        parameter is used such that very small changes to the step size are avoided and hence potentially expensive evaluations
        of the RK coefficients are avoided. 

    epsilon : float
        Relative error tolerance for system. Solver will suggest step sizes in an attempt to keep the two-norm relative error of the system
        less than this value. This is used as a tuning parameter in the solver and the relative-error is not strictly
        enforced! In general, smaller epsilon results in smaller relative-error but due to the nature of the solvers utilized, the
        adaptive stepping is prone to be less than ideal and errors can often be larger than the specified epsilon tolerance level.

    safetyF : float
        Safety factor for adaptive stepping. The 'optimal' step size computed using the embedded RK methods is multiplied by
        this factor to try to enforce the relative-error to be below the epsilon relative-error threshold. The relative-error
        computed for these solvers based on embedded methods is an inaccurate estimate and so this saftey factor can be used
        to tune the system error to be more inline with actual errors.

    adapt_cutoff : float < 1
        Limits values used in the computation of the suggested step size to those with |u| > adapt_cutoff*max(|u|). To include all
        values in the calculation of step size: set this to a very small number. 

    minh : float
        Minimum step size that can be taken by the solver before throwing an exception


    METHODS
    _______

    step(u,h_suggest):
        Propagates a given array of u one step using an RK method for stiff PDEs. 

    evolve(u,t0,tf,h_init=None,store_data=True,store_freq=1)
        Propagates an initial value (array) of u given at time t0
        until a final time tf is reached using a RK method for stiff PDEs 

    reset()
        Resets solver including erasing stored variables such as self.t, self.u, and self.logs

    """
    MAX_LOOPS = 50
    MAX_S = 4 # maximum step increase factor
    MIN_S = 0.25 # minimum step decrease factor
    
    def __init__(self,linop,NLfunc,epsilon = 1e-4,incrF = 1.25, decrF = 0.85,\
            safetyF = 0.8, adapt_cutoff = 0.01, minh = 1e-16):
        """

        linop : np.array
            Linear operator (L) in the system dtU = LU + NL(U). Can be either a 2D numpy array (matrix)
            or a 1D array (diagonal system). L can be either real-valued or complex-valued.

        NLfunc : function 
            Nonlinear function (NL(U)) in the system dtU = LU + NL(U). Can be a complex or real-valued function.

        epsilon : float
            Relative error tolerance for system. Solver will suggest step sizes in an attempt to keep the two-norm relative error of the system
            less than this value. This is used as a tuning parameter in the solver and the relative-error is not strictly
            enforced! In general, smaller epsilon results in smaller relative-error but due to the nature of the solvers utilized, the
            adaptive stepping is prone to be less than ideal and errors can often be larger than the specified epsilon tolerance level.

        incrF : float, > 1.0
            Increment factor for increasing the step size utilized in propagating a system. After each step in the propagation
            a 'optimal' step size of h_opt is computed and compared vs the current step size h_current. If h_opt > incrF*h_current,
            then the solver will suggest h_opt as the next step size, otherwise it will continue using the current step size. This
            parameter is used such that very small changes to the step size are avoided and hence potentially expensive evaluations
            of the RK coefficients are avoided. 

        decrF : float, < 1.0
            Decrement factor for decreasing the step size utilized in propagating a system. After each step in the propagation
            a 'optimal' step size of h_opt is computed and compared vs the current step size h_current. If h_opt > decrF*h_current and
            h_opt < h_current, then the solver will suggest h_new = decrF*h_current as the next step size. This
            parameter is used such that very small changes to the step size are avoided and hence potentially expensive evaluations
            of the RK coefficients are avoided. 

        safetyF : float
            Safety factor for adaptive stepping. The 'optimal' step size computed using the embedded RK methods is multiplied by
            this factor to try to enforce the relative-error to be below the epsilon relative-error threshold. The relative-error
            computed for these solvers based on embedded methods is an inaccurate estimate and so this saftey factor can be used
            to tune the system error to be more inline with actual errors.

        adapt_cutoff : float < 1
            Limits values used in the computation of the suggested step size to those with |u| > adapt_cutoff*max(|u|). To include all
            values in the calculation of step size: set this to a very small number. 

        minh : float
            Minimum step size that can be taken by the solver before throwing an exception
            self.linop = linop
        """

        self.linop = linop
        self.NLfunc = NLfunc
        self.t, self.u = [], []

        dims = linop.shape
        self._diag = True
        if len(dims) > 2:
            raise ValueError('linop must be a 1D or 2D array')
        elif len(dims) == 2:
            if (dims[0] != dims[1]):
                raise ValueError('linop must be a square matrix')
            self._diag = False

        self.epsilon = epsilon
        if self.epsilon <= 0:
            raise ValueError('epsilon must be positive but is {}'.format(self.epsilon))

        self.incrF = incrF
        if self.incrF <= 1.0:
            raise ValueError('incrF must be > 1.0 but is {}'.format(self.incrF))
            
        self.decrF = decrF
        if self.decrF >= 1.0:
            raise ValueError('decrF must be < 1.0 but is {}'.format(self.decrF))
            
        self.safetyF = safetyF
        if self.safetyF > 1.0:
            raise ValueError('safetyF must be <= 1.0 but is {}'.format(self.safetyF))

        self.adapt_cutoff = adapt_cutoff
        if self.adapt_cutoff >= 1.0:
            raise ValueError('adapt_cutoff must be < 1.0 but is {}'.format(self.adapt_cutoff))

        self.minh = minh
        if self.minh <= 0:
            raise ValueError('minh must be positive but is {}'.format(self.minh))

        self.__t0, self.__tf, self.__tc = 0,0,0
        self.__h_prev = 0.0
        self._accept = False 
        self.t, self.u, self.logs = [], [], []

    def reset(self):
        """ Resets solver to its initial state (such that its ready to call the functions evolve 
            or step on a new input). Erases stored variables such as self.t, self.u, and self.logs.
        """
        self.t, self.u, self.logs = [], [], []
        self.__t0, self.__tf, self.__tc = 0,0,0
        self.__h_prev = 0.0
        self._accept = False 
        self._reset()

    # reset solver dependent variables of the subclass
    def _reset():
        raise NotImplementedError
    
    # update RK stages, returns u_{n+1} given u_{n}, to be overwritten by subclass
    def _updateStages(self,u,h):
        raise NotImplementedError
        
    # q value in computation of suggested step size, to be overwritten by subclass
    def _q(self):
        raise NotImplementedError
        
    def step(self,u : np.ndarray,h_suggest : float) -> Tuple[np.ndarray,float,float]:
        """ 
        Propagates a given array of u one step using an RK method for stiff PDEs. 
        
        INPUTS:
            u : np.array
                input to propagate
            h_suggest : float
                suggested step size; may be reduced to achieve the desired accuracy as
                specified by the relative error threshold variable 'epsilon'

        OUTPUTS:
            unew : np.array
                result of input u propagated one step in the RK algorithm
            h : float
                actual step size taken in the algorithm (may be less than h_suggest)
            h_suggest : float
                step size the algorithm suggests you take for the next step
        """
        
        h = h_suggest
        assert h >= 0.0
        numloops = 0
        while(True):
            unew,err = self._updateStages(u,h)
            self.__h_prev = h 
            # Compute step size change factor s
            s = self._computeS(unew,err) 
            # If s is less than 1, inf, or nan, reject step and reduce step size
            if np.isinf(s) or np.isnan(s) or s < 1.0:      
                h = self._rejectStepSize(s,h)
            # If s is bigger than 1 accept h and the step
            else:
                h_suggest = self._acceptStepSize(s,h) 
                return unew, h, h_suggest 
            numloops += 1
            if numloops > self.MAX_LOOPS:
                failure_str = """Solver failed: adaptive step made too many attempts to find a step
                                 size with an acceptible amount of error. """
                self.logs.append(failure_str)
                raise Exception(failure_str)
            if h < self.minh:
                failure_str = """Solver failed: adaptive step reached minimum step size """
                self.logs.append(failure_str)
                raise Exception(failure_str)
                
        raise Exception("Propagation Failed")
        
    # helper function for computing coefficient s used in generating suggested step size
    def _computeS(self,u,err):
        # Use adapt_cutoff to ignore small modes/values in the computation of the step size
        magu = np.abs(u)
        idx = magu/magu.max() > self.adapt_cutoff
        tol = self.epsilon*np.linalg.norm(u[idx])
        s = self.safetyF*np.power(tol/np.linalg.norm(err[idx]),1.0/self._q())
        return s
    
    # helper function computing suggested step size if step is rejected
    def _rejectStepSize(self,s,h):
        self._accept = False
        # Check that s is a number
        if np.isinf(s) or np.isnan(s):
            self.logs.append('inf or nan number encountered: reducing step size to {}!'.format(h))
            return self.MIN_S*h
        
        s = np.max([s,self.MIN_S]) # dont let s be too small
        s = np.min([s,self.decrF]) # dont let s be too close to 1
        self.logs.append('step rejected with s = {:.2f}'.format(s))
        hnew = s*h
        self.logs.append('reducing step size to {}'.format(hnew))
        return hnew
    
    # helper function computing suggested step size if step is accepted
    def _acceptStepSize(self,s,h):
        self._accept = True
        s = np.min([s,self.MAX_S]) # dont let s be too big
        self.logs.append('step accepted with s = {:.2f}'.format(s))
        # if s much larger than 1, increase the step size
        if s > self.incrF:
            h_suggest = s*h
            self.logs.append('increasing step size to {}'.format(h_suggest))
            return h_suggest
        return h


    def evolve(self,u : np.ndarray,t0 : float,tf : float,\
            h_init : Optional[float]=None,\
            store_data : bool=True, store_freq : int=1) -> np.ndarray:
        """ 
        This function propagates an initial value (array) of u given at time t0
        until a final time tf is reached using a RK method for stiff PDEs 
        
        INPUTS:
            u : np.array
                initial value input 
            t0 : float
                initial time at which u is evaluated
            tf : float
                end time at which propagation stops
            h_init : float, optional
                starting step size for RK method (default of (tf-t0)/100 if not specified)
            store_data : bool, optional 
                value that determines whether to keep track of the propagation array u
                at each step of the RK method. Values stored in self.u and self.t
            store_freq : int, optional 
                store propagation data in self.t and self.u after every [store_freq] step is taken
        OUTPUTS:
            u : np.array 
                final value of the input propagated from t0 to tf 
        """
        
        self.reset()
        self.__t0, self.__tf, self.__tc = t0, tf, t0
                   
        if store_data:
            self.t.append(t0)
            self.u.append(u)
        
        # Set initial step size if none given
        if h_init is None:
            h_init = (self.__tf-self.__t0)/100.    
        h = h_init
        
        # Make sure step size isn't larger than entire propagation time
        if self.__tc+h > self.__tf:
            h = self.__tf - self.__tc
        
        step_count = 0
        while self.__tc < self.__tf:
            u,h,h_suggest = self.step(u,h)
            self.__tc += h
            step_count += 1
            if self.__tc+h_suggest > self.__tf:
                h = self.__tf - self.__tc
            else:
                h = h_suggest
            
            if store_data and (step_count % store_freq == 0):
                self.t.append(self.__tc)
                self.u.append(u)
        
        return u




