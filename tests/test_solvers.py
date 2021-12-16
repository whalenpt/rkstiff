
from rkstiff.grids import construct_x_kx_rfft, construct_x_Dx_cheb
from rkstiff.etd34 import ETD34
from rkstiff.etd35 import ETD35
from rkstiff.if34 import IF34
from rkstiff.if45dp import IF45DP
from rkstiff.etd import ETDAS
from rkstiff.etd4 import ETD4
from rkstiff.etd5 import ETD5
from rkstiff.if4 import IF4
from rkstiff.solver import StiffSolverAS
import rkstiff.models as models
import numpy as np
import pytest

# Check that soliton 'shape' is preserved through propagation
def NLstub(u):
    return 

def Lstub(N):
    return np.ones(N)

def test_solver_input():
    # Test that invalid linear operator raises ValueError
    with pytest.raises(ValueError):
        L = np.zeros(shape=(4,2))
        solver = ETD34(linop=L,NLfunc=NLstub)
    # Test that invalid linear increment factor raises ValueError
    with pytest.raises(ValueError):
        solver = ETD34(linop = Lstub(10),NLfunc=NLstub,incrF = 0.8)
    # Test that invalid linear decrement factor raises ValueError
    with pytest.raises(ValueError):
        solver = ETD34(linop = Lstub(10),NLfunc=NLstub,decrF = 1.1)
    # Test that negative epsilon raises ValueError
    with pytest.raises(ValueError):
        solver = ETD34(linop = Lstub(10),NLfunc=NLstub,epsilon = -1e-3)
    # Test that invalid safetyFactor raises ValueError 
    with pytest.raises(ValueError):
        solver = ETD34(linop = Lstub(10),NLfunc=NLstub,safetyF = 1.2)
    # Test that invalid adapt_cutoff raises ValueError
    with pytest.raises(ValueError):
        solver = ETD34(linop = Lstub(10),NLfunc=NLstub,adapt_cutoff = 1)
    with pytest.raises(ValueError):
        solver = ETD34(linop = Lstub(10),NLfunc=NLstub,minh = 0)

def test_etdsolver_input():
    # Test that modecutoff is valid
    with pytest.raises(ValueError):
        solver = ETD35(linop = Lstub(10),NLfunc=NLstub,modecutoff=1.2)
    with pytest.raises(ValueError):
        solver = ETD35(linop = Lstub(10),NLfunc=NLstub,modecutoff=0)
    # Test that contour_points is an integer
    with pytest.raises(TypeError):
        solver = ETD35(linop = Lstub(10),NLfunc=NLstub,contour_points = 12.2)
    # Test that contour_points is greater than 1
    with pytest.raises(ValueError):
        solver = ETD35(linop = Lstub(10),NLfunc=NLstub,contour_points = 1)
    # Test that contour_radius is positive
    with pytest.raises(ValueError):
        solver = ETD35(linop = Lstub(10),NLfunc=NLstub,contour_radius = 0)

def test_abc_error():
    # Test that abstract base classes cannot be instantiated
    with pytest.raises(TypeError):
        solver = ETDAS(linop = Lstub(10),NLfunc=NLstub)
    with pytest.raises(TypeError):
        solver = StiffSolverAS(linop = Lstub(10),NLfunc=NLstub)


def allen_cahn_setup():
    N = 20 
    a = -1; b = 1
    x,Dx = construct_x_Dx_cheb(N,a,b)
    epsilon = 0.01
    L,NL = models.allenCahnOps(x,Dx,epsilon)
    u0 = 0.53*x + 0.47*np.sin(-1.5*np.pi*x)
    w0 = u0 - x
    w0int = w0[1:-1]; u0int = u0[1:-1]; xint = x[1:-1]
    return xint,u0int,w0int,L,NL

def burgers_setup():
    N = 1024
    a,b = -np.pi, np.pi
    x,kx = construct_x_kx_rfft(N,a,b)

    mu = 0.0005
    L,NL = models.burgersOps(kx,mu)

    u0 = np.exp(-10*np.sin(x/2)**2)
    u0FFT = np.fft.rfft(u0)
    return u0FFT,L,NL

def kdv_soliton_setup():
    N = 256
    a,b = -30,30
    x,kx = construct_x_kx_rfft(N,a,b)
    A, x0, t0 = 1., -5., 0

    h = 0.025
    steps = 200
    tf = h*steps

    u0 = models.kdvSoliton(x,A=A,x0=x0,t=t0)
    u0FFT = np.fft.rfft(u0)
    uexactFFT = np.fft.rfft(models.kdvSoliton(x,A=A,x0=x0,t=tf))
    L,NL = models.kdvOps(kx)

    return u0FFT,L,NL,uexactFFT,h,steps

def kdv_CS_step_eval(solver,u0FFT,uexactFFT,h,steps,tol):
    for _ in range(steps):
        u0FFT = solver.step(u0FFT,h)
    rel_err = np.linalg.norm(u0FFT-uexactFFT)/np.linalg.norm(uexactFFT)
    assert rel_err < tol 

def kdv_AS_step_eval(solver,u0FFT,uexactFFT,h,steps,tol):
    for _ in range(steps):
        u0FFT,h_actual,h_suggest = solver.step(u0FFT,h)
        assert (h_actual - h) < 1e-10
    rel_err = np.linalg.norm(u0FFT-uexactFFT)/np.linalg.norm(uexactFFT)
    assert rel_err < tol 

def kdv_evolve_eval(solver,u0FFT,uexactFFT,h,tf,tol):
    uFFT = solver.evolve(u0FFT,0.0,tf,h,store_data=False)
    rel_err = np.linalg.norm(uFFT-uexactFFT)/np.linalg.norm(uexactFFT)
    assert rel_err < tol 

def test_etd34_nondiag():
    xint,u0int,w0int,L,NL = allen_cahn_setup()
    solver = ETD34(linop=L,NLfunc=NL,epsilon=1e-3,contour_points=64,contour_radius=20)
    wfint = solver.evolve(w0int,t0=0,tf=60,store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0]-ufint[0]) < 0.01
    assert np.abs(u0int[7]-ufint[7]) > 1

def test_etd34():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = ETD34(linop=L,NLfunc=NL,epsilon=1e-1) # small epsilon -> step size isn't auto-reduced
    kdv_AS_step_eval(solver,u0FFT,uexactFFT,h,steps,tol=1e-4)
    solver.reset()
    solver.epsilon = 1e-4
    kdv_evolve_eval(solver,u0FFT,uexactFFT,h,tf=h*steps,tol=1e-4)

def test_etd35():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = ETD35(linop=L,NLfunc=NL,epsilon=1e-1) # small epsilon -> step size isn't auto-reduced
    kdv_AS_step_eval(solver,u0FFT,uexactFFT,h,steps,tol=1e-4)
    solver.reset()
    solver.epsilon = 1e-4
    kdv_evolve_eval(solver,u0FFT,uexactFFT,h,tf=h*steps,tol=1e-5)

def test_etd35_nondiag():
    xint,u0int,w0int,L,NL = allen_cahn_setup()
    solver = ETD35(linop=L,NLfunc=NL,epsilon=1e-4,contour_points=32,contour_radius=10)
    wfint = solver.evolve(w0int,t0=0,tf=60,store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0]-ufint[0]) < 0.01
    assert np.abs(u0int[7]-ufint[7]) > 1

def test_if34():
    u0FFT,L,NL = burgers_setup()
    solver = IF34(linop=L,NLfunc=NL,epsilon=1e-4)
    uFFT = solver.evolve(u0FFT,t0=0,tf=0.85,store_data=False)
    rel_err = np.abs(np.linalg.norm(uFFT)-np.linalg.norm(u0FFT))/np.linalg.norm(u0FFT) 
    assert rel_err < 1e-2

def test_if34_nondiag():
    xint,u0int,w0int,L,NL = allen_cahn_setup()
    solver = IF34(linop=L,NLfunc=NL,epsilon=1e-3)
    wfint = solver.evolve(w0int,t0=0,tf=60,store_data=False)
    ufint = wfint.real + xint
    assert np.abs(u0int[0]-ufint[0]) < 0.01
    assert np.abs(u0int[7]-ufint[7]) > 1

def test_if45dp():
    u0FFT,L,NL = burgers_setup()
    solver = IF45DP(linop=L,NLfunc=NL,epsilon=1e-3)
    uFFT = solver.evolve(u0FFT,t0=0,tf=0.85,store_data=False)
    rel_err = np.abs(np.linalg.norm(uFFT)-np.linalg.norm(u0FFT))/np.linalg.norm(u0FFT) 
    assert rel_err < 1e-2


def test_etd4_step():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = ETD4(linop=L,NLfunc=NL)
    kdv_CS_step_eval(solver,u0FFT,uexactFFT,h,steps,1e-6)

def test_etd4_evolve():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = ETD4(linop=L,NLfunc=NL)
    kdv_evolve_eval(solver,u0FFT,uexactFFT,h,tf=h*steps,tol=1e-6)

def test_etd5_step():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = ETD5(linop=L,NLfunc=NL)
    kdv_CS_step_eval(solver,u0FFT,uexactFFT,h,steps,tol=1e-6)

def test_etd5_evolve():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = ETD5(linop=L,NLfunc=NL)
    kdv_evolve_eval(solver,u0FFT,uexactFFT,h=h,tf=h*steps,tol=1e-6)

def test_if4_step():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = IF4(linop=L,NLfunc=NL)
    kdv_CS_step_eval(solver,u0FFT,uexactFFT,h,steps,tol=1e-5)

def test_if4_evolve():
    u0FFT,L,NL,uexactFFT,h,steps = kdv_soliton_setup()
    solver = IF4(linop=L,NLfunc=NL)
    kdv_evolve_eval(solver,u0FFT,uexactFFT,h=h,tf=h*steps,tol=1e-5)


