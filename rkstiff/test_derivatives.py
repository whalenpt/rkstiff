
from rkstiff.grids import construct_x_kx_rfft, construct_x_kx_fft
from rkstiff.grids import construct_x_Dx_cheb
from rkstiff.derivatives import dx_rfft, dx_fft
import numpy as np

def test_periodic_dx_rfft():
    N = 100
    a, b = 0, 2*np.pi
    x,kx = construct_x_kx_rfft(N,a,b)
    u = np.sin(x)
    ux_exact = np.cos(x)
    ux_approx = dx_rfft(kx,u)
    assert np.allclose(ux_exact,ux_approx)

def test_zeroboundaries_dx_rfft():
    N = 400
    a, b = -30., 30.
    x,kx = construct_x_kx_rfft(N,a,b)
    u = 1./np.cosh(x)
    ux_exact = -np.tanh(x)/np.cosh(x)
    ux_approx = dx_rfft(kx,u)
    assert np.allclose(ux_exact,ux_approx)

def test_gauss_dx_rfft():
    N = 128
    a,b = -10,10
    x,kx = construct_x_kx_rfft(N,a,b)
    u = np.exp(-x**2)
    ux_exact = -2*x*np.exp(-x**2)
    ux_approx = dx_rfft(kx,u)
    assert np.allclose(ux_exact,ux_approx)


def test_manydx_rfft():
    N = 128 
    a, b = 0, 2*np.pi
    x,kx = construct_x_kx_rfft(N,a,b)
    u = np.sin(x)
    ux_exact = np.sin(x)

    ux_approx = u.copy()
    for _ in range(4):
        ux_approx = dx_rfft(kx,ux_approx)
    rel_err = np.linalg.norm(ux_exact-ux_approx)/np.linalg.norm(ux_exact) 
    assert rel_err < 1e-6

    ux_approx = u.copy()
    ux_approx = dx_rfft(kx,ux_approx,8)
    rel_err = np.linalg.norm(ux_exact-ux_approx)/np.linalg.norm(ux_exact) 
    assert rel_err < 0.1




def test_manydx_fft():
    N = 128 
    a, b = 0, 2*np.pi
    x,kx = construct_x_kx_fft(N,a,b)
    u = np.sin(x)
    ux_exact = np.sin(x)

    ux_approx = u.copy()
    for _ in range(4):
        ux_approx = dx_fft(kx,ux_approx)
    rel_err = np.linalg.norm(ux_exact-ux_approx)/np.linalg.norm(ux_exact) 
    assert rel_err < 1e-6

    ux_approx = u.copy()
    ux_approx = dx_fft(kx,ux_approx,8)
    rel_err = np.linalg.norm(ux_exact-ux_approx)/np.linalg.norm(ux_exact) 
    assert rel_err < 0.1 

def test_periodic_dx_fft():
    N = 100
    a, b = 0, 2*np.pi
    x,kx = construct_x_kx_fft(N,a,b)
    u = np.sin(x)
    ux_exact = np.cos(x)
    ux_approx = dx_fft(kx,u)
    assert np.allclose(ux_exact,ux_approx)

def test_zeroboundaries_dx_fft():
    N = 400
    a, b = -30., 30.
    x,kx = construct_x_kx_fft(N,a,b)
    u = 1./np.cosh(x)
    ux_exact = -np.tanh(x)/np.cosh(x)
    ux_approx = dx_fft(kx,u)
    assert np.allclose(ux_exact,ux_approx)

def test_gauss_dx_fft():
    N = 128
    a,b = -10,10
    x,kx = construct_x_kx_fft(N,a,b)
    u = np.exp(-x**2)
    ux_exact = -2*x*np.exp(-x**2)
    ux_approx = dx_fft(kx,u)
    assert np.allclose(ux_exact,ux_approx)


def test_exp_trig_x_Dx_cheb():
    # standard interval [-1,1]
    N = 20; a = -1; b = 1
    x,Dx = construct_x_Dx_cheb(N,-1,1)
    u = np.exp(x)*np.sin(5*x)
    Du_exact = np.exp(x)*(np.sin(5*x)+5*np.cos(5*x))
    Du_approx = Dx.dot(u) 
    error = Du_exact - Du_approx
    assert np.linalg.norm(error)/np.linalg.norm(Du_exact) < 1e-8

    # non-standard interval [-3,3]
    N = 30; a = -3; b = 3
    x,Dx = construct_x_Dx_cheb(N,a,b)
    u = np.exp(x)*np.sin(5*x)
    Du_exact = np.exp(x)*(np.sin(5*x)+5*np.cos(5*x))
    Du_approx = Dx.dot(u) 
    error = Du_exact - Du_approx
    assert np.linalg.norm(error)/np.linalg.norm(Du_exact) < 1e-7


