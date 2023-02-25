import rkstiff.grids
import numpy as np


def test_hankel():
    HT = rkstiff.grids.HankelTransform(nr=50, rmax=2.0)
    a = 4

    f1 = np.exp(-(a**2) * HT.r**2)
    fsp1_ex = np.exp(-HT.kr**2 / (4 * a**2)) / (2 * a**2)
    fsp1 = HT.ht(f1)
    error1 = np.linalg.norm(fsp1 - fsp1_ex) / np.linalg.norm(fsp1_ex)
    assert error1 < 1e-10

    HT.rmax = 4
    sigma = 2
    w = 0.5
    f2 = np.exp(-sigma * HT.r**2) * np.sin(w * HT.r**2)
    omega = 1.0 / (4 * (sigma**2 + w**2))
    fsp2_ex = (
        -2
        * omega
        * np.exp(-sigma * omega * HT.kr**2)
        * (-w * np.cos(w * omega * HT.kr**2) + sigma * np.sin(w * omega * HT.kr**2))
    )
    fsp2 = HT.ht(f2)
    error2 = np.linalg.norm(fsp2 - fsp2_ex) / np.linalg.norm(fsp2_ex)
    assert error2 < 1e-10

    HT.nr = 25
    HT.rmax = 3
    f3 = np.exp(-a * HT.r)
    fsp3_ex = a * np.power(a**2 + HT.kr**2, -3.0 / 2)
    fsp3 = HT.ht(f3)
    error3 = np.linalg.norm(fsp3 - fsp3_ex) / np.linalg.norm(fsp3_ex)
    assert error3 < 1e-2
