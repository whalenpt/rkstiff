from setuptools import setup

setup(
        name='rkstiff',
        version='0.0.1',
        description='Runge-Kutta adaptive-step solvers for nonlinear PDEs',
        py_modules=["etd34","etd35","if34","if45dp","grids","derivatives"],
        package_dir={'' : 'rkstiff'},
)
