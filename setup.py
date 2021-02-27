from setuptools import setup

with open("README.md","r") as f:
    long_description = f.read()

setup(
        name='rkstiff',
        version='0.0.1',
        description='Runge-Kutta adaptive-step solvers for nonlinear PDEs',
        long_description=long_description,
        packages=["rkstiff"],
        package_dir={'.' : 'rkstiff'},
        setup_requires=["numpy","scipy"]
)
