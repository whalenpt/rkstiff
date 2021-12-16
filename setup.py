from setuptools import setup

with open("README.md","r") as f:
    long_description = f.read()
exec(open("rkstiff/version.py").read())

setup(
        name='rkstiff',
        version=__version__,
        description='Runge-Kutta adaptive-step and constant-step solvers for nonlinear PDEs',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/whalenpt/rkstiff",
        author="Patrick Whalen",
        author_email="whalenpt@gmail.com",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
        extras_require= {
            "dev": [
                "pytest","twine",
                ],
            },
        packages=["rkstiff"],
        package_dir={'.' : 'rkstiff'},
        python_requires='>=3.6.0',
        install_requires=["numpy>=1.14.0","scipy>=1.3.2"]
)
