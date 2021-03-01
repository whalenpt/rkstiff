from setuptools import setup

with open("README.md","r") as f:
    long_description = f.read()

setup(
        name='rkstiff',
        version='0.0.0a',
        description='Runge-Kutta adaptive-step solvers for nonlinear PDEs',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/whalenpt/rkstiff",
        author="Patrick Whalen",
        author_email="whalenpt@gmail.com",
        extras_require= {
            "dev": [
                "pytest","twine",
                ],
            },
        packages=["rkstiff"],
        package_dir={'.' : 'rkstiff'},
        setup_requires=["numpy","scipy"]
        install_requires=["numpy","scipy"]
)
