import re

from setuptools import setup

with open("katz/prior.py") as f:
    version = re.search(r'^__version__ = "(.+)"', f.read(), re.MULTILINE).group(1)

setup(
    name="katz",
    version=version,
    description="A Python package to produce priors on functions",
    url="https://github.com/DeaglanBartlett/katz",
    author="Deaglan Bartlett",
    author_email="deaglan.bartlett@iap.fr",
    license="MIT licence",
    packages=["katz"],
    install_requires=[
        "numpy",
        "scipy",
        "sympy>=1.14",  # the corpus parse, and so the prior, depends on it
        "matplotlib",
        "pandas",
    ],
    extras_require={
        # scoring ESR function libraries (katz.esr_prior) and the ESR tests
        "esr": ["esr", "mpi4py"],
        "docs": ["sphinx>=5.0", "sphinx-rtd-theme", "myst-parser"],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
)
