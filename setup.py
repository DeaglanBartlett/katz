import re

from setuptools import setup

with open("fnprior/prior.py") as f:
    version = re.search(r'^__version__ = "(.+)"', f.read(), re.MULTILINE).group(1)

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="fnprior",
    version=version,
    description="Priors on functions for symbolic regression from n-gram language models",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/DeaglanBartlett/fnprior",
    project_urls={
        "Documentation": "https://fnprior.readthedocs.io",
        "Changelog": "https://github.com/DeaglanBartlett/fnprior/blob/main/CHANGELOG.rst",
        "Paper": "https://arxiv.org/abs/2304.06333",
    },
    author="Deaglan Bartlett, Harry Desmond",
    author_email="deaglan.bartlett@physics.ox.ac.uk",
    license="MIT",
    # katz is the name of the package before version 0.5.0, kept as an alias until 0.8.0
    packages=["fnprior", "katz"],
    package_data={"fnprior": ["data/*.csv"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "sympy>=1.14",  # the corpus parse, and so the prior, depends on it
        "matplotlib",
        "pandas",
    ],
    extras_require={
        # scoring ESR function libraries (fnprior.esr_prior) and the ESR tests. ESR itself
        # is not on PyPI and must be installed from GitHub (see the README)
        "esr": ["mpi4py"],
        "docs": ["sphinx>=5.0", "sphinx-rtd-theme", "myst-parser"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
