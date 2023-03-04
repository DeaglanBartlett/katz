from setuptools import setup

setup(
    name='katz',
    version='0.1.0',
    description='A Python package to produce priors on functions',
    url='https://github.com/DeaglanBartlett/katz',
    author='Deaglan Bartlett',
    author_email='deaglan.bartlett@iap.fr',
    license='MIT licence',
    packages=['katz'],
    install_requires=['esr',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
