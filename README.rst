Katz
----

:Katz: Function prior for symbolic regression with a Katz back-off model
:Authors: Deaglan J. Bartlett
:Homepage: https://github.com/DeaglanBartlett/katz 
:Documentation: TO DO

About
=====

Symbolic Regression algorithms attempt to find analytic expressions which accurately
fit a given data set. In many cases, one wants these to interpretable and thus one
tries to find the simplest possible expression which can accurately fit the data. 
Even if an expression is simple, if one is seeking a physical law, one may discard
an expression because it does not look like one you would expect to see in this context.
To automate this process, we produce a prior on functions based on a Katz back-off model;
by studying a corpus of previously used equations in a given field, the back-off model
returns a probability for each function based on the order of operators appearing in the
function.

Installation
=============

This repository depends on the `ESR Package <https://github.com/DeaglanBartlett/ESR>`_ to produce tree representations of equations.
To download and install the relevant code and dependencies in a new virtual environment, run

.. code:: bash

	python3 -m venv katz_env
	source katz_env/bin/activate
	git clone git@github.com:DeaglanBartlett/ESR.git
	pip install -e ESR
	git clone git@github.com:DeaglanBartlett/katz.git
	pip install -e katz

Licence and Citation
====================

Users are  required to cite the katz paper

.. code:: bibtex

  @ARTICLE{UpcomingPaper,
       author = {{Bartlett}, D.~J. and {Desmond}, H. and {Ferreira}, P.~G.},
  }


Since this depends on the ESR package, 
users are required to cite the ESR `Paper <https://arxiv.org/abs/2211.11461>`_
for which the following bibtex can be used

.. code:: bibtex

  @ARTICLE{2022arXiv2211.11461,
       author = {{Bartlett}, D.~J. and {Desmond}, H. and {Ferreira}, P.~G.},
        title = "{Exhaustive Symbolic Regression}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2022,
        month = nov,
          eid = {arXiv:2211.11461},
        pages = {arXiv:2211.11461},
  archivePrefix = {arXiv},
       eprint = {2211.11461},
  primaryClass = {astro-ph.CO},
	  url = {https://arxiv.org/abs/2211.11461},
  }

The software is available on the MIT licence:

Copyright 2023 Deaglan J. Bartlett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contributors
============

Below is a list of contributors to this repository. 

`Deaglan Bartlett <https://github.com/DeaglanBartlett>`_ (CNRS & Sorbonne Université, Institut d’Astrophysique de Paris)

Examples
========

PUT AN EXAMPLE HERE

Documentation
=============

MAKE SOME DOCUMENTATION

Acknowledgements
================
DJB is supported by the Simons Collaboration on "Learning the Universe".

TO DO
================

* Change Feynman equations to identify variables and constants as different?
* Allow more than 10 input variables when processing equations
* Make terminal node in tuples a sibling
* Allow transition between smoothed S and actual S in good_turing
* Deal with expected count if k = 0

