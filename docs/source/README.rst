Katz
----

:Katz: Function prior for symbolic regression with an n-gram language model
:Authors: Deaglan J. Bartlett, Harry Desmond
:Homepage: https://github.com/DeaglanBartlett/katz 
:Documentation: https://katz.readthedocs.io

.. image:: https://readthedocs.org/projects/katz/badge/?version=latest
  :target: https://katz.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/cs.LG-arXiv%32304.06333-B31B1B.svg
  :target: https://arxiv.org/abs/2304.06333

.. important::

   Since version 0.4.0 the default model is Kneser-Ney smoothing (``smoothing='kn'``) with an
   explicit start-of-tree context for the root (``root_context=True``). **Priors computed before
   version 0.2.0**, including those of `Bartlett, Desmond & Ferreira (2023)
   <https://arxiv.org/abs/2304.06333>`_ and the applications of the prior, **used version 0.1.0
   or earlier**, whose Katz model differs from that of later versions in the context of a right
   child, so that no setting of this version reproduces them (see version 0.3.0 in the :doc:`changelog`).
   To reproduce them, install the version used for them, e.g.
   ``pip install git+https://github.com/DeaglanBartlett/katz.git@v0.1.0``.
   ``smoothing='katz', root_context=False, normalise=False`` gives the Katz model of version
   0.2.1.

About
=====

Symbolic regression algorithms attempt to find analytic expressions which accurately
fit a given data set. In many cases, one wants these to be interpretable and thus one
tries to find the simplest possible expression which can accurately fit the data. 
Even if an expression is simple, if one is seeking a physical law, one may discard
an expression because it does not look like the kind of expression one would expect in this context.
To automate this process, we produce a prior on functions based on an n-gram language
model of the operators in a function's tree: by studying a corpus of previously used
equations in a given field, the model returns a probability for each function based on the
order of operators appearing in the function. The conditional probabilities are smoothed
with interpolated Kneser-Ney by default, or with the Katz back-off model of the original
paper.

Installation
=============

The prior itself needs only NumPy, SciPy, pandas and SymPy: equations are converted to trees by
``katz.tree_parser``, which reproduces the parser of the
`ESR Package <https://github.com/DeaglanBartlett/ESR>`_, so the prior can be used with any symbolic
regression method. ESR (with ``mpi4py``) is needed only to score ESR function libraries with
``katz.esr_prior`` and to run the ESR tests. SymPy 1.14 or later is required: older versions
parse a few corpus equations differently (a numeric factor is taken out of a square root), which
changes the prior by up to a few nats for some functions. To download and install the code with
ESR in a new virtual environment, run

.. code:: bash

	python3 -m venv katz_env
	source katz_env/bin/activate
	git clone git@github.com:DeaglanBartlett/ESR.git
	pip install -e ESR
	git clone git@github.com:DeaglanBartlett/katz.git
	pip install -e katz


Example
========

To generate a prior with ``n = 2`` from the equations of the AI Feynman database
(`Udrescu & Tegmark 2020 <https://arxiv.org/abs/1905.11481>`_), which are included in
``data/FeynmanEquations.csv``, and evaluate it on some trial functions, run the code below.
Here ``n`` is the number of symbols from which each symbol is predicted, so ``n = 2`` gives a
3-gram model (in general, ``n`` gives an (n+1)-gram model):

.. code-block:: python

	from katz.prior import FunctionPrior
	
	n = 2
	basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                ["+", "-", "*", "/", "pow"]]
    
    	kp = FunctionPrior(n, basis_functions, 'data/FeynmanEquations.csv', 'data/NewFeynman.csv')
    	
	for eq in ['x0**2', 'sin(x0) + sin(x1)', 'sin(sin(x0+x1))']:
            p = kp.logprior(eq)
            print(eq, p)


Smoothing and representation options
------------------------------------

The models and options below are described in detail in :doc:`theory`.

By default (``smoothing='kn'``) the conditional probabilities are estimated with
interpolated Kneser-Ney smoothing, with a single absolute discount ``D = 0.8`` at all
orders. This assigns a non-zero probability to every operator of the basis set in every
context and is less sensitive to the choice of ``n`` than the Katz model. The default
discount was chosen on the physics corpus, so for other corpora tune the discounts by
cross-validation with ``tune_discounts()``. They can also be set with
``smoothing_kwargs={'discounts': D}``, or estimated with the Chen-Goodman formulae
(``smoothing_kwargs={'discounts': 'mkn'}``, not recommended for corpora of fewer than
~100 equations). ``tune_discounts(method='count')`` tunes separate discounts for n-grams seen
once and n-grams seen more than once (modified Kneser-Ney), which improves on a single tuned
discount by ~0.1 nats per equation for ``n >= 4``. ``smoothing='katz'`` gives the Katz
back-off model of `Bartlett et al. (2023) <https://arxiv.org/abs/2304.06333>`__, with conditional probabilities normalised by
default (``normalise=True``); ``smoothing='katz', root_context=False, normalise=False``
gives the Katz model of version 0.2.1 (see version 0.3.0 in the :doc:`changelog`).

The root and the nodes near it are conditioned on an explicit start-of-tree context
(``root_context=True``, the default). This gives the largest improvement in held-out tests,
but makes single-leaf functions considerably less probable, since no corpus equation is a
bare leaf; ``root_context=False`` restores the original representation, in which the root
is predicted from the distribution of all first children. ``code_constants=True`` codes constants
and parameters as ``a`` rather than as variables (by default they are treated like
variables, as physical constants are in the corpus, so e.g. ``pow(x0, a0)`` and
``pow(a0, x0)`` have the same prior).

.. code-block:: python

	kp = FunctionPrior(3, basis_functions, 'data/PhysicsEquations.csv', 'data/NewPhysics.csv',
	                  input_delimiter=';')
	kp.tune_discounts()
	print(kp.logprior('sin(x0) + sin(x1)'))

``KatzPrior``, the name of this class before version 0.4.0, is kept as an alias of
``FunctionPrior`` for backwards compatibility. ``katz.esr_prior.compute_logprior``, which scores
an ESR function library, names its output files after the prior configuration (e.g.
``kn_codelen_3_5.txt``, ``kn_D0.6_codelen_3_5.txt``, ``kn_tuned_codelen_3_5.txt``,
``katz_noroot_unnorm_codelen_2_5.txt``) and writes a JSON manifest of the configuration and
corpus beside them; existing files are reused only if the manifest matches. With
``tune_discounts`` it tunes the discounts on the corpus before scoring and records the tuned
values in the manifest:

.. code-block:: python

	from katz.esr_prior import compute_logprior

	compute_logprior(5, 3, basis_functions, 'ESR/esr/function_library/core_maths',
	                 'data/PhysicsEquations.csv', 'data/NewPhysics.csv',
	                 input_delimiter=';', tune_discounts='single')

Changes
=======

The changes in each version, including how to reproduce priors computed with earlier
versions, are listed in the :doc:`changelog`.

Licence and Citation
====================

Users are required to cite the katz `paper <https://arxiv.org/abs/2304.06333>`__:

.. code:: bibtex

  @ARTICLE{2023arXiv2304.06333,
       author = {{Bartlett}, D.~J. and {Desmond}, H. and {Ferreira}, P.~G.},
        title = "{Priors for symbolic regression}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning},
         year = 2023,
        month = apr,
          eid = {arXiv:2304.06333},
        pages = {arXiv:2304.06333},
  archivePrefix = {arXiv},
       eprint = {2304.06333},
  primaryClass = {cs.LG},
          url = {https://arxiv.org/abs/2304.06333},
  }


Since katz uses code that originated in the ESR package,
users are required to cite the ESR `paper <https://arxiv.org/abs/2211.11461>`__,
for which the following BibTeX can be used:

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

The software is available under the MIT licence:

Copyright 2023 Deaglan J. Bartlett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contributors
============

Below is a list of contributors to this repository. 

* `Deaglan Bartlett <https://github.com/DeaglanBartlett>`_
* `Harry Desmond <https://github.com/harrydesmond>`_

Documentation
=============

The documentation for this project can be found
`at this link <https://katz.readthedocs.io/>`_.
