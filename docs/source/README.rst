Katz
----

:Katz: Function prior for symbolic regression with an n-gram language model
:Authors: Deaglan J. Bartlett
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
   child, so that no setting of this version reproduces them (see "Changes in version 0.3.0").
   To reproduce them, install the version used for them, e.g.
   ``pip install git+https://github.com/DeaglanBartlett/katz.git@v0.1.0``.
   ``smoothing='katz', root_context=False, normalise=False`` gives the Katz model of version
   0.2.1.

About
=====

Symbolic Regression algorithms attempt to find analytic expressions which accurately
fit a given data set. In many cases, one wants these to interpretable and thus one
tries to find the simplest possible expression which can accurately fit the data. 
Even if an expression is simple, if one is seeking a physical law, one may discard
an expression because it does not look like one you would expect to see in this context.
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

To generate a 2-gram prior from the `AI Feynman function set <https://space.mit.edu/home/tegmark/aifeynman.html>`_ 
(`arXiv:1905.11481 <https://arxiv.org/abs/1905.11481>`_) and evaluate this on some trial functions, run the following

.. code-block:: python

	from katz.prior import KatzPrior
	
	n = 2
	basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                ["+", "-", "*", "/", "pow"]]
    
    	kp = KatzPrior(n, basis_functions, 'data/FeynmanEquations.csv', 'data/NewFeynman.csv')
    	
	for eq in ['x0**2', 'sin(x0) + sin(x1)', 'sin(sin(x0+x1))']:
            p = kp.logprior(eq)
            print(eq, p)


Smoothing and representation options
------------------------------------

By default (``smoothing='kn'``) the conditional probabilities are estimated with
interpolated Kneser-Ney smoothing, with a single absolute discount ``D = 0.8`` at all
orders. This assigns a non-zero probability to every operator of the basis set in every
context and is less sensitive to the choice of ``n`` than the Katz model. The discounts can
instead be tuned by cross-validation on the corpus with ``tune_discounts()``, set with
``smoothing_kwargs={'discounts': D}``, or estimated with the Chen-Goodman formulae
(``smoothing_kwargs={'discounts': 'mkn'}``, not recommended for corpora of fewer than
~100 equations). ``tune_discounts(method='count')`` tunes separate discounts for n-grams seen
once and n-grams seen more than once (modified Kneser-Ney), which improves on a single tuned
discount by ~0.1 nats per equation for ``n >= 4``. ``smoothing='katz'`` gives the Katz
back-off model of Bartlett et al. (2023), with conditional probabilities normalised by
default (``normalise=True``); ``smoothing='katz', root_context=False, normalise=False``
gives the Katz model of version 0.2.1 (see "Changes in version 0.3.0").

The root and the nodes near it are conditioned on an explicit start-of-tree context
(``root_context=True``, the default). This gives the largest improvement in held-out tests,
but makes single-leaf functions considerably less probable, since no corpus equation is a
bare leaf; ``root_context=False`` restores the original representation, in which the root
is predicted from the distribution of all first children. ``code_constants=True`` codes constants
and parameters as ``a`` rather than as variables (by default they are treated like
variables, as physical constants are in the corpus, so e.g. ``pow(x0, a0)`` and
``pow(a0, x0)`` have the same prior).

.. code-block:: python

	kp = KatzPrior(3, basis_functions, 'data/PhysicsEquations.csv', 'data/NewPhysics.csv',
	               input_delimiter=';')
	kp.tune_discounts()
	print(kp.logprior('sin(x0) + sin(x1)'))

``FunctionPrior`` is an alias of ``KatzPrior``. ``katz.esr_prior.compute_logprior``, which scores
an ESR function library, names its output files after the prior configuration (e.g.
``kn_codelen_3_5.txt``, ``katz_noroot_unnorm_codelen_2_5.txt``) and writes a JSON
manifest of the configuration and corpus beside them; existing files are reused only if the
manifest matches.

Changes in version 0.4.0
========================

* **Kneser-Ney smoothing** (``smoothing='kn'``, the new default; ``katz.kneser_ney``):
  interpolated Kneser-Ney with absolute discounts (``D = 0.8`` at all orders by default),
  which can be tuned by cross-validation on the corpus with ``KatzPrior.tune_discounts()``. On
  held-out physics equations (leave one equation family out, root context, discounts tuned on
  the training folds, 159 equations) its codelength is lower than that of the normalised Katz model by
  0.27 +/- 0.10, 0.36 +/- 0.12, 0.56 +/- 0.17, 1.02 +/- 0.23 and 1.09 +/- 0.26 nats per
  equation at n = 2, 3, 4, 6 and 10. On the ESR recovery benchmark described under 0.3.0 it
  shows no advantage: at n = 3 with root context the generating function is ranked first in
  54% of mocks with Kneser-Ney and 61% with Katz, but 9 of the 11 mocks that differ are
  Nguyen-8 (sqrt(x)), decided by about 1 nat in the prior of sqrt(x) at the root, and without
  them the difference is not significant. Unlike Katz, Kneser-Ney assigns no function a zero
  prior probability. As in the normalised Katz model, ``y`` has zero probability after any left
  sibling other than ``x``.
* **Built-in equation parser** (``katz.tree_parser``): a copy of the parts of ESR's parser
  that the prior uses, so that computing priors no longer needs ESR, which is needed only to
  score ESR function libraries with ``katz.esr_prior`` (installed as described above). It gives
  the same trees as ESR's parser on the physics and Feynman corpora and on samples of ESR's
  ``core_maths`` function libraries up to complexity 6.
* New options: ``code_constants`` (code constants and parameters as ``a`` rather than as
  variables), ``right_backoff`` (order in which a right child's context is shortened) and
  ``smoothing_kwargs`` (Kneser-Ney discounts). ``FunctionPrior`` is an alias of
  ``KatzPrior``.

Changes in version 0.3.0
========================

Version 0.3.0 changed the default Katz model; ``smoothing='katz', root_context=False,
normalise=False`` gives the model of version 0.2.1, apart from the single-leaf correction
below.

This does not reproduce priors computed before version 0.2.0, including those of
`Bartlett, Desmond & Ferreira 2023 <https://arxiv.org/abs/2304.06333>`_ and
`Sousa et al. 2024 <https://arxiv.org/abs/2310.16786>`_, which used version 0.1.0 or earlier
versions with the same right-child model. Version 0.2.0 added the parent to the context of a
right child: the right-child model of version 0.1.0 dropped the most distant symbol of the
context, which at n = 2 is the parent. This changes most priors. Trained on the physics
corpus at n = 2, it changes 94% of the priors of ESR's ``core_maths`` functions up to
complexity 5 (median 0.5 nats, maximum 4.7 nats). For the 20 inflationary potentials ranked
with this prior in Sousa et al. (2024) (n = 9), it changes the function codelength by -1.1 to
+2.5 nats, whereas version 0.1.0 reproduces their values to within 0.2 nats. To reproduce
published priors, install the version used for them, e.g. version 0.1.0
(``pip install git+https://github.com/DeaglanBartlett/katz.git@v0.1.0``).

* **Normalised conditional probabilities** (``normalise=True``, default). The back-off
  probabilities of earlier versions need not form a probability distribution. When the seen
  continuations of a context exhaust the support of the lower-order distribution, the
  probability reserved for unseen words is lost: trained on the physics corpus, the
  right-child conditionals sum to as little as 0.90. In very small corpora a fitted
  Good-Turing slope above -1 inflates the counts, giving probabilities above one or below
  zero, and a zero denominator in a back-off weight gives NaN. Negative and non-finite values
  are now set to zero and each conditional distribution is divided by its sum over the
  vocabulary (``katz.back_off.NormalisedBackOff``). The second of two different leaves
  (coded ``y``) can only follow a leaf, so it has zero probability after any other left
  sibling; the back-off model gives it up to 0.3 there.
* **Root context** (``root_context=True``, default). Lineages that reach the root are padded
  with a start-of-tree symbol, so the root and the nodes near it are predicted from their
  position in the tree. In earlier versions the root was predicted from the distribution of
  all first children, which is dominated by leaves, whereas real roots are almost always
  ``*``, ``/``, ``+`` or ``pow``.
* Together these lower the held-out codelength of physics equations (leave one equation
  family out, 159 equations) by 1.59 +/- 0.07, 2.01 +/- 0.10 and 2.21 +/- 0.14 nats per
  equation at n = 2, 3 and 4. Normalisation alone contributes 0.07-0.15 nats, most of it from
  the zero probability of ``y`` where it cannot occur.
* On the recovery benchmark of Bartlett et al. (2023) (ESR fits up to complexity 7 of
  Nguyen-8 and Korns-1, 4, 6 and 7 mocks; 7 sample sizes and 5 noise realisations each, 175
  mocks), the generating function is ranked first in 54% of mocks at n = 2 and 61% at
  n = 3 with the new defaults, against 42% with the model of version 0.2.1 (n = 2) and 52%
  with ESR's default k log n function codelength. The gain comes from the root context: at
  n = 3 without it 43% are recovered. Normalisation changes the verdict for 1 of the 175
  mocks. Most of the gain is on Nguyen-8 (sqrt(x)), recovered in 20% of mocks by the
  version 0.2.1 model and in 63% (n = 2) and 91% (n = 3) by the new one; Korns-6 is
  recovered by none of the Katz priors. The priors of version 0.1.0 were not rerun on this
  benchmark.
* **Single-leaf functions.** A function that is a single leaf (a variable or a constant)
  counted its root event twice, both when scored and when it was a training equation. For
  corpora without single-leaf equations, including those in ``data/``, a single-leaf
  function's log-prior in the original representation is now half its previous value, and no
  other prior changes. For corpora containing single-leaf equations, the corrected training
  counts can also change the priors of other functions.
* **sympy >= 1.14 is required.** The trees of the corpus equations, and so the prior, depend
  on the sympy version: older versions pull a numerical factor out of some square roots
  (e.g. ``sqrt(2*x0**2 + 4*x1)``), which changes the trees of 6 of the 161 physics and 2 of the
  100 Feynman equations. The reference values of the tests are for sympy >= 1.14.
  Installing this version upgrades older sympy installations, including environments shared
  with ESR (which accepts any version).
* **Bug fixes that change no prior of the version 0.2.1 model for the corpora in** ``data/``: a
  bare variable (``x0``) could not be split into tokens; ``standardise_file`` with
  ``in_eqfile=None`` counted variable occurrences rather than variables; variables and parameters beyond the corpus's variable
  range (e.g. ``x12`` for a corpus in ``x0``-``x10``) were not declared real, so
  ``sqrt(x12**2)`` was not simplified and could fail to convert to a tree; a lower-case
  ``abs`` was not stripped; the order of the nullary basis depended on Python's string
  hashing; a corpus without binary operators failed with ``max() arg is an empty sequence``.
* ``katz.esr_prior``: output files are named by configuration (``katz``, ``katz_noroot``,
  ``katz_noroot_unnorm``, ...) and reused only if a JSON manifest shows that the prior
  configuration, the corpus, the candidate functions, the constant penalties, the scoring code
  (as parsed, so reformatting or editing comments does not invalidate saved priors) and the
  katz and sympy versions are unchanged; rank 0 alone writes the standardised corpus;
  functions that cannot be scored are NaN and counted; outputs are written atomically. Files
  labelled ``katz`` now hold the new default model; the model of version 0.2.1 is written as
  ``katz_noroot_unnorm`` (as by ``scripts/run_esr_prior.py``).
* ``KatzPrior(..., equations=[...])`` takes the corpus as a list of equations instead of a file.

Licence and Citation
====================

Users are  required to cite the katz `Paper <https://arxiv.org/abs/2304.06333>`_

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

Documentation
=============

The documentation for this project can be found
`at this link <https://katz.readthedocs.io/>`_

Acknowledgements
================
DJB is supported by the Simons Collaboration on `Learning the Universe <https://www.learning-the-universe.org/>`_.

