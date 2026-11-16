.. default-role:: math

Tutorial
========

This page builds a prior from the corpus of physics equations in ``data/``, looks at the
probabilities it gives to a few functions and where they come from, compares the main
options, and then scores a function library of
`Exhaustive Symbolic Regression <https://github.com/DeaglanBartlett/ESR>`_ (ESR). The code
runs from the root of the repository. The outputs shown are checked by running the page as a
doctest (``make -C docs doctest``); the last section needs ESR and ``mpi4py``.


Building a prior
----------------

The basis lists the operators that functions may contain: nullary (leaves), unary and
binary. It must contain every operator of the corpus, which for the physics corpus includes
``arccos``. This is the basis of ESR's ``core_maths`` library:

.. doctest::

   >>> import numpy as np
   >>> from katz.prior import FunctionPrior
   >>> basis_functions = [["a", "x"],
   ...                    ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
   ...                    ["+", "-", "*", "/", "pow"]]
   >>> kp = FunctionPrior(2, basis_functions, "data/PhysicsEquations.csv",
   ...                    "data/NewPhysics.csv", input_delimiter=";")
   >>> len(kp.all_eq)
   161

The corpus file is read, its variables are renamed ``x0, x1, ...``, and the standardised
equations are written to the second file. The first argument, ``n = 2``, is the number of
symbols from which each symbol of a tree is predicted: the first (or only) child of an
operator is predicted from its parent and grandparent, and the second child of a binary
operator from its parent and its left sibling. The default model is interpolated Kneser-Ney
with a discount ``D = 0.8`` and a start-of-tree context for the root (see :doc:`theory`).


The prior of a function
-----------------------

``logprior`` returns the natural logarithm of the prior probability:

.. doctest::

   >>> for f in ["x0*sin(x1)", "x1*sin(x0)", "sin(x0)", "x0**2", "sqrt(x0**2 + x1**2)",
   ...           "exp(-x0**2)", "arcsin(arccos(x0))"]:
   ...     print(f"{f:20s} {kp.logprior(f):8.3f}")
   x0*sin(x1)             -5.141
   x1*sin(x0)             -5.141
   sin(x0)                -8.451
   x0**2                  -6.766
   sqrt(x0**2 + x1**2)    -6.123
   exp(-x0**2)            -9.546
   arcsin(arccos(x0))    -11.550

Every leaf is coded ``x`` (constants and parameters too, by default), so the prior depends
only on the shape of the tree and its operators: ``x0*sin(x1)`` and ``x1*sin(x0)`` have the
same prior. The prior is not a measure of simplicity. ``sin(x0)`` has fewer nodes than
``x0*sin(x1)`` but is less probable, because no equation of the corpus has a sine at the
root (117 of the 161 are products), and the familiar ``sqrt(x0**2 + x1**2)`` is more probable than ``x0**2`` although it
contains it.

Each node contributes the probability of its symbol given its context. ``event_probs`` gives
these probabilities, and the coder's dictionary gives the symbols' names (``None`` is the
start-of-tree symbol):

.. doctest::

   >>> name = {code: word for word, code in kp.coder.code.items()}
   >>> t = kp.coder.process_all_equations(kp.n + 1, ["x0*sin(x1)"], kp.maxvar)
   >>> first, second = kp.split_events(t)
   >>> p_first, p_second = kp.event_probs(t)
   >>> for kind, events, probs in [("first", first, p_first), ("second", second, p_second)]:
   ...     for e, p in zip(events, probs):
   ...         context = ", ".join(name[w] for w in e[-kp.n - 1:-1])
   ...         print(f"{kind:6s} ({context:10s}) -> {name[e[-1]]:3s} {p:.4f}")
   first  (None, None) -> *   0.7306
   first  (None, *   ) -> x   0.9608
   first  (*, sin    ) -> x   0.9026
   second (*, x      ) -> sin 0.0092
   >>> print(f"{np.log(p_first).sum() + np.log(p_second).sum():.3f}")
   -5.141

The root is a product with probability 0.73, and its first child is a leaf with probability
0.96. Most of the cost is in the second child: given a product whose first factor is a leaf,
the model gives the second factor a probability of only 0.009 of being a sine.


Options
-------

The main options are the smoothing method, the root context and the coding of constants.
Passing ``None`` as the input file reads the standardised corpus written above:

.. doctest::

   >>> models = {
   ...     "default": kp,
   ...     "katz": FunctionPrior(2, basis_functions, None, "data/NewPhysics.csv",
   ...                           smoothing="katz"),
   ...     "no root": FunctionPrior(2, basis_functions, None, "data/NewPhysics.csv",
   ...                              root_context=False),
   ...     "constants": FunctionPrior(2, basis_functions, None, "data/NewPhysics.csv",
   ...                                code_constants=True),
   ... }
   >>> print(" " * 14 + "".join(f"{m:>11s}" for m in models))
                     default       katz    no root  constants
   >>> for f in ["x0", "a0*x0", "pow(x0, a0)", "pow(a0, x0)", "tanh(log(x0))", "x0 + 1/x1"]:
   ...     print(f"{f:14s}" + "".join(f"{m.logprior(f):11.3f}" for m in models.values()))
   x0                 -5.108     -4.063     -3.083     -5.167
   a0*x0              -2.088     -2.106     -2.376     -3.019
   pow(x0, a0)        -6.766     -7.057     -5.254     -6.818
   pow(a0, x0)        -6.766     -7.057     -5.254    -11.170
   tanh(log(x0))     -14.036       -inf    -13.697    -14.168
   x0 + 1/x1         -10.802       -inf    -11.288    -10.618

* ``smoothing="katz"`` is the Katz back-off model of the original paper (normalised and
  with the root context by default). It gives zero probability to any function containing a
  symbol never seen in that role in the corpus: here ``tanh`` and ``log``, which do not occur
  in the physics corpus, and ``inv`` as a second child. Kneser-Ney gives every function a
  non-zero probability.
* ``root_context=False`` predicts the root from the distribution of all first children, as
  before version 0.3.0. This makes single leaves and short functions more probable, since
  no corpus equation is a bare leaf.
* ``code_constants=True`` codes constants and parameters as ``a`` rather than ``x``, so that
  ``pow(x0, a0)`` (a power law) and ``pow(a0, x0)`` (an exponential) get different priors.

``smoothing="katz", root_context=False, normalise=False`` gives the Katz model of version
0.2.1. Priors computed before version 0.2.0 need version 0.1.0 (see :doc:`changelog`).


Tuning the discounts
--------------------

The default discount was chosen on the physics corpus. For another corpus, choose the
discounts by cross-validation over its equations; ``tune_discounts`` refits the model with
the best values, separately for first and second children:

.. doctest::

   >>> res = kp.tune_discounts()
   >>> print(f"{res['left']['discounts']:.3f} {res['right']['discounts']:.3f}")
   0.750 0.725
   >>> print(f"{kp.logprior('x0*sin(x1)'):.3f}")
   -5.113

``method="count"`` tunes separate discounts for n-grams seen once and more than once, and
``method="scale"`` a factor multiplying the Chen-Goodman estimates.


Scoring an ESR function library
-------------------------------

ESR uses the prior through files that give `-\log P(f)` for every function of a library.
This example generates ESR's ``core_maths`` functions of complexity 4 in a temporary
directory, computes the constant terms of the codelength with ``get_logconst``, and then
the priors with ``compute_logprior``, here with the discounts tuned on the corpus (ESR's
progress messages are suppressed):

.. doctest::
   :skipif: not have_esr

   >>> import contextlib, io, json, os, tempfile
   >>> import esr.generation.duplicate_checker
   >>> from katz.esr_prior import compute_logprior, get_logconst
   >>> tmp = tempfile.TemporaryDirectory()
   >>> dirname = os.path.join(tmp.name, "")
   >>> with contextlib.redirect_stdout(io.StringIO()):
   ...     esr.generation.duplicate_checker.main("core_maths", 4, fn_dir=dirname)
   ...     get_logconst(4, dirname)
   ...     compute_logprior(4, 2, basis_functions, dirname, "data/PhysicsEquations.csv",
   ...                      "data/NewPhysics.csv", input_delimiter=";",
   ...                      tune_discounts="single")
   >>> for f in sorted(os.listdir(dirname + "compl_4")):
   ...     if f.startswith("kn"):
   ...         print(f)
   kn_tuned_codelen_2_4.txt
   kn_tuned_logprior_2_4.txt
   kn_tuned_prior_2_4.json

The files are named after the prior (``kn_tuned``), ``n`` and the complexity.
``kn_tuned_logprior_2_4.txt`` holds `-\log P(f)`, and ``kn_tuned_codelen_2_4.txt`` adds
`\sum_i \log |c_i|` over the integer constants `c_i` of each function. The manifest records
the configuration, the tuned discounts and checksums of the inputs, so that the files are
reused only if nothing has changed:

.. doctest::
   :skipif: not have_esr

   >>> d = dirname + "compl_4/"
   >>> manifest = json.load(open(d + "kn_tuned_prior_2_4.json"))
   >>> manifest["n_functions"], manifest["n_zoo"], manifest["n_failed"]
   (64, 2, 0)
   >>> fcns = open(d + "all_equations_4.txt").read().splitlines()
   >>> codelen = np.loadtxt(d + "kn_tuned_codelen_2_4.txt")
   >>> shortest = {}  # a function can occur several times, from different trees
   >>> for f, c in zip(fcns, codelen):
   ...     if np.isfinite(c):
   ...         shortest[f] = min(c, shortest.get(f, np.inf))
   >>> ranked = sorted(shortest.items(), key=lambda fc: (fc[1], fc[0]))
   >>> for f, c in ranked[:5] + ranked[-2:]:
   ...     print(f"{f:22s} {c:7.3f}")
   a0*a1                    2.089
   a0*x                     2.089
   1/(2*x)                  3.443
   a0/a1                    3.443
   a0/x                     3.443
   pow(x,(1/a0))           17.711
   pow(x,(1/x))            17.711

The two functions that ESR simplifies to complex infinity (``zoo``) have no prior (NaN).
``a0*a1`` and ``a0*x`` have the same codelength because parameters are coded like variables.
``1/(2*x)`` occurs twice in the library, from the trees ``inv(x + x)`` and ``inv(2*x)``; only
the second contains an integer constant, so the first has the shorter codelength, which is
the one shown (ESR likewise keeps the shortest description of a function).
To use these files in ESR, set ``fnprior_prefix = "kn_tuned_codelen_2_"`` on the likelihood,
in place of the default ``"aifeyn_"``.

.. testcleanup::
   :skipif: not have_esr

   tmp.cleanup()
