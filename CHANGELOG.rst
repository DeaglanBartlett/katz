Changelog
=========

Changes to katz by version. Releases are listed at
https://github.com/DeaglanBartlett/katz/releases.

Version 0.4.0
-------------

* **Kneser-Ney smoothing** (``smoothing='kn'``, the new default; ``katz.kneser_ney``):
  interpolated Kneser-Ney with absolute discounts (``D = 0.8`` at all orders by default),
  which can be tuned by cross-validation on the corpus with ``FunctionPrior.tune_discounts()``. On
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
  score ESR function libraries with ``katz.esr_prior`` (installed as described in the README). It gives
  the same trees as ESR's parser on the physics and Feynman corpora and on samples of ESR's
  ``core_maths`` function libraries up to complexity 6.
* New options: ``code_constants`` (code constants and parameters as ``a`` rather than as
  variables), ``right_backoff`` (order in which a right child's context is shortened) and
  ``smoothing_kwargs`` (Kneser-Ney discounts).
* The prior class is now ``FunctionPrior``, as it is no longer only a Katz model.
  ``KatzPrior`` is kept as an alias of ``FunctionPrior`` for backwards compatibility.
* ``katz.esr_prior.compute_logprior`` takes ``tune_discounts`` to tune the Kneser-Ney
  discounts on the corpus before scoring, and records the tuned values in the manifest. The
  output labels name non-default discounts (e.g. ``kn_D0.6``, ``kn_tuned``), so priors with
  different discounts are written to different files.
* ``is_float`` in the parser evaluates only numbers and arithmetic instead of calling
  ``eval`` on the string.
* Documentation: a Theory page describing both models, and a worked tutorial whose outputs
  are checked with ``make -C docs doctest``.
* Bug fixes in ``katz.esr_prior``: an error on one MPI rank while reading a function library
  or computing the constant penalties is now raised on every rank (the other ranks used to
  wait indefinitely); ``get_logconst`` handles integers beyond the 64-bit range, which raised
  an error or gave a NaN penalty, and a saved NaN penalty is recomputed. A corpus equation
  that the basis cannot code raises a ``ValueError`` naming it. ``scripts/fit_benchmark.py``
  imports ESR's renamed modules (``combine_dl``, ``test_all_fisher``).

Version 0.3.0
-------------

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
* On the recovery benchmark of `Bartlett et al. (2023) <https://arxiv.org/abs/2304.06333>`__ (ESR fits up to complexity 7 of
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
* **SymPy >= 1.14 is required.** The trees of the corpus equations, and so the prior, depend
  on the SymPy version: older versions pull a numerical factor out of some square roots
  (e.g. ``sqrt(2*x0**2 + 4*x1)``), which changes the trees of 6 of the 161 physics and 2 of the
  100 Feynman equations. The reference values of the tests are for SymPy >= 1.14.
  Installing this version upgrades older SymPy installations, including environments shared
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
  katz and SymPy versions are unchanged; rank 0 alone writes the standardised corpus;
  functions that cannot be scored are NaN and counted; outputs are written atomically. Files
  labelled ``katz`` now hold the new default model; the model of version 0.2.1 is written as
  ``katz_noroot_unnorm`` (as by ``scripts/run_esr_prior.py``).
* ``KatzPrior(..., equations=[...])`` takes the corpus as a list of equations instead of a file.

Version 0.2.1
-------------

* Tree files written by ESR with ``np.str_(...)`` wrappers around the labels can be read by
  ``katz.esr_prior`` (`#15 <https://github.com/DeaglanBartlett/katz/pull/15>`_).

Version 0.2.0
-------------

* The context of a right child includes its parent. The right-child model of version 0.1.0
  dropped the most distant symbol of the context, which at n = 2 is the parent. This changes
  most priors (see version 0.3.0).
* ``GoodTuring`` no longer gives NaN when every word of a corpus has the same count; unit
  tests added (`#14 <https://github.com/DeaglanBartlett/katz/pull/14>`_).

Version 0.1.0
-------------

* The version used for priors computed before version 0.2.0, including those of
  `Bartlett, Desmond & Ferreira 2023 <https://arxiv.org/abs/2304.06333>`_ and
  `Sousa et al. 2024 <https://arxiv.org/abs/2310.16786>`_. To reproduce them, install it with
  ``pip install git+https://github.com/DeaglanBartlett/katz.git@v0.1.0``.
