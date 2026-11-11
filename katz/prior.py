import numpy as np

from katz.back_off import BackOff, FollowsRestriction, NormalisedBackOff
from katz.kneser_ney import KneserNey
from katz.process_equations import SymbolCoder, count_variables, standardise_file

__version__ = "0.4.0"


# Discount used by the Kneser-Ney model unless smoothing_kwargs or tune_discounts() set one.
# Cross-validation on the Physics corpus gives optima D = 0.6-0.93 (depending on n and on
# the left/right model), with the held-out codelength within ~0.07 nats per equation of its
# minimum for D within +-0.15 of the optimum.
DEFAULT_KN_DISCOUNT = 0.8


class KatzPrior:

    def __init__(
        self,
        n,
        basis_functions,
        in_eqfile=None,
        out_eqfile=None,
        input_delimiter=",",
        root_context=True,
        normalise=True,
        equations=None,
        smoothing="kn",
        smoothing_kwargs=None,
        code_constants=False,
        right_backoff="sibling",
    ):
        """Class to evaluate the probability of a function based on an n-gram language model
        (interpolated Kneser-Ney by default, or the Katz back-off model of Bartlett et al. 2023)

        Note: the default model changed in versions 0.3.0 (normalised Katz conditionals, root
        context) and 0.4.0 (Kneser-Ney smoothing). smoothing='katz', root_context=False,
        normalise=False gives the Katz model of version 0.2.1. For corpora without single-leaf
        equations, including those in data/, the single-leaf correction of 0.3.0 halves the
        log-prior of a single-leaf function in the original representation and changes no other
        prior. For corpora containing single-leaf equations, the corrected training counts can
        also change the priors of other functions. Priors computed before version 0.2.0,
        including those of Bartlett, Desmond & Ferreira (2023) and Sousa et al. (2024), used
        version 0.1.0 or earlier, whose right-child model omitted the parent from the context
        at n = 2 (in general, the most distant symbol of the context); no setting of this
        version reproduces them (see README).

        Args:
            :n (int): The length of the n-tuples to consider
            :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
            :in_eqfile (str): Name of file containing the equations to study. If None, then equations read from out_eqfile
            :out_eqfile (str): Name of file to output the standardised equations to
            :input_delimiter (str): The delimiter used in the input csv file
            :root_context (bool, default=True): Condition nodes near the root on an explicit start-of-tree context (see SymbolCoder). False gives the original representation, in which the root is predicted from the distribution of all first children
            :normalise (bool, default=True): For smoothing='katz', make every conditional distribution of the back-off models proper: negative or non-finite probabilities are set to zero and the rest divided by their sum over the vocabulary (see NormalisedBackOff). False gives the original Katz model, whose conditional probabilities need not sum to one. The Kneser-Ney model is normalised by construction and ignores this option
            :equations (list of str): Equations (with variables x0, x1, ...) to use as the corpus instead of reading a file. If given, in_eqfile and out_eqfile are ignored.
            :smoothing (str, default='kn'): Language model used for the conditional probabilities: 'kn' (interpolated Kneser-Ney) or 'katz' (Katz back-off with Good-Turing discounting, as in Bartlett et al. 2023)
            :smoothing_kwargs (dict): Extra keyword arguments for the Kneser-Ney model: discounts (default DEFAULT_KN_DISCOUNT = 0.8 at all orders; 'mkn' for the Chen-Goodman modified Kneser-Ney estimates) and discount_scale
            :code_constants (bool, default=False): Code numerical constants and parameters as "a" rather than as variables (see SymbolCoder)
            :right_backoff (str, default='sibling'): Order in which the context of a right child is shortened when backing off. 'sibling' (original behaviour) drops the most distant ancestor first, so the last context kept is the left sibling. 'parent' (n >= 2) drops the left sibling first, so the last context kept is the parent.

        Returns:
            KatzPrior: Prior model to find prior of a function given a previous set of equations

        """

        self.n = n
        if smoothing not in ("katz", "kn"):
            raise ValueError(f"Unknown smoothing method: {smoothing}")
        if right_backoff not in ("sibling", "parent"):
            raise ValueError(f"Unknown right_backoff option: {right_backoff}")
        if right_backoff == "parent" and n < 2:
            raise ValueError(
                "right_backoff='parent' needs n >= 2: for n = 1 the context of a right "
                "child is its left sibling alone"
            )
        self.smoothing = smoothing
        self.root_context = root_context
        self.normalise = normalise
        self.code_constants = code_constants
        self.smoothing_kwargs = (
            {} if smoothing_kwargs is None else dict(smoothing_kwargs)
        )
        if smoothing == "katz" and self.smoothing_kwargs:
            raise ValueError(
                "smoothing_kwargs apply to smoothing='kn' only; for Katz use normalise="
            )
        if smoothing == "kn":
            self.smoothing_kwargs.setdefault("discounts", DEFAULT_KN_DISCOUNT)
        self.side_kwargs = [{}, {}]  # left/right overrides, e.g. set by tune_discounts
        self.right_backoff = right_backoff

        if equations is not None:
            self.all_eq = list(equations)
            self.maxvar = count_variables(self.all_eq)
        else:
            self.all_eq, self.maxvar = standardise_file(
                in_eqfile, out_eqfile, input_delimiter
            )
        self.basis_functions = [
            list(  # ordered (not a set) so that the coding is reproducible
                dict.fromkeys(
                    basis_functions[0] + ["a"] + [f"x{i}" for i in range(self.maxvar)]
                )
            ),  # type0
            basis_functions[1],  # type1
            basis_functions[2],
        ]  # type2
        self.coder = SymbolCoder(
            self.basis_functions,
            root_context=root_context,
            code_constants=code_constants,
        )
        self.none_code = self.coder.code["None"]

        # Words which may be predicted (the vocabulary over which the conditionals are
        # normalised): leaves (constants and parameters are coded "x" unless
        # code_constants) and operators; a right child can also be a second leaf "y" that
        # differs from its left sibling, so it can only follow an "x"
        leaves = ["a", "x"] if code_constants else ["x"]
        ops = [
            o for o in basis_functions[1] + basis_functions[2] if o in self.coder.code
        ]
        self.vocab_left = [self.coder.code[s] for s in leaves + ops]
        self.vocab_right = self.vocab_left + [self.coder.code["y"]]
        # the left sibling ends the right-child context, or starts it for
        # right_backoff='parent' (see _right_tuple)
        self._follows = {self.coder.code["y"]: self.coder.code["x"]}
        self._sibling_position = -1 if right_backoff == "sibling" else 0

        self.eq_tuples = [
            self.coder.process_all_equations(n + 1, [eq], self.maxvar)
            for eq in self.all_eq
        ]
        self.fit(self.eq_tuples)

    def _right_tuple(self, tt):
        """Right-child prediction event (context..., word) from a full sibling tuple"""
        if self.right_backoff == "sibling":
            return tt
        ctx = tt[-(self.n + 1) : -1]
        return (ctx[-1],) + ctx[:-1] + (tt[-1],)

    def split_events(self, t):
        """
        Split the n-tuples of one equation into the prediction events of the left
        (first child / root) and right (second child) models

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :left (list): Tuples whose final element is a root or first child and whose other elements are its ancestors
            :right (list): Tuples whose final element is a second child and whose other elements are its context
        """
        left = [t[0]] + [tt[:-1] for tt in t[1:]]
        right = [self._right_tuple(tt) for tt in t[1:] if tt[-1] != self.none_code]
        return left, right

    def fit(self, eq_tuples):
        """
        (Re)build the left and right language models from the n-tuples of a set of equations

        Args:
            :eq_tuples (list): List (one entry per equation) of lists of n-tuples

        Returns:
            :None
        """
        data_left = []
        data_right = []
        for t in eq_tuples:
            left, right = self.split_events(t)
            data_left += left
            data_right += right
        if self.smoothing == "katz":
            if len(data_left) == 0 or len(data_right) == 0:
                raise ValueError(
                    "The Katz model needs a corpus with at least one binary operator "
                    "(second-child event); use smoothing='kn' for other corpora"
                )
            self.backoff_left = BackOff(data_left)
            self.backoff_right = BackOff(data_right)
            if self.normalise:
                self.backoff_left = NormalisedBackOff(
                    self.backoff_left, self.vocab_left
                )
                self.backoff_right = NormalisedBackOff(
                    self.backoff_right,
                    self.vocab_right,
                    follows=self._follows,
                    follows_position=self._sibling_position,
                )
        else:
            order = self.n + 1
            self.backoff_left = KneserNey(
                data_left,
                order=order,
                vocab=self.vocab_left,
                **{**self.smoothing_kwargs, **self.side_kwargs[0]},
            )
            self.backoff_right = FollowsRestriction(
                KneserNey(
                    [tt[-order:] for tt in data_right],
                    order=order,
                    vocab=self.vocab_right,
                    **{**self.smoothing_kwargs, **self.side_kwargs[1]},
                ),
                self._follows,
                self._sibling_position,
            )

    def tune_discounts(self, method="single", k=10, seed=0, grid=None):
        """
        Choose the Kneser-Ney discounts of the left and right models by k-fold
        cross-validation over the corpus equations, minimising the held-out negative
        log-probability, then refit on the full corpus with the chosen discounts.

        Args:
            :method (str, default='single'): 'single' (one discount D used at every order and count), 'count' (modified Kneser-Ney: discount D1 for n-grams seen once and D2+ for n-grams seen two or more times, the same at every order) or 'scale' (factor multiplying the Chen-Goodman estimates). On the Physics corpus 'count' improves on 'single' by ~0.1 nats per equation for n >= 4 and does not help for n <= 2.
            :k (int, default=10): Number of folds
            :seed (int, default=0): Seed of the random assignment of equations to folds
            :grid (array, default=None): Values to try. Defaults to 0.05-1.0 ('single'), 0.1-2.0 ('scale') or all pairs (D1, D2+) with D1 = 0.05-1.0 and D2+ = 0.1-2.0 in steps of 0.05 and 0.1 ('count', an array of shape (M, 2))

        Returns:
            :result (dict): The chosen left and right settings and the cross-validated NLL (nats per equation) on the grid
        """
        if self.smoothing != "kn":
            raise ValueError("tune_discounts requires smoothing='kn'")
        if method not in ("single", "scale", "count"):
            raise ValueError(f"Unknown tuning method: {method}")
        N = len(self.eq_tuples)
        if N < 2:
            raise ValueError("tune_discounts needs a corpus of at least two equations")
        k = min(int(k), N)
        if k < 2:
            raise ValueError(f"tune_discounts needs at least two folds, not k={k}")
        if grid is None:
            if method == "single":
                grid = np.linspace(0.05, 1.0, 39)
            elif method == "scale":
                grid = np.linspace(0.1, 2.0, 39)
            else:
                d1, d2 = np.meshgrid(
                    np.linspace(0.05, 1.0, 20), np.linspace(0.1, 2.0, 20)
                )
                grid = np.column_stack([d1.ravel(), d2.ravel()])
        grid = np.asarray(grid, dtype=float)
        if method == "count" and (grid.ndim != 2 or grid.shape[1] != 2):
            raise ValueError("grid for method='count' must have shape (M, 2)")
        if method != "count" and grid.ndim != 1:
            raise ValueError(f"grid for method='{method}' must be one-dimensional")
        if len(grid) == 0 or not np.all(np.isfinite(grid) & (grid > 0)):
            raise ValueError("grid must be non-empty, finite and positive")

        def discounts(g):
            if method == "count":  # (D1, D2, D3+) at every order, with D3+ = D2
                return np.tile([g[0], g[1], g[1]], (self.n + 1, 1)).tolist()
            return float(g)

        folds = np.array_split(np.random.default_rng(seed).permutation(N), k)
        nll = np.zeros((2, len(grid)))
        try:
            for f in folds:
                test = set(f.tolist())
                self.fit([self.eq_tuples[i] for i in range(N) if i not in test])
                models = (self.backoff_left, self.backoff_right)
                events = ([], [])
                for i in f:
                    left, right = self.split_events(self.eq_tuples[i])
                    events[0].extend((tt[-1], tt[:-1]) for tt in left)
                    events[1].extend((tt[-1], tt[-(self.n + 1) : -1]) for tt in right)
                for j, g in enumerate(grid):
                    for side in (0, 1):
                        if method == "scale":
                            models[side].set_discounts("mkn", float(g))
                        else:
                            models[side].set_discounts(discounts(g))
                        nll[side, j] -= sum(
                            np.log(models[side].prob(w, h)) for w, h in events[side]
                        )
        except BaseException:
            self.fit(self.eq_tuples)  # restore the full-corpus models
            raise
        best = [grid[np.argmin(nll[side])] for side in (0, 1)]
        if method == "scale":
            self.side_kwargs = [
                {"discounts": "mkn", "discount_scale": float(b)} for b in best
            ]
        else:
            # the candidates were evaluated with discount_scale 1, which must also
            # override any discount_scale given in smoothing_kwargs
            self.side_kwargs = [
                {"discounts": discounts(b), "discount_scale": 1.0} for b in best
            ]
        self.fit(self.eq_tuples)
        return {
            "left": self.side_kwargs[0],
            "right": self.side_kwargs[1],
            "grid": grid,
            "cv_nll": nll / N,
        }

    def event_probs(self, t):
        """
        Conditional probabilities of every node of an equation

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :pleft (np.ndarray): Probabilities of the root and first children
            :pright (np.ndarray): Probabilities of the second children given the first
        """
        tleft, tright = self.split_events(t)
        pleft = np.array([self.backoff_left.get_pbo(tt[-1], tt[:-1]) for tt in tleft])
        pright = np.array(
            [
                self.backoff_right.get_pbo(tt[-1], tt[-(self.n + 1) : -1])
                for tt in tright
            ]
        )
        return pleft, pright

    def logprior_tuples(self, t):
        """
        Natural logarithm of the prior of an equation already converted to n-tuples

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :p (float): The natural logarithm of the prior
        """
        pleft, pright = self.event_probs(t)
        with np.errstate(divide="ignore"):
            return np.sum(np.log(pleft)) + np.sum(np.log(pright))

    def logprior(self, eq):
        """
        Compute the natural logarithm of the prior of a given equation

        Args:
            :eq (str): The equation to find the prior probability of

        Returns:
            :p (float): The natural logarithm of the prior of the supplied equation
        """

        t = self.coder.process_all_equations(self.n + 1, [eq], self.maxvar)
        return self.logprior_tuples(t)


# Name that does not refer to a particular smoothing method
FunctionPrior = KatzPrior
