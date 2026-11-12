import numpy as np

from katz.back_off import BackOff, NormalisedBackOff
from katz.process_equations import SymbolCoder, count_variables, standardise_file

__version__ = "0.3.0"


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
    ):
        """Class to evaluate the probability of a function based on an n-gram Katz back-off model

        Note: before version 0.3.0 the model was that of root_context=False, normalise=False,
        and a function consisting of a single leaf (x or a constant) counted the root event
        twice. Pass root_context=False, normalise=False for the model of version 0.2.1. For
        corpora without single-leaf equations, including those in data/, the single-leaf
        correction halves the log-prior of a single-leaf function in the original
        representation and changes no other prior. For corpora containing single-leaf
        equations, the corrected training counts can also change the priors of other
        functions. Priors computed before version 0.2.0, including those of Bartlett, Desmond
        & Ferreira (2023) and Sousa et al. (2024), used version 0.1.0 or earlier, whose
        right-child model omitted the parent from the context at n = 2 (in general, the most
        distant symbol of the context); no setting of this version reproduces them (see
        README).

        Args:
            :n (int): The length of the n-tuples to consider
            :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
            :in_eqfile (str): Name of file containing the equations to study. If None, then equations read from out_eqfile
            :out_eqfile (str): Name of file to output the standardised equations to
            :input_delimiter (str): The delimiter used in the input csv file
            :root_context (bool, default=True): Condition nodes near the root on an explicit start-of-tree context (see SymbolCoder). False gives the original representation, in which the root is predicted from the distribution of all first children
            :normalise (bool, default=True): Make every conditional distribution of the back-off models proper: negative or non-finite probabilities are set to zero and the rest divided by their sum over the vocabulary (see NormalisedBackOff). False gives the original model, whose conditional probabilities need not sum to one
            :equations (list of str): Equations (with variables x0, x1, ...) to use as the corpus instead of reading a file. If given, in_eqfile and out_eqfile are ignored.

        Returns:
            KatzPrior: Prior model to find prior of a function given a previous set of equations

        """

        self.n = n
        self.root_context = root_context
        self.normalise = normalise
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
        self.coder = SymbolCoder(self.basis_functions, root_context=root_context)
        none = self.coder.code["None"]
        data_left = []
        data_right = []
        for eq in self.all_eq:
            t = self.coder.process_all_equations(n + 1, [eq], self.maxvar)
            data_left += [t[0]] + [tt[:-1] for tt in t[1:]]
            data_right += [tt for tt in t[1:] if tt[-1] != none]
        if len(data_left) == 0 or len(data_right) == 0:
            raise ValueError(
                "The Katz model needs a corpus with at least one binary operator "
                "(second-child event)"
            )
        self.backoff_left = BackOff(data_left)
        self.backoff_right = BackOff(data_right)
        if normalise:
            # words that can be predicted: leaves (constants and parameters are coded "x")
            # and operators; a right child can also be a second, distinct variable "y"
            ops = [
                o
                for o in list(basis_functions[1]) + list(basis_functions[2])
                if o in self.coder.code
            ]
            vocab_left = [self.coder.code[s] for s in ["x"] + ops]
            vocab_right = vocab_left + [self.coder.code["y"]]
            self.backoff_left = NormalisedBackOff(self.backoff_left, vocab_left)
            self.backoff_right = NormalisedBackOff(self.backoff_right, vocab_right)

    def logprior_tuples(self, t):
        """
        Natural logarithm of the prior of an equation already converted to n-tuples

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :p (float): The natural logarithm of the prior (-inf for zero probability)
        """
        none = self.coder.code["None"]
        tleft = [t[0]] + [tt[:-1] for tt in t[1:]]
        pleft = np.array([self.backoff_left.get_pbo(tt[-1], tt[:-1]) for tt in tleft])

        tright = [tt for tt in t[1:] if tt[-1] != none]
        pright = np.array(
            [
                self.backoff_right.get_pbo(tt[-1], tt[-(self.n + 1) : -1])
                for tt in tright
            ]
        )
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
