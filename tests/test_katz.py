import itertools
import os
import tempfile
import unittest

import esr.generation.duplicate_checker
import numpy as np

from katz.back_off import BackOff
from katz.esr_prior import compute_logprior, get_logconst
from katz.good_turing import GoodTuring
from katz.prior import KatzPrior


class TestKatzPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 2
        basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
            ["+", "-", "*", "/", "pow"],
        ]
        cls.kp_feynman = KatzPrior(
            n, basis_functions, "data/FeynmanEquations.csv", "data/NewFeynman.csv"
        )
        cls.kp_feynman_no_input = KatzPrior(
            n, basis_functions, None, "data/NewFeynman.csv"
        )

        basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
            ["+", "-", "*", "/", "pow"],
        ]
        cls.kp_physics = KatzPrior(
            n,
            basis_functions,
            "data/PhysicsEquations.csv",
            "data/NewPhysics.csv",
            input_delimiter=";",
        )

    def test_logprior_string_equations(self):
        equations = ["x0**2", "sin(x0) + sin(x1)", "sin(sin(x0+x1))"]

        # Feynman prior
        expected_results = [np.float64(-2.3), np.float64(-18.5), np.float64(-17.3)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_feynman.logprior(eq), expected, places=1)

        # Physics prior
        expected_results = [np.float64(-2.6), np.float64(-19.9), np.float64(-17.4)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_physics.logprior(eq), expected, places=1)

    def test_logprior_list_equations(self):
        equations = [
            ["+", "x0", "x0"],
            ["*", "2", "x0"],
            ["+", "x0", "x1"],
            ["+", "sin", "x0", "sin", "x1"],
        ]

        # Feynman prior
        expected_results = [
            np.float64(-9.5),
            np.float64(-4.0),
            np.float64(-5.9),
            np.float64(-18.5),
        ]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_feynman.logprior(eq), expected, places=1)
                self.assertAlmostEqual(
                    self.kp_feynman_no_input.logprior(eq), expected, places=1
                )

        # Physics prior
        expected_results = [
            np.float64(-9.2),
            np.float64(-4.0),
            np.float64(-5.5),
            np.float64(-19.9),
        ]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_physics.logprior(eq), expected, places=1)

    def test_op2str(self):
        expected = {
            "pi": "a",
            "0.5": "a",
            "Symbol": "x",
            "x0": "x",
            "x1": "x",
            "a0": "x",
            "a1": "x",
            "Add": "+",
            "Sub": "-",
            "Mul": "*",
            "Div": "/",
            "Pow": "pow",
            "sin": "sin",
        }
        for op, string in expected.items():
            with self.subTest(op=op):
                self.assertEqual(self.kp_feynman.coder.op2str(op), string)
                self.assertEqual(self.kp_feynman_no_input.coder.op2str(op), string)
                self.assertEqual(self.kp_physics.coder.op2str(op), string)

        # Check unknown operator raises Exception
        bad_ops = ["unknown", "Cosh", "Sinh", "garbage", "y0", "b2"]
        for op in bad_ops:
            with self.assertRaises(ValueError):
                self.kp_feynman.coder.op2str(op)
            with self.assertRaises(ValueError):
                self.kp_feynman_no_input.coder.op2str(op)
            with self.assertRaises(ValueError):
                self.kp_physics.coder.op2str(op)


class TestBackOff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 1
        cls.data_file = "data/romeoandjuliet.txt"
        with open(cls.data_file, "r") as f:
            cls.data = f.readlines()[80:]
        cls.data = [line.strip().split() for line in cls.data]
        cls.data = list(itertools.chain(*cls.data))
        cls.data = [(word,) for word in cls.data]

    def test_bo(self):
        self.bo = BackOff(self.data)
        words_in_play = [
            "Romeo",
            "Montague",
            "Juliet",
            "Capulet",
            "the",
            "a",
            "and",
            "to",
            "of",
            "in",
        ]
        for word in words_in_play:
            with self.subTest(word=word):
                self.assertIn(word, self.bo.words)
                self.assertTrue(self.bo.get_pbo(word, ()) > 0)
        words_not_in_play = [
            "Macbeth",
            "Hamlet",
            "Othello",
            "Computer",
            "Avocado",
            "Pineapple",
        ]
        for word in words_not_in_play:
            with self.subTest(word=word):
                self.assertNotIn(word, self.bo.words)
                self.assertEqual(self.bo.get_pbo(word, ()), 0)


class TestESRPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.comp = 5
        esr.generation.duplicate_checker.main("core_maths", cls.comp)
        cls.dirname = "ESR/esr/function_library/core_maths/"
        cls.n = 2
        cls.basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
            ["+", "-", "*", "/", "pow"],
        ]
        cls.in_eqfile = "data/FeynmanEquations.csv"
        cls.out_eqfile = "data/NewFeynman.csv"
        cls.input_delimiter = ","

    def test_compute_logprior(self):

        # Ensure the directory exists
        os.makedirs(self.dirname, exist_ok=True)

        # Test get_logconst function
        for overwrite in [True, False]:
            get_logconst(self.comp, self.dirname, overwrite=overwrite)

        # Check if the logconst file is created
        logconst_file = os.path.join(
            self.dirname, f"compl_{self.comp}", f"logconst_{self.comp}.txt"
        )
        self.assertTrue(os.path.isfile(logconst_file))

        for use_tree in [False, True]:

            # Test compute_logprior function
            for overwrite in [True, False]:
                compute_logprior(
                    self.comp,
                    self.n,
                    self.basis_functions,
                    self.dirname,
                    self.in_eqfile,
                    self.out_eqfile,
                    overwrite=overwrite,
                    input_delimiter=self.input_delimiter,
                    use_tree=use_tree,
                )

            # Check if equation file is created
            eq_file = os.path.join(
                self.dirname, f"compl_{self.comp}", f"all_equations_{self.comp}.txt"
            )
            self.assertTrue(os.path.isfile(eq_file))

            # Check if the logprior file is created
            logprior_file = os.path.join(
                self.dirname,
                f"compl_{self.comp}",
                f"katz_logprior_{self.n}_{self.comp}.txt",
            )
            self.assertTrue(os.path.isfile(logprior_file))

            # Check if the codelen file is created
            codelen_file = os.path.join(
                self.dirname,
                f"compl_{self.comp}",
                f"katz_codelen_{self.n}_{self.comp}.txt",
            )
            self.assertTrue(os.path.isfile(codelen_file))

            # Load equations and logprior values
            logprior = np.loadtxt(logprior_file)
            codelen = np.loadtxt(codelen_file)
            with open(eq_file, "r") as f:
                equations = f.readlines()
            equations = [eq.strip() for eq in equations]
            self.assertEqual(len(equations), len(logprior), len(codelen))

            # Check that equations with same string have same katz prior
            if not use_tree:
                unique_strings = {}
                for index, string in enumerate(equations):
                    if string not in unique_strings:
                        unique_strings[string] = []
                    unique_strings[string].append(index)
                for s, indices in unique_strings.items():
                    if np.isfinite(logprior[indices[0]]):
                        self.assertTrue(
                            np.all(logprior[indices] == logprior[indices[0]])
                        )
                    else:
                        print(s, logprior[indices])
                        self.assertTrue(np.all(~np.isfinite(logprior[indices])))

            # Check that at least some values are finite
            self.assertTrue(np.any(np.isfinite(logprior)))


class TestGoodTuringSingleBucket(unittest.TestCase):
    """Tests for GoodTuring edge cases that previously caused NaN.

    When every token in the corpus is distinct (each appears exactly once),
    Nr has a single frequency bucket.  Before the fix this left Zr[-1] = 0,
    causing log(0) = -inf and linregress returning NaN for all outputs.
    """

    def _all_distinct_corpus(self):
        """Return a corpus where every item appears exactly once."""
        return [(i,) for i in range(5)]

    def test_zr_last_element_finite(self):
        """Zr[-1] must be positive and finite for a single-bucket Nr."""
        gt = GoodTuring(self._all_distinct_corpus())
        self.assertTrue(
            np.all(np.isfinite(gt.Zr)), f"Zr contains non-finite values: {gt.Zr}"
        )
        self.assertTrue(np.all(gt.Zr > 0), f"Zr contains non-positive values: {gt.Zr}")

    def test_slope_intercept_finite(self):
        """slope and intercept must be finite (not NaN) for a single-bucket Nr."""
        gt = GoodTuring(self._all_distinct_corpus())
        self.assertTrue(np.isfinite(gt.slope), f"slope is not finite: {gt.slope}")
        self.assertTrue(
            np.isfinite(gt.intercept), f"intercept is not finite: {gt.intercept}"
        )

    def test_get_S_finite(self):
        """get_S must return a positive, finite value for r >= 1."""
        gt = GoodTuring(self._all_distinct_corpus())
        for r in [1, 2, 3]:
            with self.subTest(r=r):
                s = gt.get_S(r)
                self.assertTrue(
                    np.isfinite(s) and s > 0,
                    f"get_S({r}) = {s} is not positive and finite",
                )

    def test_expected_count_finite(self):
        """expected_count must return a positive, finite value for seen words."""
        gt = GoodTuring(self._all_distinct_corpus())
        for word in self._all_distinct_corpus():
            with self.subTest(word=word):
                ec = gt.expected_count(word)
                self.assertTrue(
                    np.isfinite(ec) and ec > 0,
                    f"expected_count({word}) = {ec} is not positive and finite",
                )

    def test_single_element_corpus(self):
        """A corpus with exactly one element must not raise and must be finite."""
        gt = GoodTuring([(42,)])
        self.assertTrue(np.all(np.isfinite(gt.Zr)))
        self.assertTrue(np.isfinite(gt.slope))
        self.assertTrue(np.isfinite(gt.intercept))
        self.assertTrue(np.isfinite(gt.get_S(1)))


class TestGoodTuringMultiBucket(unittest.TestCase):
    """Sanity checks for GoodTuring with multiple frequency buckets (normal path)."""

    @classmethod
    def setUpClass(cls):
        # Two distinct frequency levels: (1,) appears twice, (2,) appears once.
        cls.corpus = [(1,), (1,), (2,)]
        cls.gt = GoodTuring(cls.corpus)

    def test_zr_finite(self):
        self.assertTrue(np.all(np.isfinite(self.gt.Zr)))

    def test_slope_intercept_finite(self):
        self.assertTrue(np.isfinite(self.gt.slope))
        self.assertTrue(np.isfinite(self.gt.intercept))

    def test_actual_count(self):
        self.assertEqual(self.gt.actual_count((1,)), 2)
        self.assertEqual(self.gt.actual_count((2,)), 1)
        self.assertEqual(self.gt.actual_count((99,)), 0)

    def test_expected_count_positive(self):
        for word in [(1,), (2,)]:
            with self.subTest(word=word):
                self.assertGreater(self.gt.expected_count(word), 0)


class TestSparseKatzPrior(unittest.TestCase):
    """Tests for KatzPrior with a tiny single-equation corpus.

    This is the case that triggered NaN: the equation being evaluated is
    present in the corpus but get_pbo returned NaN because GoodTuring
    failed on single-bucket Nr arrays.

    The right back-off model is now trained on full (parent, left_sibling,
    right_child) tuples so that the query context correctly includes the
    parent node, matching the fix to logprior.
    """

    # Minimal SimpleEquations-style CSV (semicolon-delimited)
    _SIMPLE_EQ_HEADER = (
        "Filename;Number;Output;Formula;# variables;"
        "v1_name;v1_low;v1_high;v2_name;v2_low;v2_high;"
        "v3_name;v3_low;v3_high;v4_name;v4_low;v4_high;"
        "v5_name;v5_low;v5_high;v6_name;v6_low;v6_high;"
        "v7_name;v7_low;v7_high;v8_name;v8_low;v8_high;"
        "v9_name;v9_low;v9_high;v10_name;v10_low;v10_high"
    )
    _SIMPLE_EQ_ROW = "Eq0;2;f;sin(x) + sin(x - y);2;x;1;3;y;1;3;;;;;;;;;;;;;;;;;;;;;;;"

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._in_file = os.path.join(cls._tmpdir, "SimpleEquations.csv")
        cls._out_file = os.path.join(cls._tmpdir, "NewSimple.csv")

        with open(cls._in_file, "w") as f:
            f.write(cls._SIMPLE_EQ_HEADER + "\n")
            f.write(cls._SIMPLE_EQ_ROW + "\n")

        cls.basis_functions = [["x"], ["sin"], ["+", "-"]]
        cls.n = 2
        cls.kp = KatzPrior(
            cls.n,
            cls.basis_functions,
            cls._in_file,
            cls._out_file,
            input_delimiter=";",
        )
        cls.eq = "sin(x0) + sin(x0 - x1)"

    def test_logprior_is_finite(self):
        """logprior must return a finite value, not NaN, for an equation in the corpus."""
        p = self.kp.logprior(self.eq)
        self.assertTrue(np.isfinite(p), f"logprior returned non-finite value: {p}")

    def test_logprior_is_negative(self):
        """Log-probabilities must be non-positive."""
        p = self.kp.logprior(self.eq)
        self.assertLessEqual(p, 0.0)

    def test_logprior_expected_value(self):
        """logprior must match the expected value for the minimal corpus.

        With n=2 and a single training equation the right back-off corpus
        contains two 3-grams, each appearing exactly once.  The Good-Turing
        fallback (d=1, no discounting) means both right-child probabilities
        equal 1, contributing 0 to the log-prior.  The value is therefore
        determined entirely by the left probabilities, which are unchanged
        by this fix.
        """
        p = self.kp.logprior(self.eq)
        self.assertAlmostEqual(p, -3.0, places=1)

    def test_out_file_created(self):
        """standardise_file must write the output CSV."""
        self.assertTrue(os.path.isfile(self._out_file))

    def test_out_file_contains_standardised_equation(self):
        """The output CSV must contain the standardised equation."""
        import pandas as pd

        df = pd.read_csv(self._out_file)
        self.assertIn("sin(x0)+sin(x0-x1)", df["New Formula"].tolist())


if __name__ == "__main__":
    unittest.main()
