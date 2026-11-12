import itertools
import os
import tempfile
import unittest

import esr
import esr.generation.duplicate_checker
import numpy as np

from katz.back_off import BackOff
from katz import esr_prior
from katz.esr_prior import compute_logprior, default_label, get_logconst
from katz.good_turing import GoodTuring
from katz.prior import KatzPrior
from katz.process_equations import (
    count_variables,
    split_by_punctuation,
    standardise_file,
)

BASIS = [
    ["a", "x"],
    ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
    ["+", "-", "*", "/", "pow"],
]


def _require_sympy_114(test):
    """The corpus is parsed with sympy, and sympy < 1.14 writes e.g. sqrt(2*x0**2 + 4*x1)
    as a*sqrt(...) rather than sqrt(...), which changes the trees of 6 of the 161 physics
    and 2 of the 100 Feynman equations and so the prior. Reference values are for sympy
    >= 1.14 (the version required in setup.py)."""
    import sympy

    if tuple(int(v) for v in sympy.__version__.split(".")[:2]) < (1, 14):
        test.skipTest(f"reference values need sympy >= 1.14, not {sympy.__version__}")


class TestKatzPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 2
        basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
            ["+", "-", "*", "/", "pow"],
        ]
        # Reference values below are for the model of Bartlett et al. (2023)
        cls.kp_feynman = KatzPrior(
            n,
            basis_functions,
            "data/FeynmanEquations.csv",
            "data/NewFeynman.csv",
            root_context=False,
            normalise=False,
        )
        cls.kp_feynman_no_input = KatzPrior(
            n, basis_functions, None, "data/NewFeynman.csv", root_context=False,
            normalise=False,
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
            root_context=False,
            normalise=False,
        )

    def test_logprior_string_equations(self):
        _require_sympy_114(self)
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
        _require_sympy_114(self)
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

    def test_labels(self):
        kp = KatzPrior(2, BASIS, equations=["x0 + x1"])
        import sympy

        x = {f"x{i}": sympy.Symbol(f"x{i}", real=True) for i in range(2)}
        a = {f"a{i}": sympy.Symbol(f"a{i}", real=True) for i in range(2)}
        labels = kp.coder.equation2labels("x0 - x1", {**x, **a})
        self.assertEqual(labels, ["-", "x0", "x1"])
        labels = kp.coder.equation2labels("sqrt(x0)/x1", {**x, **a})
        self.assertEqual(labels, ["/", "sqrt", "x0", "x1"])
        labels = kp.coder.equation2labels("pow(Abs(a0), x0)", {**x, **a})
        self.assertEqual(labels, ["pow", "a0", "x0"])
        # lower-case abs is parsed to Abs by sympy and must be stripped too
        labels = kp.coder.equation2labels("pow(abs(a0), x0)", {**x, **a})
        self.assertEqual(labels, ["pow", "a0", "x0"])
        eq = "x0*abs(exp(x0) - 1)**x1"
        self.assertEqual(
            kp.coder.labels2ntuples(3, kp.coder.equation2labels(eq, {**x, **a})),
            kp.coder.labels2ntuples(3, kp.coder.equation2labels(eq.replace("abs", ""), {**x, **a})),
        )


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
        # Location of the installed ESR package's function library (equal to
        # ESR/esr/function_library/core_maths/ when ESR is cloned into the repo root)
        cls.dirname = os.path.join(
            os.path.dirname(esr.__file__), "function_library", "core_maths", ""
        )
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

        for use_tree, (label, prior_kwargs) in itertools.product(
            [False, True],
            [("katz_noroot_unnorm", {"root_context": False, "normalise": False}),
             ("katz", None)],
        ):

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
                    prior_kwargs=prior_kwargs,
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
                f"{label}_logprior_{self.n}_{self.comp}"
                + ("_tree" if use_tree else "")
                + ".txt",
            )
            self.assertTrue(os.path.isfile(logprior_file))

            # Check if the codelen file is created
            codelen_file = os.path.join(
                self.dirname,
                f"compl_{self.comp}",
                f"{label}_codelen_{self.n}_{self.comp}"
                + ("_tree" if use_tree else "")
                + ".txt",
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

    def test_manifest_and_labels(self):
        """Output is reused only if its manifest matches the requested configuration"""
        import json

        get_logconst(self.comp, self.dirname, overwrite=False)
        self.assertEqual(default_label({}), "katz")
        self.assertEqual(default_label({"root_context": False}), "katz_noroot")
        self.assertEqual(default_label({"normalise": False}), "katz_unnorm")
        self.assertEqual(
            default_label({"root_context": False, "normalise": False}), "katz_noroot_unnorm"
        )
        base = os.path.join(self.dirname, f"compl_{self.comp}", "")
        prior_file = base + f"clash_logprior_{self.n}_{self.comp}.txt"
        manifest_file = base + f"clash_prior_{self.n}_{self.comp}.json"
        results = []
        for rc in [False, True, True]:
            compute_logprior(
                self.comp,
                self.n,
                self.basis_functions,
                self.dirname,
                self.in_eqfile,
                self.out_eqfile,
                overwrite=False,
                prior_kwargs={"root_context": rc},
                label="clash",
            )
            with open(manifest_file) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["config"]["root_context"], rc)
            self.assertTrue(manifest["config"]["normalise"])
            self.assertEqual(manifest["n_failed"], 0)
            results.append((np.loadtxt(prior_file), os.path.getmtime(prior_file)))
        # A different configuration under the same label is recomputed...
        self.assertFalse(np.allclose(results[0][0], results[1][0], equal_nan=True))
        # ... and the same configuration is reused
        self.assertEqual(results[1][1], results[2][1])
        for f in os.listdir(base):
            if f.startswith("clash_"):
                os.remove(base + f)


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
            root_context=False,
            normalise=False,
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


class TestKatzPriorOptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.basis_functions = BASIS
        cls.kp = KatzPrior(2, cls.basis_functions, None, "data/NewPhysics.csv")

    def test_defaults(self):
        """Root context and normalisation are on by default; switching both off gives the
        model of Bartlett et al. (2023)"""
        from katz.back_off import NormalisedBackOff

        self.assertTrue(self.kp.root_context and self.kp.normalise)
        self.assertIsInstance(self.kp.backoff_left, NormalisedBackOff)
        orig = KatzPrior(
            2, self.basis_functions, equations=self.kp.all_eq, root_context=False,
            normalise=False,
        )
        self.assertIsInstance(orig.backoff_left, BackOff)
        for eq in ["x0**2", "sin(x0) + x1"]:
            with self.subTest(eq=eq):
                self.assertTrue(np.isfinite(self.kp.logprior(eq)))
                self.assertNotEqual(self.kp.logprior(eq), orig.logprior(eq))

    def test_equations_argument(self):
        """Building from a list of equations matches building from the file"""
        kp = KatzPrior(2, self.basis_functions, equations=self.kp.all_eq)
        for eq in ["x0**2", "sin(x0)*exp(x1)", "x0 + 3*x1"]:
            self.assertEqual(kp.logprior(eq), self.kp.logprior(eq))

    def test_root_context(self):
        n = 2
        kp = KatzPrior(n, self.basis_functions, equations=self.kp.all_eq[:30])
        none = kp.coder.code["None"]
        t = kp.coder.process_all_equations(n + 1, ["sin(sin(x0 + x1))"], 2)
        self.assertEqual(t[0], (none, none, kp.coder.code["sin"]))
        self.assertTrue(all(len(tt) == n + 2 for tt in t[1:]))
        self.assertTrue(np.isfinite(kp.logprior("sin(sin(x0 + x1))")))
        kp = KatzPrior(n, self.basis_functions, equations=self.kp.all_eq[:30],
                       root_context=False)
        t = kp.coder.process_all_equations(n + 1, ["sin(sin(x0 + x1))"], 2)
        self.assertEqual(t[0], (kp.coder.code["sin"],))

    def test_single_leaf(self):
        """A single-leaf function contributes exactly one (root) event"""
        for rc, norm in itertools.product([False, True], [False, True]):
            kp = KatzPrior(
                2,
                self.basis_functions,
                equations=self.kp.all_eq[:40],
                root_context=rc,
                normalise=norm,
            )
            for eq in ["x0", "a0", ["x0"], ["a0"]]:
                with self.subTest(root_context=rc, normalise=norm, eq=eq):
                    t = kp.coder.process_all_equations(3, [eq], kp.maxvar)
                    self.assertEqual(len(t), 1)
                    root = t[0]
                    p = kp.backoff_left.get_pbo(root[-1], root[:-1])
                    self.assertAlmostEqual(kp.logprior(eq), np.log(p))


class TestEdgeCases(unittest.TestCase):
    """Edge cases found by an independent audit of the code"""

    basis_functions = BASIS

    def test_bare_leaf_and_maxvar(self):
        self.assertEqual(split_by_punctuation("x0"), ["x0"])
        self.assertEqual(split_by_punctuation("a0"), ["a0"])
        self.assertEqual(split_by_punctuation("x0+x1"), ["x0", "+", "x1"])
        self.assertEqual(count_variables(["x10 + 1"]), 11)
        self.assertEqual(count_variables(["sin(x0)", "a0", "exp(x2)"]), 3)
        self.assertEqual(count_variables(["2*a0"]), 1)
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "eq.csv")
            with open(fname, "w") as f:
                f.write("New Formula\nx10 + 1\nx0\n")
            eqs, maxvar = standardise_file(None, fname, ",")
            self.assertEqual((eqs, maxvar), (["x10 + 1", "x0"], 11))
        for fname, expected in [("data/NewPhysics.csv", 11), ("data/NewFeynman.csv", 9)]:
            with self.subTest(fname=fname):
                self.assertEqual(standardise_file(None, fname, ",")[1], expected)

    def test_basis_order_is_reproducible(self):
        kp = KatzPrior(2, self.basis_functions, equations=["x0 + x1"])
        self.assertEqual(kp.basis_functions[0], ["a", "x", "x0", "x1"])

    def test_symbols_beyond_corpus_range(self):
        # x_i and a_i outside the corpus's variable range (x0..x10) are real symbols and leaves
        import pandas as pd

        eqs = list(pd.read_csv("data/NewPhysics.csv")["New Formula"])
        kp = KatzPrior(2, self.basis_functions, equations=eqs)
        self.assertEqual(kp.maxvar, 11)
        for e0, e1 in [("sqrt(x0**2)", "sqrt(x12**2)"), ("sqrt(a0**2)", "sqrt(a15**2)"),
                       ("x0*a2 + a5", "x12*a2 + a15")]:
            with self.subTest(eq=e1):
                self.assertTrue(np.isfinite(kp.logprior(e0)))
                self.assertAlmostEqual(kp.logprior(e1), kp.logprior(e0), places=12)

    def test_corpus_parse_is_reproducible(self):
        # the parsed corpus defines the prior: fail if sympy changes how it writes it
        import json

        import pandas as pd
        import sympy

        _require_sympy_114(self)
        with open(os.path.join(os.path.dirname(__file__), "corpus_labels_sympy1.14.json")) as f:
            ref = json.load(f)
        for name in ("NewPhysics", "NewFeynman"):
            eqs = list(pd.read_csv(f"data/{name}.csv")["New Formula"])
            kp = KatzPrior(2, ref["basis_functions"], equations=eqs)
            locs = {f"{c}{i}": sympy.Symbol(f"{c}{i}", real=True)
                    for c in "xa" for i in range(kp.maxvar)}
            for i, eq in enumerate(eqs):
                with self.subTest(corpus=name, eq=eq):
                    self.assertEqual(kp.coder.equation2labels(eq, locs), ref["labels"][name][i])

    def test_default_model_pickles(self):
        # the default (normalised) model survives pickling and copying, e.g. for
        # multiprocessing
        import copy
        import pickle

        kp = KatzPrior(2, self.basis_functions, equations=["x0 + x1", "sin(x0)*x1"])
        for clone in (pickle.loads(pickle.dumps(kp)), copy.deepcopy(kp)):
            for eq in ["x0 + x1", "sin(x0)"]:
                self.assertEqual(clone.logprior(eq), kp.logprior(eq))

    def test_single_leaf_training_equation(self):
        # a single-leaf corpus equation contributes one root event to the training data
        eqs = ["x0", "x0 + x1", "sin(x0)*x1", "x0*x1 + x1"]
        kp = KatzPrior(2, self.basis_functions, equations=eqs, root_context=False,
                       normalise=False)
        none = kp.coder.code["None"]
        left, right = [], []
        for eq in eqs:
            t = kp.coder.process_all_equations(3, [eq], kp.maxvar)
            left += [t[0]] + [tt[:-1] for tt in t[1:]]
            right += [tt for tt in t[1:] if tt[-1] != none]
        self.assertEqual(sum(len(tt) == 1 for tt in left), len(eqs))  # one root each
        ref = BackOff(left)
        double = BackOff(left + [left[0]])  # the root event of "x0" counted twice
        words = [kp.coder.code[w] for w in ["x", "+", "*", "sin"]]
        for w in words:
            self.assertEqual(kp.backoff_left.get_pbo(w, ()), ref.get_pbo(w, ()))
        self.assertNotEqual(kp.backoff_left.get_pbo(words[1], ()),
                            double.get_pbo(words[1], ()))

    def test_corpus_without_binary_operators(self):
        eqs = ["sin(x0)", "exp(sqrt(x0))", "x0"]
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=eqs)

    def test_normalised_katz(self):
        # the original Katz conditionals need not sum to one; the normalised ones do
        import pandas as pd

        eqs = list(pd.read_csv("data/NewPhysics.csv")["New Formula"])
        raw = KatzPrior(3, self.basis_functions, equations=eqs, normalise=False)
        norm = KatzPrior(3, self.basis_functions, equations=eqs)
        none = norm.coder.code["None"]
        contexts = {
            tuple(tt[-(norm.n + 1):-1])
            for eq in eqs
            for tt in norm.coder.process_all_equations(norm.n + 1, [eq], norm.maxvar)[1:]
            if tt[-1] != none
        }
        m, r = norm.backoff_right, raw.backoff_right
        raw_sums = [sum(r.get_pbo(w, h) for w in m.words) for h in contexts]
        self.assertLess(min(raw_sums), 0.99)  # the original model loses mass
        for h in contexts:
            p = [m.get_pbo(w, h) for w in m.words]
            self.assertTrue(min(p) >= 0)
            self.assertAlmostEqual(sum(p), 1.0, places=12)
        # BackOff itself is unchanged: normalise=False is the original model (the sums are
        # taken in a different order, so compare to rounding)
        self.assertAlmostEqual(
            raw.logprior("sin(x0) + x1"),
            np.log(raw.backoff_left.get_pbo(*self._root(raw, "sin(x0) + x1")))
            + self._rest(raw, "sin(x0) + x1"),
            places=10,
        )

    @staticmethod
    def _root(kp, eq):
        t = kp.coder.process_all_equations(kp.n + 1, [eq], kp.maxvar)
        return t[0][-1], t[0][:-1]

    @staticmethod
    def _rest(kp, eq):
        t = kp.coder.process_all_equations(kp.n + 1, [eq], kp.maxvar)
        none = kp.coder.code["None"]
        p = [kp.backoff_left.get_pbo(tt[-2], tt[:-2]) for tt in t[1:]]
        p += [kp.backoff_right.get_pbo(tt[-1], tt[-(kp.n + 1):-1]) for tt in t[1:]
              if tt[-1] != none]
        return float(np.sum(np.log(p)))


class TestPriorCache(unittest.TestCase):
    """compute_logprior reuses saved scores only if every input and output is unchanged"""

    corpus = ["x0*sin(x0) + x0", "sin(x0)**2", "x0/(x0 + 1)", "sqrt(x0) + sin(x0)"]
    basis = [["a", "x"], ["sqrt", "sin", "square", "inv"], ["+", "-", "*", "/", "pow"]]

    def _library(self, td, fcn, tree):
        d = os.path.join(td, "compl_1")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "all_equations_1.txt"), "w") as f:
            f.write(fcn + "\n")
        with open(os.path.join(d, "trees_1.txt"), "w") as f:
            f.write(repr(tree) + "\n")
        esr_prior.get_logconst(1, td)

    def _run(self, td, **kw):
        esr_prior.compute_logprior(1, 2, self.basis, td, None, None,
                                    prior_kwargs={"equations": self.corpus}, **kw)
        base = os.path.join(td, "compl_1", "katz_")
        return (float(np.loadtxt(base + "logprior_2_1.txt")),
                os.stat(base + "prior_2_1.json").st_mtime_ns)

    def test_reuse_and_invalidation(self):
        kp = KatzPrior(2, self.basis, equations=self.corpus)
        with tempfile.TemporaryDirectory() as td:
            self._library(td, "x0", ["x0"])
            v0, t0 = self._run(td)
            self.assertAlmostEqual(v0, -kp.logprior("x0"), places=10)
            self.assertEqual(self._run(td), (v0, t0))  # unchanged inputs: reused
            # new candidate library (the audit's example)
            self._library(td, "sin(x0)", ["sin", "x0"])
            v1, t1 = self._run(td)
            self.assertAlmostEqual(v1, -kp.logprior("sin(x0)"), places=10)
            # changed constant penalties
            lc = os.path.join(td, "compl_1", "logconst_1.txt")
            np.savetxt(lc, [1.5])
            v2, t2 = self._run(td)
            self.assertNotEqual(t2, t1)
            code = float(np.loadtxt(os.path.join(td, "compl_1", "katz_codelen_2_1.txt")))
            self.assertAlmostEqual(code, v1 + 1.5, places=10)
            # tampered output
            np.savetxt(os.path.join(td, "compl_1", "katz_logprior_2_1.txt"), [0.0])
            v3, t3 = self._run(td)
            self.assertAlmostEqual(v3, v1, places=10)
            self.assertNotEqual(t3, t2)
            # logconst follows the tree file
            self._library(td, "sqrt(x0)", ["sqrt", "x0"])
            self.assertEqual(float(np.loadtxt(lc)), 0.0)


    def test_esr_parser_in_code_hash(self):
        # the scores depend on ESR's parser, so a change to it must invalidate saved output
        from unittest import mock

        import esr.generation.generator as gen

        with tempfile.TemporaryDirectory() as td:
            copy_ = os.path.join(td, "generator.py")
            with open(gen.__file__) as f, open(copy_, "w") as g:
                g.write(f.read() + "\n# changed\n")
            h0 = esr_prior._code_sha256()
            with mock.patch.object(gen, "__file__", copy_):
                self.assertNotEqual(esr_prior._code_sha256(), h0)
            self._library(td, "x0", ["x0"])
            v0, t0 = self._run(td)
            with mock.patch.object(esr_prior, "_code_sha256", return_value="other"):
                v1, t1 = self._run(td)
            self.assertNotEqual(t1, t0)
            self.assertEqual(v1, v0)

    def test_nan_scores_are_counted(self):
        # the unnormalised model can return NaN (zero denominator in a back-off weight);
        # such functions are NaN in the output and counted as failed
        import json

        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "compl_1")
            os.makedirs(d)
            with open(os.path.join(d, "all_equations_1.txt"), "w") as f:
                f.write("x0 + x0\nx0 + x1\n")
            with open(os.path.join(d, "trees_1.txt"), "w") as f:
                f.write("['+', 'x0', 'x0']\n['+', 'x0', 'x1']\n")
            esr_prior.get_logconst(1, td)
            esr_prior.compute_logprior(
                1, 2, self.basis, td, None, None, label="nan",
                prior_kwargs={"equations": ["x0 + x1"], "normalise": False},
            )
            vals = np.loadtxt(os.path.join(d, "nan_logprior_2_1.txt"))
            with open(os.path.join(d, "nan_prior_2_1.json")) as f:
                manifest = json.load(f)
            self.assertTrue(np.isnan(vals[0]) and np.isfinite(vals[1]))
            self.assertEqual(manifest["n_failed"], 1)


if __name__ == "__main__":
    unittest.main()
