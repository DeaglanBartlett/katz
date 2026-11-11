import itertools
import os
import tempfile
import unittest

import numpy as np

try:  # ESR is needed only to compare the built-in parser with it and for TestESRPrior
    import esr
    import esr.generation.duplicate_checker
    import esr.generation.generator

    HAVE_ESR = True
except ImportError:
    HAVE_ESR = False

try:  # katz.esr_prior needs mpi4py
    from katz import esr_prior
    from katz.esr_prior import compute_logprior, default_label, get_logconst

    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

from katz import tree_parser
from katz.back_off import BackOff
from katz.good_turing import GoodTuring
from katz.kneser_ney import KneserNey, chen_goodman_discounts
from katz.prior import KatzPrior
from katz.process_equations import (
    SymbolCoder,
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
        # Reference values below are for the Katz model of version 0.2.1
        cls.kp_feynman = KatzPrior(
            n,
            basis_functions,
            "data/FeynmanEquations.csv",
            "data/NewFeynman.csv",
            root_context=False,
            normalise=False,
            smoothing="katz",
        )
        cls.kp_feynman_no_input = KatzPrior(
            n,
            basis_functions,
            None,
            "data/NewFeynman.csv",
            root_context=False,
            normalise=False,
            smoothing="katz",
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
            smoothing="katz",
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
            kp.coder.labels2ntuples(
                3, kp.coder.equation2labels(eq.replace("abs", ""), {**x, **a})
            ),
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


@unittest.skipUnless(HAVE_ESR and HAVE_MPI, "needs ESR and mpi4py")
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
            [
                (
                    "katz_noroot_unnorm",
                    {"smoothing": "katz", "root_context": False, "normalise": False},
                ),
                ("kn", None),
            ],
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
        self.assertEqual(default_label({}), "kn")
        self.assertEqual(
            default_label({"normalise": False}), "kn"
        )  # KN is always normalised
        self.assertEqual(default_label({"code_constants": True}), "kn_const")
        self.assertEqual(default_label({"smoothing": "katz"}), "katz")
        self.assertEqual(
            default_label({"smoothing": "katz", "root_context": False}), "katz_noroot"
        )
        self.assertEqual(
            default_label(
                {"smoothing": "katz", "root_context": False, "normalise": False}
            ),
            "katz_noroot_unnorm",
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
            self.assertEqual(manifest["config"]["smoothing_kwargs"]["discounts"], 0.8)
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
            smoothing="katz",
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


class TestKneserNey(unittest.TestCase):

    def test_short_history(self):
        # a history shorter than order - 1 words is used in full, and an unseen extra
        # word in front of it backs off to the same distribution
        m = KneserNey(
            [("a", "b", "x")] * 10 + [("c", "b", "y")] * 10, order=4, discounts=0.5
        )
        p = (
            9.5 / 10 + (0.5 / 10) * 0.5
        )  # (c(abx) - D)/c(ab.) + D N1+(ab.)/c(ab.) P(x|b)
        self.assertAlmostEqual(m.prob("x", ("a", "b")), p, places=12)
        self.assertAlmostEqual(m.prob("x", ("q", "a", "b")), p, places=12)
        self.assertAlmostEqual(m.prob("x", ("b",)), 0.5, places=12)

    def test_worked_example(self):
        """Two bigrams seen once each: an unseen bigram keeps a share of the mass"""
        kn = KneserNey([("sin", "x"), ("cos", "y")], discounts=0.75)
        self.assertAlmostEqual(kn.prob("x", ("sin",)), 0.625)
        self.assertAlmostEqual(kn.prob("y", ("sin",)), 0.375)

    def test_hand_computed_trigram(self):
        """Compare against a hand calculation with a fixed discount D = 0.5"""
        corpus = [
            ("a", "b", "c"),
            ("a", "b", "c"),
            ("d", "b", "c"),
            ("a", "b", "e"),
            ("b", "e"),  # shorter event: raw count at order 2
        ]
        kn = KneserNey(corpus, order=3, vocab=["z"], discounts=0.5)
        # Order 1: continuation counts c=1, e=1; gamma = 0.5; uniform over {c, e, z}
        p1 = {"c": 0.25 + 0.5 / 3, "e": 0.25 + 0.5 / 3, "z": 0.5 / 3}
        for w, p in p1.items():
            self.assertAlmostEqual(kn.prob(w, ()), p)
        # Order 2, context b: adjusted counts (b,c)=2 [types a,d], (b,e)=2 [raw 1 + type a]
        p2 = {w: (2 - 0.5) / 4 * (w != "z") + 0.25 * p1[w] for w in p1}
        for w, p in p2.items():
            self.assertAlmostEqual(kn.prob(w, ("b",)), p)
        # Order 3, context (a, b): raw counts c=2, e=1
        p3 = {"c": 1.5 / 3 + p2["c"] / 3, "e": 0.5 / 3 + p2["e"] / 3, "z": p2["z"] / 3}
        for w, p in p3.items():
            self.assertAlmostEqual(kn.prob(w, ("a", "b")), p)
        self.assertAlmostEqual(kn.prob("c", ("d", "b")), 0.5 + 0.5 * p2["c"])
        # Unseen context backs off to the longest seen suffix
        self.assertAlmostEqual(kn.prob("c", ("q", "b")), p2["c"])
        # Contexts longer than order - 1 are truncated
        self.assertAlmostEqual(kn.prob("c", ("q", "a", "b")), p3["c"])
        self.assertEqual(kn.get_pbo("c", ("a", "b")), kn.prob("c", ("a", "b")))

    def test_normalisation_and_positivity(self):
        rng = np.random.default_rng(42)
        V = list("abcdef")
        corpus = [tuple(rng.choice(V, size=rng.integers(1, 5))) for _ in range(200)]
        for discounts in ["mkn", "kn", 0.3, [0.2, 0.5, 0.9, 0.7]]:
            kn = KneserNey(corpus, order=4, vocab=V + ["g"], discounts=discounts)
            for L in range(4):
                for h in itertools.product(V + ["g"], repeat=L):
                    p = np.array([kn.prob(w, h) for w in kn.vocab])
                    with self.subTest(discounts=discounts, h=h):
                        self.assertAlmostEqual(p.sum(), 1.0, places=12)
                        self.assertTrue(np.all(p > 0))
        self.assertEqual(kn.prob("unknown", ()), 0.0)

    def test_continuation_counts(self):
        """A frequent word seen in a single context gets less lower-order mass than a
        rarer word seen in many contexts (and vice versa for raw unigram frequency)"""
        corpus = [("s", "f")] * 10 + [(c, "g") for c in "abcde"]
        kn = KneserNey(corpus, order=2, discounts=0.5)
        self.assertLess(kn.prob("f", ()), kn.prob("g", ()))
        self.assertLess(kn.prob("f", ("q",)), kn.prob("g", ("q",)))
        self.assertGreater(kn.prob("f", ("s",)), 0.9)

    def test_chen_goodman_discounts(self):
        D = chen_goodman_discounts({1: 10, 2: 5, 3: 3, 4: 2})
        np.testing.assert_allclose(D, [0.5, 1.1, 3 - 4 * 0.5 * 2 / 3])
        # Undefined estimates fall back to 0.5 and estimates are clipped to [1e-3, r]
        np.testing.assert_allclose(chen_goodman_discounts({3: 1}), [0.5, 0.5, 0.5])
        D = chen_goodman_discounts({1: 1, 2: 1, 3: 100, 4: 1})
        self.assertTrue(np.all(D >= 1e-3) and np.all(D <= [1, 2, 3]))

    def test_discount_options(self):
        corpus = [("a", "b"), ("a", "c"), ("b", "c"), ("c",)]
        kn = KneserNey(corpus, discounts=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_allclose(kn.discounts[1:], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        kn = KneserNey(corpus, discounts=0.9, discount_scale=2.0)
        np.testing.assert_allclose(kn.discounts[1:], [[1.0, 1.8, 1.8]] * 2)
        with self.assertRaises(ValueError):
            KneserNey(corpus, discounts="bad")
        with self.assertRaises(ValueError):
            KneserNey([])


class TestKatzPriorOptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.basis_functions = BASIS
        cls.kp = KatzPrior(2, cls.basis_functions, None, "data/NewPhysics.csv")
        cls.kp_kn = cls.kp

    def test_defaults(self):
        """Kneser-Ney with root context is the default; Katz is normalised by default, and
        smoothing='katz', root_context=False, normalise=False gives the model of version
        0.2.1"""
        from katz.back_off import NormalisedBackOff

        self.assertEqual(self.kp.smoothing, "kn")
        self.assertTrue(self.kp.root_context)
        katz = KatzPrior(
            2, self.basis_functions, equations=self.kp.all_eq, smoothing="katz"
        )
        self.assertIsInstance(katz.backoff_left, NormalisedBackOff)
        orig = KatzPrior(
            2,
            self.basis_functions,
            equations=self.kp.all_eq,
            smoothing="katz",
            root_context=False,
            normalise=False,
        )
        self.assertIsInstance(orig.backoff_left, BackOff)
        for eq in ["x0**2", "sin(x0) + x1"]:
            with self.subTest(eq=eq):
                self.assertTrue(np.isfinite(katz.logprior(eq)))
                self.assertNotEqual(katz.logprior(eq), orig.logprior(eq))
        with self.assertRaises(
            ValueError
        ):  # Katz takes normalise=, not smoothing_kwargs
            KatzPrior(
                2,
                self.basis_functions,
                equations=["x0 + x1"],
                smoothing="katz",
                smoothing_kwargs={"normalise": True},
            )

    def test_equations_argument(self):
        """Building from a list of equations matches building from the file"""
        kp = KatzPrior(2, self.basis_functions, equations=self.kp.all_eq)
        for eq in ["x0**2", "sin(x0)*exp(x1)", "x0 + 3*x1"]:
            self.assertEqual(kp.logprior(eq), self.kp.logprior(eq))
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=["x0"], smoothing="bad")
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=["x0"], right_backoff="bad")

    def test_root_context(self):
        n = 2
        kp = KatzPrior(n, self.basis_functions, equations=self.kp.all_eq[:30])
        none = kp.coder.code["None"]
        t = kp.coder.process_all_equations(n + 1, ["sin(sin(x0 + x1))"], 2)
        self.assertEqual(t[0], (none, none, kp.coder.code["sin"]))
        self.assertTrue(all(len(tt) == n + 2 for tt in t[1:]))
        self.assertTrue(np.isfinite(kp.logprior("sin(sin(x0 + x1))")))
        kp = KatzPrior(
            n, self.basis_functions, equations=self.kp.all_eq[:30], root_context=False
        )
        t = kp.coder.process_all_equations(n + 1, ["sin(sin(x0 + x1))"], 2)
        self.assertEqual(t[0], (kp.coder.code["sin"],))

    def test_single_leaf(self):
        """A single-leaf function contributes exactly one (root) event"""
        for rc, (sm, norm) in itertools.product(
            [False, True], [("katz", False), ("katz", True), ("kn", True)]
        ):
            kp = KatzPrior(
                2,
                self.basis_functions,
                equations=self.kp.all_eq[:40],
                root_context=rc,
                normalise=norm,
                smoothing=sm,
            )
            for eq in ["x0", "a0", ["x0"], ["a0"]]:
                with self.subTest(root_context=rc, smoothing=sm, normalise=norm, eq=eq):
                    t = kp.coder.process_all_equations(3, [eq], kp.maxvar)
                    self.assertEqual(len(t), 1)
                    root = t[0]
                    p = kp.backoff_left.get_pbo(root[-1], root[:-1])
                    self.assertAlmostEqual(kp.logprior(eq), np.log(p))

    def test_second_leaf_follows_leaf(self):
        # "y" (a leaf that differs from its left sibling) can only follow an "x": the
        # normalised Katz and the Kneser-Ney models give it zero probability after any other
        # left sibling, wherever right_backoff puts the sibling in the context
        eqs = list(self.kp.all_eq)
        for smoothing in ("kn", "katz"):
            for rb in ("sibling", "parent"):
                for n in (2, 3):
                    kp = KatzPrior(
                        n,
                        self.basis_functions,
                        equations=eqs,
                        smoothing=smoothing,
                        right_backoff=rb,
                    )
                    x, y = kp.coder.code["x"], kp.coder.code["y"]
                    pos = -1 if rb == "sibling" else 0
                    ctxs = {
                        tuple(tt[-(n + 1) : -1])
                        for t in kp.eq_tuples
                        for tt in kp.split_events(t)[1]
                    }
                    seen = set()
                    for h in ctxs:
                        p = [kp.backoff_right.get_pbo(w, h) for w in kp.vocab_right]
                        self.assertAlmostEqual(sum(p), 1.0, places=12)
                        py = kp.backoff_right.get_pbo(y, h)
                        if h[pos] == x:
                            self.assertGreater(py, 0)
                        else:
                            self.assertEqual(py, 0.0)
                        seen.add(h[pos] == x)
                    self.assertEqual(seen, {True, False})
                    self.assertTrue(all(np.isfinite(kp.logprior(e)) for e in eqs[:20]))
                    if smoothing == "kn":
                        kp.tune_discounts("single", k=3, grid=[0.5, 0.8])
                        h = next(h for h in ctxs if h[pos] != x)
                        self.assertEqual(kp.backoff_right.get_pbo(y, h), 0.0)

    def test_kn_finite_and_normalised(self):
        eqs = list(self.kp_kn.all_eq[:20]) + [
            "arccos(arccos(arccos(x0)))",
            "tanh(tanh(x0))",
            "sin(x0) + sin(x1)",
        ]
        for eq in eqs:
            with self.subTest(eq=eq):
                p = self.kp_kn.logprior(eq)
                self.assertTrue(np.isfinite(p) and p < 0)
        for model, vocab in [
            (self.kp_kn.backoff_left, self.kp_kn.vocab_left),
            (self.kp_kn.backoff_right, self.kp_kn.vocab_right),
        ]:
            contexts = set()
            for k in range(1, model.order + 1):
                contexts.update(model.total[k].keys())
            for h in contexts:
                self.assertAlmostEqual(sum(model.prob(w, h) for w in vocab), 1.0, 12)

    def test_code_constants(self):
        eqs = ["pow(x0, a0)", "pow(a0, x0)"]
        for flag in [False, True]:
            kp = KatzPrior(
                2, self.basis_functions, equations=["x0**2"], code_constants=flag
            )
            t = [kp.coder.process_all_equations(3, [eq], 1) for eq in eqs]
            with self.subTest(code_constants=flag):
                if flag:
                    self.assertNotEqual(t[0], t[1])
                    self.assertIn(kp.coder.code["a"], t[0][1])
                else:
                    self.assertEqual(t[0], t[1])

    def test_right_backoff_needs_two_words(self):
        with self.assertRaises(ValueError):
            KatzPrior(
                1,
                self.basis_functions,
                equations=["x0 + sin(x1)"],
                right_backoff="parent",
            )

    def test_right_backoff(self):
        n = 2
        kp = KatzPrior(
            n,
            self.basis_functions,
            equations=self.kp_kn.all_eq[:30],
            right_backoff="parent",
        )
        code = kp.coder.code
        t = kp.coder.process_all_equations(n + 1, ["x0 + sin(x1)"], 2)
        _, right = kp.split_events(t)
        # (parent, left sibling, right) -> (left sibling, parent, right)
        self.assertEqual(right[0], (code["x"], code["+"], code["sin"]))
        self.assertTrue(np.isfinite(kp.logprior("x0 + sin(x1)")))

    def test_tune_discounts_overrides_scale(self):
        # tuned absolute discounts are applied without a discount_scale given at
        # construction, which the cross-validation did not use
        kp = KatzPrior(
            2,
            self.basis_functions,
            equations=self.kp_kn.all_eq[:40],
            smoothing_kwargs={"discount_scale": 0.5},
        )
        for method, grid in (("single", [0.4]), ("count", [(0.4, 0.9)])):
            with self.subTest(method=method):
                res = kp.tune_discounts(method, k=3, grid=grid)
                D = np.asarray(res["left"]["discounts"])
                self.assertTrue(
                    np.allclose(
                        kp.backoff_left.discounts[1:],
                        np.broadcast_to(D, (3, 3)) if D.ndim else D,
                    )
                )

    def test_tune_discounts(self):
        kp = KatzPrior(
            2,
            self.basis_functions,
            equations=self.kp_kn.all_eq,
            smoothing="kn",
            root_context=True,
        )
        res = kp.tune_discounts("single", k=5, grid=np.linspace(0.1, 1.0, 10))
        for side in ("left", "right"):
            self.assertTrue(0.1 <= res[side]["discounts"] <= 1.0)
        # Chosen discounts are applied to the refitted models
        self.assertAlmostEqual(
            kp.backoff_left.discounts[1, 0], res["left"]["discounts"]
        )
        self.assertAlmostEqual(
            kp.backoff_right.discounts[2, 2], res["right"]["discounts"]
        )
        # Minimum of the CV curve is at the chosen value
        j = np.argmin(res["cv_nll"][0])
        self.assertAlmostEqual(res["grid"][j], res["left"]["discounts"])
        res = kp.tune_discounts("scale", k=3, grid=[0.5, 1.0])
        self.assertIn(res["right"]["discount_scale"], [0.5, 1.0])
        # Count-dependent discounts (D1, D2+), the same at every order
        grid = [(0.5, 0.5), (0.8, 0.8), (0.6, 1.2), (0.9, 1.5)]
        res = kp.tune_discounts("count", k=3, grid=grid)
        for side, model in (("left", kp.backoff_left), ("right", kp.backoff_right)):
            D = np.asarray(res[side]["discounts"])
            self.assertEqual(D.shape, (3, 3))
            self.assertIn(tuple(D[0, :2]), grid)
            self.assertTrue(np.all(D == D[0]) and D[0, 1] == D[0, 2])
            self.assertTrue(np.allclose(model.discounts[1:], D))
        j = np.argmin(res["cv_nll"][1])
        self.assertEqual(tuple(res["grid"][j]), tuple(res["right"]["discounts"][0][:2]))
        self.assertTrue(np.isfinite(kp.logprior("sin(x0) + x1")))
        with self.assertRaises(ValueError):
            kp.tune_discounts("count", grid=[0.5, 0.8])
        with self.assertRaises(ValueError):
            KatzPrior(
                2, self.basis_functions, equations=["x0+x1"], smoothing="katz"
            ).tune_discounts()

    def test_default_is_kn(self):
        """Kneser-Ney with a fixed discount is the default; FunctionPrior is an alias"""
        from katz.prior import DEFAULT_KN_DISCOUNT, FunctionPrior

        eqs = self.kp_kn.all_eq[:40]
        kp = FunctionPrior(2, self.basis_functions, equations=eqs)
        self.assertEqual(kp.smoothing, "kn")
        self.assertEqual(kp.smoothing_kwargs["discounts"], DEFAULT_KN_DISCOUNT)
        explicit = KatzPrior(
            2,
            self.basis_functions,
            equations=eqs,
            smoothing="kn",
            smoothing_kwargs={"discounts": DEFAULT_KN_DISCOUNT},
        )
        mkn = KatzPrior(
            2,
            self.basis_functions,
            equations=eqs,
            smoothing="kn",
            smoothing_kwargs={"discounts": "mkn"},
        )
        for eq in ["x0**2", "sin(x0) + x1"]:
            self.assertEqual(kp.logprior(eq), explicit.logprior(eq))
            self.assertNotEqual(kp.logprior(eq), mkn.logprior(eq))


class TestEdgeCases(unittest.TestCase):
    """Edge cases found by an independent audit of the code"""

    basis_functions = BASIS

    def test_root_taken_from_tree(self):
        # the root event is the root of the tree even when the first lineage tuple is
        # skipped because the root's children include an ignored operator ("re")
        basis = [["a", "x", "x0", "x1"], ["sin", "re"], ["+", "*"]]
        for root_context in (False, True):
            coder = SymbolCoder(basis, root_context=root_context)
            for n in (2, 3):
                for labels, root in [
                    (["*", "sin", "sin", "x0", "re", "x1"], "*"),
                    (["*", "re", "sin", "x0", "x1"], "*"),
                    (["re", "sin", "x0"], "sin"),
                ]:
                    t = coder.labels2ntuples(n, labels)
                    self.assertEqual(t[0][-1], coder.code[root])

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
        for fname, expected in [
            ("data/NewPhysics.csv", 11),
            ("data/NewFeynman.csv", 9),
        ]:
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
        for e0, e1 in [
            ("sqrt(x0**2)", "sqrt(x12**2)"),
            ("sqrt(a0**2)", "sqrt(a15**2)"),
            ("x0*a2 + a5", "x12*a2 + a15"),
        ]:
            with self.subTest(eq=e1):
                self.assertTrue(np.isfinite(kp.logprior(e0)))
                self.assertAlmostEqual(kp.logprior(e1), kp.logprior(e0), places=12)

    def test_corpus_parse_is_reproducible(self):
        # the parsed corpus defines the prior: fail if sympy changes how it writes it
        import json

        import pandas as pd
        import sympy

        _require_sympy_114(self)
        with open(
            os.path.join(os.path.dirname(__file__), "corpus_labels_sympy1.14.json")
        ) as f:
            ref = json.load(f)
        for name in ("NewPhysics", "NewFeynman"):
            eqs = list(pd.read_csv(f"data/{name}.csv")["New Formula"])
            kp = KatzPrior(2, ref["basis_functions"], equations=eqs)
            locs = {
                f"{c}{i}": sympy.Symbol(f"{c}{i}", real=True)
                for c in "xa"
                for i in range(kp.maxvar)
            }
            for i, eq in enumerate(eqs):
                with self.subTest(corpus=name, eq=eq):
                    self.assertEqual(
                        kp.coder.equation2labels(eq, locs), ref["labels"][name][i]
                    )

    def test_default_model_pickles(self):
        # the default (normalised) model survives pickling and copying, e.g. for
        # multiprocessing
        import copy
        import pickle

        for sm in ("kn", "katz"):
            kp = KatzPrior(
                2,
                self.basis_functions,
                equations=["x0 + x1", "sin(x0)*x1"],
                smoothing=sm,
            )
            for clone in (pickle.loads(pickle.dumps(kp)), copy.deepcopy(kp)):
                for eq in ["x0 + x1", "sin(x0)"]:
                    with self.subTest(smoothing=sm, eq=eq):
                        self.assertEqual(clone.logprior(eq), kp.logprior(eq))

    def test_single_leaf_training_equation(self):
        # a single-leaf corpus equation contributes one root event to the training data
        eqs = ["x0", "x0 + x1", "sin(x0)*x1", "x0*x1 + x1"]
        kp = KatzPrior(
            2,
            self.basis_functions,
            equations=eqs,
            smoothing="katz",
            root_context=False,
            normalise=False,
        )
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
        self.assertNotEqual(
            kp.backoff_left.get_pbo(words[1], ()), double.get_pbo(words[1], ())
        )

    def test_corpus_without_binary_operators(self):
        eqs = ["sin(x0)", "exp(sqrt(x0))", "x0"]
        kp = KatzPrior(2, self.basis_functions, equations=eqs)
        self.assertEqual(len(kp.backoff_right.counts[1]), 0)
        for eq in ["sin(x0)", "x0 * sin(x0)", "x0"]:
            self.assertTrue(np.isfinite(kp.logprior(eq)))
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=eqs, smoothing="katz")

    def test_normalised_katz(self):
        # the original Katz conditionals need not sum to one; the normalised ones do
        import pandas as pd

        eqs = list(pd.read_csv("data/NewPhysics.csv")["New Formula"])
        raw = KatzPrior(
            3, self.basis_functions, equations=eqs, smoothing="katz", normalise=False
        )
        norm = KatzPrior(3, self.basis_functions, equations=eqs, smoothing="katz")
        none = norm.coder.code["None"]
        contexts = {
            tuple(tt[-(norm.n + 1) : -1])
            for eq in eqs
            for tt in norm.coder.process_all_equations(norm.n + 1, [eq], norm.maxvar)[
                1:
            ]
            if tt[-1] != none
        }
        m, r = norm.backoff_right, raw.backoff_right
        raw_sums = [sum(r.get_pbo(w, h) for w in m.words) for h in contexts]
        self.assertLess(min(raw_sums), 0.99)  # the unnormalised model loses mass
        for h in contexts:
            p = [m.get_pbo(w, h) for w in m.words]
            self.assertTrue(min(p) >= 0)
            self.assertAlmostEqual(sum(p), 1.0, places=12)
        # BackOff itself is unchanged: normalise=False is the unnormalised model (the sums are
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
        p += [
            kp.backoff_right.get_pbo(tt[-1], tt[-(kp.n + 1) : -1])
            for tt in t[1:]
            if tt[-1] != none
        ]
        return float(np.sum(np.log(p)))

    def test_kn_parameter_validation(self):
        corpus = [("r", "a"), ("r", "b")]
        for kw in [
            dict(discounts=0.0),
            dict(discounts=float("nan")),
            dict(discounts=-0.5),
            dict(discount_scale=-1.0),
            dict(discount_scale=float("inf")),
            dict(discounts=[0.5, 0.5, 0.5]),  # wrong shape for order 2
            dict(order=0),
        ]:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    KneserNey(corpus, vocab=["c"], **kw)
        m = KneserNey(corpus, vocab=["c"], discounts=[0.5, 0.7])
        self.assertTrue(m.prob("c", ("r",)) > 0)
        # Discounts above their maximum D_r = r are capped
        m = KneserNey(corpus, discounts=0.9, discount_scale=3.0)
        self.assertTrue(np.all(m.discounts[1:] == [1.0, 2.0, 2.7]))

    def test_kn_empty_corpus(self):
        m = KneserNey([], order=2, vocab=["a", "b", "c", "d"])
        for w in "abcd":
            self.assertAlmostEqual(m.prob(w, ("z",)), 0.25)
        with self.assertRaises(ValueError):
            KneserNey([])
        m = KneserNey([("r", "a")] * 5, cache_size=2)
        p = [m.prob(w, ("r",)) for w in "a" * 3]
        self.assertTrue(len(m._cache) <= 2 and p[0] == p[2])

    def test_tune_discounts_validation(self):
        eqs = ["sin(x0) + x1", "x0*x1", "exp(x0) - x1", "sqrt(x0)/x1"]
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=eqs[:1]).tune_discounts()
        kp = KatzPrior(2, self.basis_functions, equations=eqs)
        before = kp.logprior("sin(x0) * x1")
        for kw in [
            dict(k=1),
            dict(grid=[]),
            dict(grid=[0.5, np.nan]),
            dict(grid=[0.0]),
        ]:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    kp.tune_discounts(**kw)
        self.assertEqual(kp.logprior("sin(x0) * x1"), before)
        res = kp.tune_discounts(k=2)  # two equations per fold
        self.assertTrue(np.all(np.isfinite(res["cv_nll"])))

    def test_cache_size(self):
        for bad in (-1, 2.8, True, "10"):
            with self.subTest(cache_size=bad), self.assertRaises(ValueError):
                KneserNey([("a", "b")], order=2, cache_size=bad)
        corpus = [("r", "a", "b"), ("r", "b", "a"), ("r", "a", "a")]
        ref = KneserNey(corpus, vocab=["a", "b", "c"])
        for size in (0, 1, 3):
            m = KneserNey(corpus, vocab=["a", "b", "c"], cache_size=size)
            for w, h in itertools.product("abcz", [(), ("a",), ("r", "a")]):
                self.assertEqual(m.prob(w, h), ref.prob(w, h))
            self.assertLessEqual(len(m._cache), size)

    def test_random_normalisation(self):
        # every conditional distribution is a probability distribution over the vocabulary
        rng = np.random.default_rng(420)
        for j in range(200):
            order = int(rng.integers(1, 6))
            corpus = [
                tuple(rng.integers(0, 4, size=int(rng.integers(1, order + 1))))
                for _ in range(int(rng.integers(0, 40)))
            ]
            discounts = ["mkn", "kn", 0.8, np.tile([0.8, 1.3, 1.7], (order, 1))][j % 4]
            m = KneserNey(corpus, order=order, vocab=range(4), discounts=discounts)
            for k in range(min(order, 4)):
                h = tuple(rng.integers(0, 4, size=k))
                p = np.array([m.prob(w, h) for w in range(4)])
                self.assertTrue(np.all(np.isfinite(p)) and np.all(p >= 0))
                self.assertAlmostEqual(p.sum(), 1.0, places=12)


@unittest.skipUnless(HAVE_MPI, "katz.esr_prior needs mpi4py")
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
        esr_prior.compute_logprior(
            1,
            2,
            self.basis,
            td,
            None,
            None,
            prior_kwargs={"equations": self.corpus},
            **kw,
        )
        base = os.path.join(td, "compl_1", "kn_")
        return (
            float(np.loadtxt(base + "logprior_2_1.txt")),
            os.stat(base + "prior_2_1.json").st_mtime_ns,
        )

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
            code = float(np.loadtxt(os.path.join(td, "compl_1", "kn_codelen_2_1.txt")))
            self.assertAlmostEqual(code, v1 + 1.5, places=10)
            # tampered output
            np.savetxt(os.path.join(td, "compl_1", "kn_logprior_2_1.txt"), [0.0])
            v3, t3 = self._run(td)
            self.assertAlmostEqual(v3, v1, places=10)
            self.assertNotEqual(t3, t2)
            # logconst follows the tree file
            self._library(td, "sqrt(x0)", ["sqrt", "x0"])
            self.assertEqual(float(np.loadtxt(lc)), 0.0)

    def test_parser_in_code_hash(self):
        # the scores depend on the equation parser, so a change to it must invalidate
        # saved output
        from unittest import mock

        srcs = esr_prior._scoring_sources()
        names = [os.path.basename(p) for p in srcs]
        self.assertTrue(
            {"tree_parser.py", "kneser_ney.py", "process_equations.py"} <= set(names)
        )
        with tempfile.TemporaryDirectory() as td:
            i = names.index("tree_parser.py")
            copy_ = os.path.join(td, "tree_parser.py")
            with open(srcs[i]) as f:
                src = f.read()
            h0 = esr_prior._code_sha256()
            changed = srcs[:i] + [copy_] + srcs[i + 1 :]
            # comments, docstrings and formatting do not enter the hash; code does
            for extra, differs in [
                ("\n# changed\n", False),
                ('\n\n\ndef _extra():\n    """Docstring"""\n', True),
                ("\n_changed = 1\n", True),
            ]:
                with open(copy_, "w") as g:
                    g.write(src + extra)
                with mock.patch.object(
                    esr_prior, "_scoring_sources", return_value=changed
                ):
                    self.assertEqual(esr_prior._code_sha256() != h0, differs)
            self._library(td, "x0", ["x0"])
            v0, t0 = self._run(td)
            with mock.patch.object(esr_prior, "_code_sha256", return_value="other"):
                v1, t1 = self._run(td)
            self.assertNotEqual(t1, t0)
            self.assertEqual(v1, v0)

    def test_manifest_config_is_json(self):
        # numpy booleans (e.g. from a parameter grid) are recorded as JSON booleans
        import json

        cfg = esr_prior._prior_config(
            3,
            self.basis,
            {"root_context": np.bool_(True), "code_constants": np.bool_(False)},
        )
        self.assertEqual(json.loads(json.dumps(cfg))["root_context"], True)

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
                1,
                2,
                self.basis,
                td,
                None,
                None,
                label="nan",
                prior_kwargs={
                    "equations": ["x0 + x1"],
                    "smoothing": "katz",
                    "normalise": False,
                },
            )
            vals = np.loadtxt(os.path.join(d, "nan_logprior_2_1.txt"))
            with open(os.path.join(d, "nan_prior_2_1.json")) as f:
                manifest = json.load(f)
            self.assertTrue(np.isnan(vals[0]) and np.isfinite(vals[1]))
            self.assertEqual(manifest["n_failed"], 1)


class TestTreeParser(unittest.TestCase):
    """The built-in (sympy-only) tree parser"""

    basis_functions = [
        ["a", "x"],
        ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
        ["+", "-", "*", "/", "pow"],
    ]
    equations = [
        "x0**2",
        "sin(x0) + x1",
        "x0 - x1",
        "x0/x1",
        "1/x0",
        "sqrt(x0)*exp(-x1)",
        "pow(Abs(a0), x0) + a1",
        "a0*x0**3 + 2.5",
        "x0",
        "arccos(tanh(x0))/log(x1)",
    ]

    @unittest.skipUnless(HAVE_ESR, "ESR is not installed")
    def test_same_as_esr(self):
        import katz.process_equations as pe

        import pandas as pd

        eqs = self.equations + list(pd.read_csv("data/NewPhysics.csv")["New Formula"])
        kp = KatzPrior(3, self.basis_functions, equations=eqs)  # basis x0 ... x10
        for rc, cc in [(False, False), (True, True)]:
            coder = pe.SymbolCoder(
                kp.basis_functions, code_constants=cc, root_context=rc
            )
            try:
                pe.generator = esr.generation.generator
                ref = [coder.process_all_equations(4, [e], 11) for e in eqs]
            finally:
                pe.generator = tree_parser
            new = [coder.process_all_equations(4, [e], 11) for e in eqs]
            self.assertEqual(new, ref)


if __name__ == "__main__":
    unittest.main()
