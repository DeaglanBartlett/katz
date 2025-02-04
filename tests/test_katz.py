import unittest
import numpy as np
import os
import esr.generation.duplicate_checker

from katz.prior import KatzPrior
from katz.esr_prior import get_logconst, compute_logprior

class TestKatzPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 2
        basis_functions = [["a", "x"],
                           ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                           ["+", "-", "*", "/", "pow"]]
        cls.kp = KatzPrior(n, basis_functions, 'data/FeynmanEquations.csv', 'data/NewFeynman.csv')

    def test_logprior_string_equations(self):
        equations = ['x0**2', 'sin(x0) + sin(x1)', 'sin(sin(x0+x1))']
        expected_results = [np.float64(-3.185524492278148), np.float64(-18.45574760511283), np.float64(-17.08632036411751)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp.logprior(eq), expected, places=4)

    def test_logprior_list_equations(self):
        equations = [['+', 'x0', 'x0'], ['*', '2', 'x0'], ['+', 'x0', 'x1'], ['+', 'sin', 'x0', 'sin', 'x1']]
        expected_results = [self.kp.logprior(eq) for eq in equations]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertEqual(self.kp.logprior(eq), expected)


class TestESRPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.comp = 5
        esr.generation.duplicate_checker.main('core_maths', cls.comp)
        cls.dirname = 'ESR/esr/function_library/core_maths/'
        cls.n = 2
        cls.basis_functions = [["a", "x"],
                               ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                               ["+", "-", "*", "/", "pow"]]
        cls.in_eqfile = 'data/FeynmanEquations.csv'
        cls.out_eqfile = 'data/NewFeynman.csv'
        cls.input_delimiter = ','
        cls.use_tree = False

    def test_get_logconst(self):
        # Ensure the directory exists
        os.makedirs(self.dirname, exist_ok=True)
        # Test get_logconst function
        get_logconst(self.comp, self.dirname, overwrite=True)
        # Check if the logconst file is created
        logconst_file = os.path.join(self.dirname, f'compl_{self.comp}', f'logconst_{self.comp}.txt')
        self.assertTrue(os.path.isfile(logconst_file))

    def test_compute_logprior(self):
        # Ensure the directory exists
        os.makedirs(self.dirname, exist_ok=True)
        # Test compute_logprior function
        compute_logprior(self.comp, self.n, self.basis_functions, self.dirname, 
                         self.in_eqfile, self.out_eqfile, overwrite=True, 
                         input_delimiter=self.input_delimiter, use_tree=self.use_tree)
        # Check if the logprior file is created
        logprior_file = os.path.join(self.dirname, f'compl_{self.comp}', f'katz_logprior_{self.n}_{self.comp}.txt')
        self.assertTrue(os.path.isfile(logprior_file))
        # Check if the codelen file is created
        codelen_file = os.path.join(self.dirname, f'compl_{self.comp}', f'katz_codelen_{self.n}_{self.comp}.txt')
        self.assertTrue(os.path.isfile(codelen_file))


if __name__ == '__main__':
    unittest.main()