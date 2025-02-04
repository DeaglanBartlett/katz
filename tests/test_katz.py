import unittest
import numpy as np
import os
import esr.generation.duplicate_checker
import itertools

from katz.prior import KatzPrior
from katz.esr_prior import get_logconst, compute_logprior
from katz.back_off import BackOff

class TestKatzPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 2
        basis_functions = [["a", "x"],
                           ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                           ["+", "-", "*", "/", "pow"]]
        cls.kp_feynman = KatzPrior(n, basis_functions, 'data/FeynmanEquations.csv', 'data/NewFeynman.csv')

        basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
                ["+", "-", "*", "/", "pow"]]
        cls.kp_physics = KatzPrior(n, basis_functions, 'data/PhysicsEquations.csv', 'data/NewPhysics.csv', input_delimiter=';')

    def test_logprior_string_equations(self):
        equations = ['x0**2', 'sin(x0) + sin(x1)', 'sin(sin(x0+x1))']

        # Feynman prior
        expected_results = [np.float64(-3.185524492278148), np.float64(-18.45574760511283), np.float64(-17.08632036411751)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_feynman.logprior(eq), expected, places=4)

        # Physics prior
        expected_results = [np.float64(-3.540935719892494), np.float64(-19.93661285074596), np.float64(-17.21883673566916)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_physics.logprior(eq), expected, places=4)

    def test_logprior_list_equations(self):
        equations = [['+', 'x0', 'x0'], ['*', '2', 'x0'], ['+', 'x0', 'x1'], ['+', 'sin', 'x0', 'sin', 'x1']]

        # Feynman prior
        expected_results = [np.float64(-9.30695481431519), np.float64(-2.8626638924112906), np.float64(-5.474299859409948), np.float64(-18.45574760511283)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_feynman.logprior(eq), expected, places=4)

        # Physics prior
        expected_results = [np.float64(-9.135616417476996), np.float64(-3.0849759909660257), np.float64(-4.759488768141304), np.float64(-19.93661285074596)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_physics.logprior(eq), expected, places=4)

    def test_op2str(self):
        expected = {
            'pi': 'a',
            '0.5': 'a',
            'Symbol': 'x',
            'x0': 'x',
            'x1': 'x',
            'a0': 'x',
            'a1': 'x',
            'Add': '+',
            'Sub': '-',
            'Mul': '*',
            'Div': '/',
            'Pow': 'pow',
            'sin': 'sin',
        }
        for op, string in expected.items():
            with self.subTest(op=op):
                self.assertEqual(self.kp_feynman.coder.op2str(op), string)
                self.assertEqual(self.kp_physics.coder.op2str(op), string)

        # Check unknown operator raises Exception
        bad_ops = ['unknown', 'Cosh', 'Sinh', 'garbage', 'y0', 'b2']
        for op in bad_ops:
            with self.assertRaises(Exception):
                self.kp_feynamn.coder.op2str(op)
            with self.assertRaises(Exception):
                self.kp_physics.coder.op2str(op)


class TestBackOff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 1
        cls.data_file = 'data/romeoandjuliet.txt'
        with open (cls.data_file, 'r') as f:
            cls.data = f.readlines()[80:]
        cls.data = [line.strip().split() for line in cls.data]
        cls.data = list(itertools.chain(*cls.data))
        cls.data = [(word,) for word in cls.data]

    def test_bo(self):
        self.bo = BackOff(self.data)
        words_in_play = ['Romeo', 'Montague', 'Juliet', 'Capulet', 'the', 'a', 'and', 'to', 'of', 'in']
        for word in words_in_play:
            with self.subTest(word=word):
                self.assertIn(word, self.bo.words)
                self.assertTrue(self.bo.get_pbo(word, ()) > 0)
        words_not_in_play = ['Macbeth', 'Hamlet', 'Othello', 'Computer', 'Avocado', 'Pineapple']
        for word in words_not_in_play:
            with self.subTest(word=word):
                self.assertNotIn(word, self.bo.words)
                self.assertEqual(self.bo.get_pbo(word, ()), 0)

class TestESRPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.comp = 5
        esr.generation.duplicate_checker.main('core_maths', cls.comp)
        cls.dirname = 'ESR/esr/function_library/core_maths/'
        cls.n = 2
        cls.basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
                ["+", "-", "*", "/", "pow"]]
        cls.in_eqfile = 'data/FeynmanEquations.csv'
        cls.out_eqfile = 'data/NewFeynman.csv'
        cls.input_delimiter = ','

    def test_compute_logprior(self):

        # Ensure the directory exists
        os.makedirs(self.dirname, exist_ok=True)
        
        # Test get_logconst function
        for overwrite in [True, False]:
            get_logconst(self.comp, self.dirname, overwrite=overwrite)

        # Check if the logconst file is created
        logconst_file = os.path.join(self.dirname, f'compl_{self.comp}', f'logconst_{self.comp}.txt')
        self.assertTrue(os.path.isfile(logconst_file))

        for use_tree in [False, True]:

            # Test compute_logprior function
            for overwrite in [True, False]:
                compute_logprior(self.comp, self.n, self.basis_functions, self.dirname, 
                                self.in_eqfile, self.out_eqfile, overwrite=overwrite, 
                                input_delimiter=self.input_delimiter, use_tree=use_tree)
            
            # Check if equation file is created
            eq_file = os.path.join(self.dirname, f'compl_{self.comp}', f'all_equations_{self.comp}.txt')
            self.assertTrue(os.path.isfile(eq_file))
            
            # Check if the logprior file is created
            logprior_file = os.path.join(self.dirname, f'compl_{self.comp}', f'katz_logprior_{self.n}_{self.comp}.txt')
            self.assertTrue(os.path.isfile(logprior_file))
            
            # Check if the codelen file is created
            codelen_file = os.path.join(self.dirname, f'compl_{self.comp}', f'katz_codelen_{self.n}_{self.comp}.txt')
            self.assertTrue(os.path.isfile(codelen_file))

            # Load equations and logprior values
            logprior = np.loadtxt(logprior_file)
            codelen = np.loadtxt(codelen_file)
            with open(eq_file, 'r') as f:
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
                        self.assertTrue(np.all(logprior[indices] == logprior[indices[0]]))
                    else:
                        print(s, logprior[indices])
                        self.assertTrue(np.all(~np.isfinite(logprior[indices])))
        
            # Check that at least some values are finite
            self.assertTrue(np.any(np.isfinite(logprior)))
            
if __name__ == '__main__':
    unittest.main()