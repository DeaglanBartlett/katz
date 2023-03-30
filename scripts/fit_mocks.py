import sys
import os
import numpy as np
from mpi4py import MPI

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier
import esr.fitting.test_all as test_all
import esr.fitting.test_all_Fisher as test_all_Fisher
import esr.fitting.match as match
import esr.fitting.combine_DL as combine_DL
import esr.fitting.plot as plot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class MockLikelihood:

    def __init__(self, name, nx, fracsig_x):
        """Likelihood class used to fit mocks

        Args:
            :name (str): The name of the mock dataset
            :nx (int): Number of data points in mock
            :fracsig_x (float): Fraction of std deviation used as errors
        """

        esr_dir = os.path.abspath(os.path.join(os.path.dirname(esr.generation.simplifier.__file__), '..', '')) + '/'
        fname = f'{name}_{nx}_{frac_sigx}'
        self.data_dir = '../data/'
        self.data_file = self.data_dir + fname + '.txt'
        self.fn_dir = esr_dir + "function_library/new_osc_maths/"

        self.fnprior_prefix = "aifeyn_"
        self.combineDL_prefix = "combine_DL_"
        self.final_prefix = "final_"

        self.base_out_dir = "output/"
        self.temp_dir = self.base_out_dir + "/partial_" + fname
        self.out_dir = self.base_out_dir + "/output_" + fname
        self.fig_dir = self.base_out_dir + "/figs_" + fname

        self.ylabel = r'$y$'  # for plotting
        self.xvar, self.yvar, self.yerr = np.genfromtxt(self.data_file, unpack=True)
        self.inv_cov = 1 / self.yerr ** 2


    def get_pred(self, x, a, eq_numpy, **kwargs):
        """Return evaluated function

        Args:
            :x (float or np.array): input variable
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2

        Returns:
            :y (float or np.array): the predicted value of y

        """
        return eq_numpy(x, *a)


    def clear_data(self):
        """Clear data used for numerical integration (not required here)"""
        pass


    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function

        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y

        Returns:
            :nll (float): - log(likelihood) for this function and parameters

        """

        y = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(y)):
            return np.inf
        nll = np.sum(0.5 * (y - self.yvar) ** 2 * self.inv_cov)  # inv_cov diagonal, so is vector here
        if np.isnan(nll):
            return np.inf
        return nll


    def run_sympify(self, fcn_i, **kwargs):
        """Sympify a function

        Args:
            :fcn_i (str): string representing function we wish to fit to data

        Returns:
            :fcn_i (str): string representing function we wish to fit to data (with superfluous characters removed)
            :eq (sympy object): sympy object representing function we wish to fit to data
            :integrated (bool, always False): whether we analytically integrated the function (True) or not (False)

        """

        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace('\'', '')

        eq = sympy.sympify(fcn_i,
                    locals={"inv": inv,
                            "square": square,
                            "cube": cube,
                            "sqrt": sqrt,
                            "log": log,
                            "pow": pow,
                            "x": x,
                            "a0": a0,
                            "a1": a1,
                            "a2": a2,
                            "a3": a3})
        return fcn_i, eq, False


def print_text(text):
    if rank != 0:
        return
    stars = ["*" * 20]
    print('\n')
    print(*stars)
    print(text)
    print(*stars)
    print('\n')
    return


comp = int(sys.argv[1])
tmax = 5

nx = 10000
frac_sigx = 0.5
#name = 'nguyen_8'
name = 'korns_4'
likelihood = MockLikelihood(name, nx, frac_sigx)

print_text('test_all')
test_all.main(comp, likelihood, tmax=5)
comm.Barrier()

print_text('test_all_Fisher')
test_all_Fisher.main(comp, likelihood, tmax=tmax)
comm.Barrier()

print_text('match')
match.main(comp, likelihood, tmax=tmax)
comm.Barrier()

print_text('combine_DL')
combine_DL.main(comp, likelihood)
comm.Barrier()

print_text('plot')
if rank == 0:
    plot.main(comp, likelihood, tmax=tmax)

