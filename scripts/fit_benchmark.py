import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.abc import x
import sys
import os
from mpi4py import MPI
import itertools
import csv

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


def get_sig(f, x_range, frac_sigx):
    """
    Determine the noise level to use. This is given as frac_sigx times
    the standard deviation of the functions values evaluated on 10^5
    randomly generated points within the x_range
    
    Args:
        :f (str): The function which should be evaluated
        :x_range (list[float, floar]): The [min, max] values of x to consider
        :frac_sigx (float): The fraction of the standard deviation to use as sigma
        
    Returns:
        :sig (float): The value of sigma to use
    
    """

    np.random.seed(0)
    nx = 10000

    expr = sympy.sympify(f)
    fun = sympy.lambdify(x, expr, modules='numpy')

    xdata = np.random.uniform(*x_range, nx)
    xdata = np.sort(xdata)

    ydata = fun(xdata)
    sig = frac_sigx * np.std(ydata)

    return sig

def make_data(name, f, x_range, nx, frac_sigx, samp_num, sig, make_fig=False):
    """
    Make a mock data sample from a given equation with a certain number of data points
    and a given noise level. The results are saved to a file in the directory '../data/'
    
    Args:
        :name (str): The name of the equation to be used
        :f (str): The function which should be evaluated
        :x_range (list[float, floar]): The [min, max] values of x to consider
        :nx (int): The number of data points to be used in the mock
        :frac_sigx (float): The fraction of the standard deviation to use as sigma
        :samp_num (int): The mock number (which sets the seed of the random number generator)
        :sig (float): The value of sigma to use for Gaussian noise
        :make_fig (bool): Whether to make a plot of the mock data with the generating function
        
    Returns:
        :None
    
    """

    np.random.seed(samp_num)

    expr = sympy.sympify(f)
    fun = sympy.lambdify(x, expr, modules='numpy')

    xdata = np.random.uniform(*x_range, nx)
    xdata = np.sort(xdata)

    # Truth
    ydata = fun(xdata)
    plt.plot(xdata, ydata)

    # Scatter
    ydata = ydata + np.random.normal(size=nx) * sig

    fname = f'{name}_{nx}_{frac_sigx}_{samp_num}'

    if make_fig:
        plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':-1,
                 'capsize':1, 'elinewidth':0.5, 'color':'k', 'alpha':1}
        plt.errorbar(xdata,
                ydata,
                yerr=sig,
                **plot_kwargs)

        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(name + ': ' + r'$%s$'%sympy.latex(expr))
        plt.tight_layout()

        plt.savefig(f'../figs/{fname}.png')
        plt.clf()

    np.savetxt(f'../data/{fname}.txt', np.vstack([xdata, ydata, sig*np.ones(len(xdata))]).transpose())

    return

class MockLikelihood:

    def __init__(self, name, nx, frac_sigx, samp_num):
        """Likelihood class used to fit mocks

        Args:
            :name (str): The name of the mock dataset
            :nx (int): Number of data points in mock
            :fracsig_x (float): Fraction of std deviation used as errors
        """

        esr_dir = os.path.abspath(os.path.join(os.path.dirname(esr.generation.simplifier.__file__), '..', '')) + '/'
        fname = f'{name}_{nx}_{frac_sigx}_{samp_num}'
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

    def get_pred(self, xdata, a, eq_numpy, **kwargs):
        """Return evaluated function

        Args:
            :xdata (float or np.array): input variable
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2

        Returns:
            :y (float or np.array): the predicted value of y

        """
        return eq_numpy(xdata, *a)


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
    """
    Function to print progress announcements in standardised format
    
    Args:
        :text (str): The text to be printed
        
    Returns:
        :None
        
    """
    if rank == 0:
        stars = ["*" * len(text)]
        print('\n')
        print(*stars)
        print(text)
        print(*stars)
        print('\n')
    comm.Barrier()
    return


def fit_mocks(name, nx, frac_sigx, samp_num, comp, tmax=5):
    """
    Run ESR for a given mock sample using the default MDL prescription
    
    Args:
        :name (str): The name of the equation to be used
        :nx (int): The number of data points to be used in the mock
        :frac_sigx (float): The fraction of the standard deviation to use as sigma
        :samp_num (int): The mock number
        :comp (int): The complexity of function to consider
        :tmax (float): maximum time in seconds to run any one part of simplification procedure for a given function
        
    Returns:
        :None
        
    """

    likelihood = MockLikelihood(name, nx, frac_sigx, samp_num)

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

    return


def _apply_language_prior(name, nx, frac_sigx, samp_num, comp, tmax=5):
    """
    Function which applies language-model function prior model selection
    methods to the results of an ESR run.
    
    Args:
        :name (str): The name of the equation to be used
        :nx (int): The number of data points to be used in the mock
        :frac_sigx (float): The fraction of the standard deviation to use as sigma
        :samp_num (int): The mock number
        :comp (int): The complexity of function to consider
        :tmax (float): maximum time in seconds to run any one part of simplification procedure for a given function
        
    Returns:
        :None
    
    """

    likelihood = MockLikelihood(name, nx, frac_sigx, samp_num)
    apply_language_prior(likelihood, comp, tmax=tmax)
    
    return

def apply_language_prior(likelihood, comp, tmax=5):
    """
    Function which applies language-model function prior model selection
    methods to the results of an ESR run given a likelihood class
    
    Args:
        :likelihood (esr.fitting.likelihood object): object containing data and likelihood function
        :comp (int): The complexity of function to consider
        :tmax (float): maximum time in seconds to run any one part of simplification procedure for a given function
    
    Returns:
        :None
        
    """
    

    fnprior_prefix = likelihood.fnprior_prefix
    combineDL_prefix = likelihood.combineDL_prefix
    final_prefix = likelihood.final_prefix
    fig_dir = likelihood.fig_dir

    # INCLUDE LOG(CONST) TERMS

    likelihood.fnprior_prefix = f"katz_codelen_2_"
    likelihood.combineDL_prefix = "combine_DL_katz_2_"
    likelihood.final_prefix = "final_katz_2_"
    likelihood.fig_dir += "_katz_2"

    print_text('combine_DL language model (with const)')
    combine_DL.main(comp, likelihood)
    comm.Barrier()

    print_text('plot language model')
    if rank == 0:
        plot.main(comp, likelihood, tmax=tmax)

    # NO LOG(CONST) TERMS

    likelihood.fnprior_prefix = f"katz_logprior_2_"
    likelihood.combineDL_prefix = "combine_DL_noconst_katz_2_"
    likelihood.final_prefix = "final_noconst_katz_2_"
    likelihood.fig_dir += "_noconst_katz_2"

    print_text('combine_DL language model (no const)')
    combine_DL.main(comp, likelihood)
    comm.Barrier()

    print_text('plot language model')
    if rank == 0:
        plot.main(comp, likelihood, tmax=tmax)

    # Restore file names

    likelihood.fig_dir = fig_dir
    likelihood.fnprior_prefix = fnprior_prefix
    likelihood.combineDL_prefix = combineDL_prefix
    likelihood.final_prefix = final_prefix

    return


def get_split_idx(L):
    """
    Find the indices of an array of length L which should be considered
    by this rank. For output data_start, data_end, the rank considers the
    entries array[data_start:data_end] where len(array) = L.
    
    Args:
        :L (int): The length of the array we wish to split among ranks.
        
    Returns:
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank.
        
    """

    if rank==0:
        print("Number of cores:", size, flush=True)

    nLs = int(np.ceil(L / float(size)))       # Number of lines per file for given thread

    while nLs*(size-1) > L:
        if rank==0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank==0:
        print("Total number of functions: ", L, flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank==size-1:
        data_end = L

    return data_start, data_end


def process_data(dirname, final_prefix, all_comp):
    """
    Convert results of all optimisations into a list of functions,
    where we attempt to keep only the highest ranked of any duplicate
    equation. This will not catch all duplicates, so the user must
    check for them.
    
    Args:
        :dirname (str): Directory name containing the optimisation results
        :final_prefix (str): Start of file names which contain result
        :all_comp (list[int]): All complexity of equation to consider
        
    Returns:
        :fun (list[str]): The list of functions selected
        :res (np.ndarray): The terms used for model selection of the returned functions
        :params (np.ndarray): The maximimum likelihood parameters of the returned functions
        :store_comp (np.ndarray): The complexities of the returned functions
    
    """


    res = []
    fun = []
    params = []
    store_comp = []

    # (1) Load the data
    for i, compl in enumerate(all_comp):
        fname = dirname + final_prefix + '%i.dat'%compl
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                reader = csv.reader(f, delimiter=';')
                data = [row for row in reader]

                res += [d[2:7] for d in data]
                fun += [d[1] for d in data]
                params += [d[7:] for d in data]
                store_comp += [compl] * len(data)
    res = np.array(res, dtype=float)
    params = np.array(params, dtype=float)
    store_comp = np.array(store_comp, dtype=int)

    # (2) Remove low-ranked functions and sort data
    imax = res.shape[0]   # MAYBE WANT TO CHANGE THIS
    m = np.argsort(res[:,0], kind='stable')
    res = res[m,:]
    params = params[m,:]
    store_comp = store_comp[m]
    fun = [fun[i] for i in m]
    res = res[:imax,:]
    params = params[:imax,:]
    store_comp = store_comp[:imax]
    fun = fun[:imax]

    # (3) Remove duplicates by likelihood
    _, uniq_idx = np.unique(res[:,2], return_index=True)
    uniq_idx = np.sort(uniq_idx)
    fun = [fun[i] for i in uniq_idx]
    res = res[uniq_idx,:]
    store_comp = store_comp[uniq_idx]
    params = params[uniq_idx,:]

    # (4) Remove duplicates by name
    new_fun = []
    uniq_idx = []
    for i in range(len(fun)):
        if fun[i] not in new_fun:
            uniq_idx.append(i)
            new_fun.append(fun[i])
    fun = new_fun
    res = res[uniq_idx,:]
    params = params[uniq_idx,:]
    store_comp = store_comp[uniq_idx]
    
    return fun, res, params, store_comp


def _process_fit(name, all_true_eq, nx, frac_sigx, samp_num, all_comp):
    """
    Run the function process_fit for a given mock sample
    
    Args:
        :name (str): The name of the equation to be used
        :all_true_eq (list[str]): List of variants of the true equation to find
        :nx (int): The number of data points to be used in the mock
        :frac_sigx (float): The fraction of the standard deviation to use as sigma
        :samp_num (int): The mock number
        :all_comp (list[int]): All complexity of equation to consider
    
    Returns:
        :None
    
    """
    fname = f'{name}_{nx}_{frac_sigx}_{samp_num}'
    dirname = f'output/output_{fname}/'
    process_fit(dirname, all_comp, nx, all_true_eq)
    return


def process_fit(dirname, all_comp, nx, all_true_eq=None):
    """
    Process the results of all fits to give a function ranking according
    to different model selection methods. If all_true_eq is not None, then
    this will also find the location of the true equation in the rankings.
    The results are outputted to a file called selection_summary.csv in
    the directory given by dirname.
    
    Args:
        :dirname (str): Directory name containing the optimisation results
        :all_comp (list[int]): All complexity of equation to consider
        :nx (int): The number of data points to be used in the mock
        :all_true_eq (list[str] or None): List of variants of the true equation to find
        
    Returns:
        :None
    
    """

    nkeep = 10

    # Get data from MDL a la Bartlett et al. 2022
    fun, res, params, store_comp = process_data(dirname, 'final_', all_comp)

    # METHOD 1: Max likelihood
    m = np.argsort(res[:,2], kind='stable')
    m1_f0, m1_f1, m1_f2 = fun[m[0]], fun[m[1]], fun[m[2]]
    m1_fun = [None] * nkeep
    m1_loss = np.ones(nkeep) * np.nan
    for i in range(min(nkeep, len(m))):
        m1_fun[i] = fun[m[i]]
        m1_loss[i] = res[m[i],2]
    if all_true_eq is None:
        idx = []
    else:
        idx = [fun.index(true_eq) for true_eq in all_true_eq if true_eq in fun]
    if len(idx) == 0:
        m1_ltrue = np.nan
        m1_ftrue = None
    else:
        all_m1_ltrue = np.array([res[i,2] for i in idx])
        m1_ltrue = np.amin(all_m1_ltrue)
        idx = idx[np.argmin(all_m1_ltrue)]
        m1_ftrue = fun[idx]
    if m1_ftrue not in m1_fun[:2]:
        print('\nMethod 1', dirname, [fun[mm] for mm in m[:4]])

    # METHOD 2: Max dL/dc (a la PySR)
    new_fun = [None] * len(all_comp)
    loss = np.zeros(len(all_comp))
    for i, c in enumerate(all_comp):
        m = store_comp==c
        idx = np.arange(res.shape[0])[m]
        ll = res[m,2]
        loss[i] = np.amin(ll)
        new_fun[i] = fun[idx[np.argmin(ll)]]
    loss = (loss[1:] - loss[:-1]) / (all_comp[1:] - all_comp[:-1])
    new_fun = new_fun[1:]
    m = np.argsort(loss)
    m2_fun = [None] * nkeep
    m2_loss = np.ones(nkeep) * np.nan
    for i in range(min(nkeep, len(m))):
        m2_fun[i] = new_fun[m[i]]
        m2_loss[i] = loss[m[i]]
    if all_true_eq is None:
        idx = []
    else:
        idx = [new_fun.index(true_eq) for true_eq in all_true_eq if true_eq in new_fun]
    if len(idx) == 0:
        m2_ltrue = np.nan
        m2_ftrue = None
    else:
        all_m2_ltrue = np.array([loss[i] for i in idx])
        m2_ltrue = np.amin(all_m2_ltrue)
        idx = idx[np.argmin(all_m2_ltrue)]
        m2_ftrue = new_fun[idx]
    if m2_ftrue not in m2_fun[:2]:
        print('\nMethod 2', dirname, [new_fun[mm] for mm in m[:4]])

    # METHOD 3: MDL (a la Bartlett et al. 2022)
    m = np.argsort(res[:,0], kind='stable')
    m3_fun = [None] * nkeep
    m3_loss = np.ones(nkeep) * np.nan
    for i in range(min(nkeep, len(m))):
        m3_fun[i] = fun[m[i]]
        m3_loss[i] = res[m[i],0]
    if all_true_eq is None:
        idx = []
    else:
        idx = [fun.index(true_eq) for true_eq in all_true_eq if true_eq in fun]
    if len(idx) == 0:
        m3_ltrue = np.nan
        m3_ftrue = None
    else:
        all_m3_ltrue = np.array([res[i,0] for i in idx])
        m3_ltrue = np.amin(all_m3_ltrue)
        idx = idx[np.argmin(all_m3_ltrue)]
        m3_ftrue = fun[idx]
    if m3_ftrue not in m3_fun[:2]:
        print('\nMethod 3', dirname, [fun[mm] for mm in m[:4]])

    # Now get language model data including the constant terms
    fun, res, params, store_comp = process_data(dirname, 'final_katz_2_', all_comp)
    b = 1 / np.sqrt(nx)
    m = params!=0
    p = np.sum(m, axis=1)
    nup = np.exp(1 - np.log(3))

    # METHOD 4: MDL with language model prior 
    m = np.argsort(res[:,0], kind='stable')
    m4_fun = [None] * nkeep
    m4_loss = np.ones(nkeep) * np.nan
    for i in range(min(nkeep, len(m))):
        m4_fun[i] = fun[m[i]]
        m4_loss[i] = res[m[i],0]
    if all_true_eq is None:
        idx = []
    else:
        idx = [fun.index(true_eq) for true_eq in all_true_eq if true_eq in fun]
    if len(idx) == 0:
        m4_ltrue = np.nan
        m4_ftrue = None
    else:
        all_m4_ltrue = np.array([res[i,0] for i in idx])
        m4_ltrue = np.amin(all_m4_ltrue)
        idx = idx[np.argmin(all_m4_ltrue)]
        m4_ftrue = fun[idx]
    if m4_ftrue not in m4_fun[:2]:
        print('\nMethod 4', dirname, [fun[mm] for mm in m[:4]])

    # METHOD 5: MDL with FBF and Language Model Prior
    loss = (1 - b) * res[:,2] - p/2 * np.log(b) + res[:,4] + p/2 * np.log(2 * np.pi * nup)
    m = np.argsort(loss, kind='stable')
    m5_fun = [None] * nkeep
    m5_loss = np.ones(nkeep) * np.nan
    for i in range(min(nkeep, len(m))):
        m5_fun[i] = fun[m[i]]
        m5_loss[i] = loss[m[i]]
    if all_true_eq is None:
        idx = []
    else:
        idx = [fun.index(true_eq) for true_eq in all_true_eq if true_eq in fun]
    if len(idx) == 0:
        m5_ltrue = np.nan
        m5_ftrue = None
    else:
        all_m5_ltrue = np.array([loss[i] for i in idx])
        m5_ltrue = np.amin(all_m5_ltrue)
        idx = idx[np.argmin(all_m5_ltrue)]
        m5_ftrue = fun[idx]
    if m5_ftrue not in m5_fun[:2]:
        print('\nMethod 5', dirname, [fun[mm] for mm in m[:4]])

    # Now get language model data excluding the constant temrs
    fun, res, params, store_comp = process_data(dirname, 'final_noconst_katz_2_', all_comp)
    b = 1 / np.sqrt(nx)
    m = params!=0
    p = np.sum(m, axis=1)
    nup = np.exp(1 - np.log(3))

    # METHOD 6: Evidence with FBF and Language Model Priors
    loss = (1 - b) * res[:,2] - p/2 * np.log(b) + res[:,4] + p/2 * np.log(2 * np.pi * nup)
    m = np.argsort(loss, kind='stable')
    m6_fun = [None] * nkeep
    m6_loss = np.ones(nkeep) * np.nan
    for i in range(min(nkeep, len(m))):
        m6_fun[i] = fun[m[i]]
        m6_loss[i] = loss[m[i]]
    for mm in m[:10]:
        print(fun[mm], loss[mm])
    if all_true_eq is None:
        idx = []
    else:
        idx = [fun.index(true_eq) for true_eq in all_true_eq if true_eq in fun]
    if len(idx) == 0:
        m6_ltrue = np.nan
        m6_ftrue = None
    else:
        all_m6_ltrue = np.array([loss[i] for i in idx])
        m6_ltrue = np.amin(all_m6_ltrue)
        idx = idx[np.argmin(all_m6_ltrue)]
        m6_ftrue = fun[idx]

    # PRINT RESULTS TO FILE
    with open(dirname + '/selection_summary.csv', "w") as f:
        writer = csv.writer(f, delimiter=';')
        if all_true_eq is None:
            writer.writerow(['Method'] + ['f%i'%i for i in range(nkeep)] + ['loss%i'%i for i in range(nkeep)])
            writer.writerow([1] + m1_fun + list(m1_loss))
            writer.writerow([2] + m2_fun + list(m2_loss))
            writer.writerow([3] + m3_fun + list(m3_loss))
            writer.writerow([4] + m4_fun + list(m4_loss))
            writer.writerow([5] + m5_fun + list(m5_loss))
            writer.writerow([6] + m6_fun + list(m6_loss))
        else:
            writer.writerow(['Method'] + ['f%i'%i for i in range(nkeep)] + ['loss%i'%i for i in range(nkeep)] + ['ftrue', 'losstrue'])
            writer.writerow([1] + m1_fun + list(m1_loss) + [m1_ftrue, m1_ltrue])
            writer.writerow([2] + m2_fun + list(m2_loss) + [m2_ftrue, m2_ltrue])
            writer.writerow([3] + m3_fun + list(m3_loss) + [m3_ftrue, m3_ltrue])
            writer.writerow([4] + m4_fun + list(m4_loss) + [m4_ftrue, m4_ltrue])
            writer.writerow([5] + m5_fun + list(m5_loss) + [m5_ftrue, m5_ltrue])
            writer.writerow([6] + m6_fun + list(m6_loss) + [m6_ftrue, m6_ltrue])

    return


def main():
    """
    Run the benchmarks
    """

    benchmarks = {
        'nguyen_6':['sin(x) + sin(x + x^2)', [-1, 1]],
        'nguyen_8':['sqrt(x)', [0, 4]],
        'korns_1':['1.57 + 2.43 * x', [-50, 50]],
        'korns_4':['-2.3 + 0.13 * sin(x)', [-50, 50]],
        'korns_6':['1.3 + 0.13 * sqrt(x)', [0, 50]],
        'korns_7':['213.80940889 * (1 - exp(-0.54723748542 * x))', [0, 50]],
        'korns_11':['6.87 + 11 * cos(2.73 * x^3)',[0, 50]],
        'keijzer_1':['0.3 * x * sin( 2 * pi * x)', [-1, 1]]
    }
    all_true_eq = {
        'nguyen_8': ['sqrt(x)'],
        'korns_1': ['a0 + a1*x', 'a0 + x/a1', 'a0*(-a1 + x)', 'a0*(a1 - x)', 'a0*(a1 + x)', 
                    '(-a0 + x)/pow(Abs(a1),(1/4))'],
        'korns_4': ['a0 + a1*sin(x)', 'a0 - sin(x)*cos(a1)', 'a0 + sin(x)*cos(a1)'],
        'korns_6': ['a0 + a1*sqrt(x)', 'a0 + sqrt(x)*sqrt(Abs(a1))', 'a0*(a1 - sqrt(x))'],
        'korns_7': ['a0 - pow(Abs(a1),x)', 'a0 + pow(Abs(a1),x)/a2', 'a0 + a1/pow(Abs(a2),x)', 'a0*(a1 - pow(Abs(a2),x))',
                    'a0 + a1*pow(Abs(a2),x)']
    }


    all_N = [10, 30, 100, 300, 1000, 3000, 10000]
    #all_N = [100]
    all_sigx = [0.5]
    nsamp = 5
    #all_name = ['korns_4']
    #all_name = ['nguyen_8', 'korns_1', 'korns_6', 'korns_7']
    #all_name = ['korns_6', 'korns_7']
    all_name = ['korns_7']
    all_comp = np.arange(1, 8)
    
    do_make_mocks = False
    do_fit_mocks = False
    do_language_model = False
    do_process_mocks = True

    # All possible N-samp combinations
    combo = list(itertools.product(all_N, list(np.arange(nsamp)))) 

    # Split among ranks for generation
    data_start, data_end = get_split_idx(len(combo))

    for name in all_name:

        print_text(name)

        f, x_range = benchmarks[name]
        true_eq = all_true_eq[name]

        for frac_sigx in all_sigx:

            sig = get_sig(f, x_range, frac_sigx)

            # Make mocks
            if do_make_mocks:
                for nx, samp_num in combo[data_start:data_end]:
                    make_data(name, f, x_range, nx, frac_sigx, samp_num, sig, make_fig=False)
            comm.Barrier()

            # Fit mocks
            if do_fit_mocks:
                for nx, samp_num in combo:
                    for comp in all_comp:
                        print_text(f'FITTING {name}, {frac_sigx}, {nx}, {samp_num}, {comp}')
                        fit_mocks(name, nx, frac_sigx, samp_num, comp)

            # Apply language model
            if do_language_model:
                for nx, samp_num in combo:
                    for comp in all_comp:
                        print_text(f'LANGUAGE {name}, {frac_sigx}, {nx}, {samp_num}, {comp}')
                        _apply_language_prior(name, nx, frac_sigx, samp_num, comp)

            # Process mocks:
            if do_process_mocks:
                print_text(f'PROCESSING RESULTS {name}, {frac_sigx}')
                for nx, samp_num in combo[data_start:data_end]:
                    _process_fit(name, true_eq, nx, frac_sigx, samp_num, all_comp)

    return


if __name__ == "__main__":
    main()
