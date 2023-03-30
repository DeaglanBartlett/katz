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


def get_sig(name, f, x_range, frac_sigx):

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


def get_split_idx(L):

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


def process_fit(name, nx, frac_sigx, samp_num, all_comp):
    """
    Currently this deals with methods which do not involve the function prior
    """

    # (1) Load the data
    fname = f'{name}_{nx}_{frac_sigx}_{samp_num}'
    dirname = f'output/output_{fname}/'
    print(dirname)

    res = []
    fun = []
    params = []
    store_comp = []
    for i, compl in enumerate(all_comp):
        fname = dirname + 'final_%i.dat'%compl
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
    imax = -1   # MAYBE WANT TO CHANGE THIS
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

    # METHOD 1: Max likelihood
    m = np.argsort(res[:,2], kind='stable')
    m1_f0, m1_f1 = fun[m[0]], fun[m[1]]
    m1_l0, m1_l1 = res[m[:2],2]

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
    m2_f0, m2_f1 = new_fun[m[0]], new_fun[m[1]]
    m2_l0, m2_l1 = res[m[:2],2]

    # METHOD 3: MDL (a la Bartlett et al. 2022)
    m = np.argsort(res[:,0], kind='stable')
    m3_f0, m3_f1 = fun[m[0]], fun[m[1]]
    m3_l0, m3_l1 = res[m[:2],0]

    # PRINT RESULTS TO FILE
    with open(dirname + '/selection_summary.csv', "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Method', 'f0', 'loss0', 'f1', 'loss1'])
        writer.writerow([1, m1_f0, m1_l0, m1_f1, m1_l1])
        writer.writerow([2, m2_f0, m2_l0, m2_f1, m2_l1])
        writer.writerow([3, m3_f0, m3_l0, m3_f1, m3_l1])

    return


def main():

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
    #all_N = [10, 30, 100, 300, 1000]
    all_N = [10]
    all_sigx = [0.5]
    nsamp = 5
    all_name = ['korns_4']
    all_comp = np.arange(1, 8)
    
    do_make_mocks = False
    do_fit_mocks = False
    do_process_mocks = True

    # All possible N-samp combinations
    combo = list(itertools.product(all_N, list(np.arange(nsamp)))) 

    # Split among ranks for generation
    data_start, data_end = get_split_idx(len(combo))

    for name in all_name:

        print_text(name)

        f, x_range = benchmarks[name]

        for frac_sigx in all_sigx:

            sig = get_sig(name, f, x_range, frac_sigx)

            # Make mocks
            if do_make_mocks:
                for nx, samp_num in combo[data_start:data_end]:
                    make_data(name, f, x_range, nx, frac_sigx, samp_num, sig, make_fig=False)
            comm.Barrier()

            # Fit mocks
            if do_fit_mocks:
                for nx, samp_num in combo:
                    for comp in all_comp:
                        print_text(f'STARTING {name}, {frac_sigx}, {nx}, {samp_num}, {comp}')
                        fit_mocks(name, nx, frac_sigx, samp_num, comp)

            # Process mocks:
            if do_process_mocks:
                print_text(f'PROCESSING RESULTS {name}, {frac_sigx}')
                for nx, samp_num in combo[data_start:data_end]:
                    process_fit(name, nx, frac_sigx, samp_num, all_comp)

    return


if __name__ == "__main__":
    main()

"""
TO DO
- Add in language model prior to fitting and processing
"""
