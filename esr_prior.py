import sys
import numpy as np
import itertools
from mpi4py import MPI
from katz.prior import KatzPrior

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_functions(comp, dirname):

    if comp==8:
        sys.setrecursionlimit(2000)
    elif comp==9:
        sys.setrecursionlimit(2500)
    elif comp==10:
        sys.setrecursionlimit(3000)
    
    fname = dirname + f'/compl_{comp}/all_equations_{comp}.txt'
    
    if rank==0:
        print("Number of cores:", size, flush=True)

    with open(fname, "r") as f:
        fcn_list = f.readlines()

    nLs = int(np.ceil(len(fcn_list) / float(size)))       # Number of lines per file for given thread

    while nLs*(size-1) > len(fcn_list):
        if rank==0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank==0:
        print("Total number of functions: ", len(fcn_list), flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank==size-1:
        data_end = len(fcn_list)
    
    return fcn_list[data_start:data_end], data_start, data_end
    
    
def compute_logprior(comp, n, basis_functions, dirname, in_eqfile, out_eqfile):

    fcn_list_proc, _, _ = get_functions(comp, dirname)

    kp = KatzPrior(n, basis_functions, in_eqfile, out_eqfile)
    logprior_proc = np.empty(len(fcn_list_proc))
    
    for i, eq in enumerate(fcn_list_proc):
        if 'zoo' in eq:
            logprior_proc[i] = np.nan
        else:
            try:
                logprior_proc[i] = kp.logprior(eq.strip())
            except:
                print("BAD EQ:", eq.strip())

    logprior = comm.gather(logprior_proc, root=0)
    if rank == 0:
        logprior = np.array(list(itertools.chain(*logprior)))
        outname = dirname + f'/compl_{comp}/katz_logprior_{n}_{comp}.txt'
        np.savetxt(outname, logprior)

    return
    
    
def main():

    basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                ["+", "-", "*", "/", "pow"]]
    in_eqfile = 'data/FeynmanEquations.csv'
    out_eqfile = 'data/NewFeynman.csv'
    n = 2

    dirname = '../function_sets/core_maths/'
    for comp in range(1, 7):
#    for comp in [7, 8]:
        if rank == 0:
            print('\nCOMPLEXITY:', comp, flush=True)
        compute_logprior(comp, n, basis_functions, dirname, in_eqfile, out_eqfile)
    
    return
    
if __name__ == "__main__":
    main()
    
    
    
