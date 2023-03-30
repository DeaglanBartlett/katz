import os
import sys
import numpy as np
import itertools
from mpi4py import MPI
from katz.prior import KatzPrior

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_indices(n):

    nLs = int(np.ceil(n / float(size)))       # Number of lines per file for given thread

    while nLs*(size-1) > n:
        if rank==0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank==0:
        print("Total number of functions: ", n, flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank==size-1:
        data_end = n
        
    return data_start, data_end

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

    data_start, data_end = get_indices(len(fcn_list))
    
    return fcn_list[data_start:data_end], data_start, data_end
    
    
def get_logconst(comp, dirname, overwrite=False):

    outname = dirname + f'/compl_{comp}/logconst_{comp}.txt'
    if os.path.isfile(outname) and (not overwrite):
        return

    fname = dirname + f'/compl_{comp}/trees_{comp}.txt'
    with open(fname, "r") as f:
        tree_list = f.read().splitlines()
    data_start, data_end = get_indices(len(tree_list))
    
    tree_list = tree_list[data_start:data_end]
    logconst = [None] * len(tree_list)
    for i in range(len(tree_list)):
        tree = tree_list[i].split("'")
        tree = [tt for tt in tree if tt not in ["[", "]", " ", ", "]]
        
        n = np.array([int(tt) for tt in tree if tt.lstrip("-").isdigit()])  # Integers
        n[n==0] = 1  # So we have log(1) for 0 instead of log(0)
        logconst[i] = np.sum(np.log(np.abs(n)))
    
    logconst = comm.gather(logconst, root=0)
    if rank == 0:
        logconst = np.array(list(itertools.chain(*logconst)))
        np.savetxt(outname, logconst)
    
    return
    
def compute_logprior(comp, n, basis_functions, dirname, in_eqfile, out_eqfile, overwrite=False):

    outname_code = dirname + f'/compl_{comp}/katz_codelen_{n}_{comp}.txt'
    outname_prior = dirname + f'/compl_{comp}/katz_logprior_{n}_{comp}.txt'
    if os.path.isfile(outname_code) and os.path.isfile(outname_prior) and (not overwrite):
        return

    fcn_list_proc, _, _ = get_functions(comp, dirname)

    kp = KatzPrior(n, basis_functions, in_eqfile, out_eqfile)
    logprior_proc = np.empty(len(fcn_list_proc))
    
    for i, eq in enumerate(fcn_list_proc):
        if 'zoo' in eq:
            logprior_proc[i] = np.nan
        else:
            try:
                logprior_proc[i] = kp.logprior(eq.strip())
            except Exception as e:
                print("BAD EQ:", eq.strip(), e)
                logprior_proc[i] = np.nan

    logprior = comm.gather(logprior_proc, root=0)
    if rank == 0:
        logprior = np.array(list(itertools.chain(*logprior)))
        np.savetxt(outname_prior, logprior)
        logconst = np.loadtxt(dirname + f'/compl_{comp}/logconst_{comp}.txt')
        np.savetxt(outname_code, - logprior + logconst)# will use as replacement to aifeyn term, so need to sum these

    return
    
    
def main():

    basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                ["+", "-", "*", "/", "pow"]]
    in_eqfile = '../data/FeynmanEquations.csv'
    out_eqfile = '../data/NewFeynman.csv'
    n = 1

    #dirname = '../../ESR/esr/function_library/core_maths/'
    dirname = '../../ESR/esr/function_library/new_osc_maths/'
    for comp in range(1, 7):
        if rank == 0:
            print('\nCOMPLEXITY:', comp, flush=True)
        get_logconst(comp, dirname)
        compute_logprior(comp, n, basis_functions, dirname, in_eqfile, out_eqfile, overwrite=True)
    
    
    return
    
if __name__ == "__main__":
    main()
    
    
    
