import os
import sys
import numpy as np
import itertools
import warnings
from mpi4py import MPI
from katz.prior import KatzPrior

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_indices(n):
    """
    Find the indices of an array of length n which should be considered
    by this rank. For output data_start, data_end, the rank considers the
    entries array[data_start:data_end] where len(array) = n.
    
    Args:
        :n (int): The length of the array we wish to split among ranks.
        
    Returns:
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank.
        
    """

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
    """
    Load all functions from a given directory at a certain complexity
    and divide these among the ranks.
    
    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the function lists
        
    Returns:
        :fcn_list (list[str]): The list of functions considered by this rank
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank
        
    """

    recursion_limits = {8: 2000, 9: 2500, 10: 3000}
    sys.setrecursionlimit(recursion_limits.get(comp, sys.getrecursionlimit()))
    
    fname = dirname + f'/compl_{comp}/all_equations_{comp}.txt'
    
    if rank==0:
        print("Number of cores:", size, flush=True)

    with open(fname, "r") as f:
        fcn_list = f.readlines()

    data_start, data_end = get_indices(len(fcn_list))
    
    return fcn_list[data_start:data_end], data_start, data_end
    
    
def get_trees(comp, dirname):
    """
    Load the tree files for ESR for a given complexity and convert to
    lists of strings. These are split among ranks.
    
    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the ESR functions
        
    Returns:
        :tree_list (list): The list of lists giving trees to be considered by this rank
    
    """
    
    fname = dirname + f'/compl_{comp}/trees_{comp}.txt'
    with open(fname, "r") as f:
        tree_list = f.read().splitlines()
    data_start, data_end = get_indices(len(tree_list))
    
    tree_list = tree_list[data_start:data_end]
    for i in range(len(tree_list)):
        tree = tree_list[i].split("'")
        tree_list[i] = [tt for tt in tree if tt not in ["[", "]", " ", ", "]]
        
    return tree_list
    
    
    
def get_logconst(comp, dirname, overwrite=False):
    """
    Determine the sum(log(c_i)) for constants c_i appearning in all equations
    and save results to file.
    
    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the function lists
        :overwrite (bool): Whether to overwrite the constants file if it already exists
    
    Returns:
        :None
        
    """

    outname = dirname + f'/compl_{comp}/logconst_{comp}.txt'
    if os.path.isfile(outname) and (not overwrite):
        return

    tree_list = get_trees(comp, dirname)
    logconst = [None] * len(tree_list)
    for i, tree in enumerate((tree_list)):
        n = np.array([int(tt) for tt in tree if tt.lstrip("-").isdigit()])  # Integers
        n[n==0] = 1  # So we have log(1) for 0 instead of log(0)
        logconst[i] = np.sum(np.log(np.abs(n)))
    
    logconst = comm.gather(logconst, root=0)
    if rank == 0:
        logconst = np.array(list(itertools.chain(*logconst)))
        np.savetxt(outname, logconst)
    
    return
    
def compute_logprior(comp, n, basis_functions, dirname, in_eqfile, out_eqfile, overwrite=False, input_delimiter=',', use_tree=False):
    """
    Compute the log of the function prior for all functions at a give complexity
    given a corpus of equations. Saves results to two files: katz_logprior_{n}_{comp}.txt
    contains just -log(prior) values, and katz_logprior_{n}_{comp}.txt also contains
    the sum(log(c_i)) terms for constants c_i which appear in the equations
    
    Args:
        :comp (int): The complexity of function to consider
        :n (int): The length of the n-tuples to use for back-off model
        :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
        :dirname (str): The directory containing the function lists
        :in_eqfile (str): Name of file containing the equations to use as corpus
        :out_eqfile (str): Name of file to output standardised corpus equations to
        :overwrite (bool): Whether to overwrite files if they already exist
        :input_delimiter (str): The delimiter used in the input csv file
        :use_tree (bool, default=False): Whether to compute the prior using the tree representation from file (True) or using the string representation (False) and convert this to a tree automatically.
        
    Returns:
        :None
    
    """

    outname_code = dirname + f'/compl_{comp}/katz_codelen_{n}_{comp}'
    outname_prior = dirname + f'/compl_{comp}/katz_logprior_{n}_{comp}'
    if use_tree:
        outname_code += '_tree'
        outname_prior += '_tree'
    outname_code += '.txt'
    outname_prior += '.txt'
        
    if os.path.isfile(outname_code) and os.path.isfile(outname_prior) and (not overwrite):
        return

    if use_tree:
        fcn_list_proc = get_trees(comp, dirname)
    else:
        fcn_list_proc, _, _ = get_functions(comp, dirname)

    kp = KatzPrior(n, basis_functions, in_eqfile, out_eqfile, input_delimiter=input_delimiter)
    logprior_proc = np.empty(len(fcn_list_proc))
    
    for i, eq in enumerate(fcn_list_proc):
        if 'zoo' in eq:
            logprior_proc[i] = np.nan
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if use_tree:
                        logprior_proc[i] = kp.logprior(eq)
                    else:
                        logprior_proc[i] = kp.logprior(eq.strip())
            except Exception as e:
                if use_tree:
                    print("BAD EQ:", eq, e)
                else:
                    print("BAD EQ:", eq.strip(), e)
                logprior_proc[i] = np.nan

    logprior = comm.gather(logprior_proc, root=0)
    if rank == 0:
        logprior = np.array(list(itertools.chain(*logprior)))
        np.savetxt(outname_prior,  - logprior)
        logconst = np.loadtxt(dirname + f'/compl_{comp}/logconst_{comp}.txt')
        np.savetxt(outname_code, - logprior + logconst)# will use as replacement to aifeyn term, so need to sum these

    return
