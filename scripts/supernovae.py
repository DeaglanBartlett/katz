import esr.fitting.combine_DL
from esr.fitting.likelihood import PanthLikelihood
import os
from prettytable import PrettyTable
import csv
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def print_results(fname, nrow):

    with open(fname, "r") as f:
        reader = csv.reader(f, delimiter=';')
        data = [row for row in reader]

        res = [d[2:7] for d in data]
        fun = [d[1] for d in data]
        params = [d[7:] for d in data]
        store_comp = [compl] * len(data)
        
    res = np.array(res, dtype=float)
    params = np.array(params, dtype=float)
    store_comp = np.array(store_comp, dtype=int)
        
    # Sort by DL
    m = np.argsort(res[:,0], kind='stable')
    res = res[m,:]
    params = params[m,:]
    store_comp = store_comp[m]
    fun = [fun[i] for i in m]

    ptab = PrettyTable()
    ptab.field_names = ["Rank", "Function", "Complexity", "Res", "Fun", "Param", "DL", "a0", "a1", "a2"]
    for i in range(nrow):
        ptab.add_row([i+1, fun[i], store_comp[i], '%.2f'%res[i,2], '%.2f'%res[i,4], '%.2f'%res[i,3], '%.2f'%res[i,0], '%.2f'%params[i,0], '%.2f'%params[i,1], '%.2f'%params[i,2]])
    print(ptab)
    
    return

#like_dir = os.getcwd() + '/../test_results/'
compl = 4
nrow = 10

likelihood = PanthLikelihood()

# Change the path
#likelihood.like_dir = like_dir
likelihood.fnprior_prefix = f"katz_codelen_2_"
likelihood.combineDL_prefix = "combine_DL_katz_2_"
likelihood.final_prefix = "final_katz_2_"

#for compl in range(1, 6):
for compl in [7, 8]:
    esr.fitting.combine_DL.main(compl, likelihood)

comm.Barrier()
quit()
if rank == 0:
    old_fname = likelihood.out_dir + f'/final_{compl}.dat'
    print_results(old_fname, nrow)
    new_fname = likelihood.out_dir + '/' + likelihood.final_prefix + f'{compl}.dat'
    print_results(new_fname, nrow)


        
