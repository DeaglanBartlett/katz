from esr.fitting.likelihood import PanthLikelihood
import numpy as np
import pandas as pd
from mpi4py import MPI
import sympy

from fit_benchmark import apply_language_prior, process_fit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

likelihood = PanthLikelihood()
all_comp = np.arange(1, 11)
dirname = '/mnt/zfsusers/deaglan/symbolic_regression/function_prior/ESR/esr//fitting//output//output_panth_dimful/'

do_language = False
do_process = False
print_results = True

if do_language:
    for comp in all_comp:
        apply_language_prior(likelihood, comp, tmax=5)
        comm.Barrier()
if rank == 0:
    #dirname = likelihood.out_dir + '/'

    if do_process:
        nx = len(likelihood.xvar)
        #nx = 1590
        process_fit(dirname, all_comp, nx)

    if print_results:
        df = pd.read_csv(dirname + 'selection_summary.csv', delimiter=';')
        print(df.shape)
        print('\nYOU WILL HAVE TO INSPECT THESE YOURSELF AS DUPLICATES MAY REMAIN\n')
        for m in range(df.shape[0]):
            res = []
            for i in range(8):
                f = df['f%i'%i][m]
                if str(f) != str(None):
                    _, eq, _ = likelihood.run_sympify(f, try_integration=False)
                    f = sympy.latex(eq)
                if m == 1:
                    #res = ['%s & %.2e'%(eq, df['loss%i'%i][m]) for i in range(3)]
                    res.append('$%s$ & %.2e'%(f, df['loss%i'%i][m]))
                else:
                    #res = ['%s & %.2f'%(eq, df['loss%i'%i][m]) for i in range(3)]
                    res.append('$%s$ & %.2f'%(f, df['loss%i'%i][m]))
            res = '%i & '%(m+1) + ' & '.join(res) + ' \\\\'
            print(res)
            print('\\hline')
            
        
