import katz.esr_prior
import esr.generation.duplicate_checker
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    """
    Run the prior generation for ESR functions
    """

    basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
                ["+", "-", "*", "/", "pow"]]
#    in_eqfile = '../data/FeynmanEquations.csv'
#    out_eqfile = '../data/NewFeynman.csv'
    in_eqfile = '../data/PhysicsEquations.csv'
    out_eqfile = '../data/NewPhysics.csv'
    input_delimiter = ';'
    
    all_comp = [1, 2, 3, 4]
    all_n = [1, 2, 3]
    
    use_tree = True
    
    # Generate functions needed for examples
    runname = 'core_maths'
    for comp in all_comp:
        esr.generation.duplicate_checker.main(runname, comp)

    # Compute katz prior probabilities
    dirname = f'../../ESR/esr/function_library/{runname}/'
    for comp in all_comp:
        if rank == 0:
            print('\nCOMPLEXITY:', comp, flush=True)
        katz.esr_prior.get_logconst(comp, dirname, overwrite=True)
        for n in all_n:
            if rank == 0:
                print('\nN-GRAM:', n, flush=True)
            katz.esr_prior.compute_logprior(comp, n, basis_functions, dirname, in_eqfile, out_eqfile, overwrite=True, input_delimiter=input_delimiter, use_tree=use_tree)
    
    
    return
    
if __name__ == "__main__":
    main()
