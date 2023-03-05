import numpy as np
from .back_off import BackOff
from .process_equations import standardise_file, SymbolCoder

class KatzPrior:

    def __init__(self, n, basis_functions, in_eqfile, out_eqfile):
        """Class to evaluate the probability of a function based on an n-gram Katz back-off model
        
        Args:
            :n (int): The length of the n-tuples to consider
            :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
            :in_eqfile (str): Name of file containing the equations to study
            :out_eqfile (str): Name of file to output the standardised equations to
            
        Returns:
            KatzPrior: Prior model to find prior of a function given a previous set of equations
        
        """
    
        self.n = n
        self.all_eq, self.maxvar = standardise_file(in_eqfile, out_eqfile)
        self.basis_functions = [list(set(basis_functions[0] + ["a"] + [f"x{i}" for i in range(self.maxvar)])),  # type0
                                basis_functions[1],  # type1
                                basis_functions[2]]  # type2
        self.coder = SymbolCoder(self.basis_functions)
        data = self.coder.process_all_equations(n, self.all_eq, self.maxvar)
        self.backoff = BackOff(data)
        
    def logprior(self, eq):
        """
        Compute the natural logarithm of the prior of a given equation
        
        Args:
            :eq (str): The equation to find the prior probability of
            
        Returns:
            :p (float): The natural logarithm of the prior of the supplied equation
        """
        t = self.coder.process_all_equations(self.n, [eq], self.maxvar)
        p = np.array([self.backoff.get_pbo(tt[-1], tt[:-1]) for tt in t])
        p = np.sum(np.log(p))
        return p
