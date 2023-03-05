import numpy as np
from .back_off import BackOff
from .process_equations import standardise_file, SymbolCoder

class KatzPrior:

    def __init__(self, n, basis_functions, in_eqfile, out_eqfile):
    
        self.n = n
        self.all_eq, self.maxvar = standardise_file(in_eqfile, out_eqfile)
        self.basis_functions = [list(set(basis_functions[0] + ["a"] + [f"x{i}" for i in range(self.maxvar)])),  # type0
                                basis_functions[1],  # type1
                                basis_functions[2]]  # type2
        self.coder = SymbolCoder(self.basis_functions)
        data = self.coder.process_all_equations(n, self.all_eq, self.maxvar)
        self.backoff = BackOff(data)
        
    def logprior(self, eq):
        t = self.coder.process_all_equations(self.n, [eq], self.maxvar)
        p = np.array([self.backoff.get_pbo(tt[-1], tt[:-1]) for tt in t])
        p = np.sum(np.log(p))
        return p
