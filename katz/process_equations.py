import pandas as pd
import numpy as np
import string
import csv
import sympy
import itertools
import esr.generation.generator as generator

def split_by_punctuation(s):
    pun = string.punctuation.replace('_', '') # allow underscores in variable names
    pun = pun + ' '
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    split_str = [s[:where_pun[0]]]
    for i in range(len(where_pun)-1):
        split_str += [s[where_pun[i]]]
        split_str += [s[where_pun[i]+1:where_pun[i+1]]]
    split_str += [s[where_pun[-1]]]
    if where_pun[-1] != len(s) - 1:
        split_str += [s[where_pun[-1]+1:]]
    return split_str
    
    
def standardise_file(in_name, out_name):

    df = pd.read_csv(in_name)
    
    maxvar = int(df['# variables'].max())
    if maxvar > 10:
        raise NotImplementedError(f"Cannot have more than 10 input variables: you have {maxvar}")
        
    all_eq = []
    
    with open(out_name, 'w') as f:
        csvwriter = csv.writer(f)
        
        csvwriter.writerow(['Filename', 'Number', 'Old Formula', 'New Formula'])
    
        for index, row in df.iterrows():
        
            if not np.isfinite(row['# variables']):
                continue
            eq = row['Formula']
            
            # If equation already has variable 'x{i}" then we don't need to replace it
            # Note: At least Eqs I.18.12, I.18.14 and II.37.1 in AIFeynman has wrong number of vars
            # The next two lines fix this
            vars = [row[f'v{i+1}_name'] for i in range(maxvar)]
            vars = [v for v in vars if isinstance(v, str)]
                
            names = [f'x{i}' for i in range(len(vars))]
            to_change = list(sorted(set(vars) - set(names), key=vars.index))
            to_sub = list(sorted(set(names) - (set(vars) - set(to_change)), key=names.index))
            
            # Must split by punctuation to avoid replacing e.g. "t" in "sqrt" if we have a variable "t"
            sub_dict = dict(zip(to_change,to_sub))
            split_eq = split_by_punctuation(eq)
            for i, s in enumerate(split_eq):
                if s in to_change: split_eq[i] = sub_dict[s]
            eq = ''.join(split_eq)
                
            all_eq.append(eq)
            
            csvwriter.writerow([row['Filename'], row['Number'], row['Formula'], eq])

    return all_eq, maxvar
    
    
class SymbolCoder:

    def __init__(self, basis_functions):
        self.basis_functions = basis_functions
        
        self.sympy_numerics = ['Number', 'Float', 'Rational', 'Integer', 'AlgebraicNumber',
                    'NumberSymbol', 'RealNumber', 'igcd', 'ilcm', 'seterr', 'Zero',
                    'One', 'NegativeOne', 'Half', 'NaN', 'Infinity', 'NegativeInfinity',
                    'ComplexInfinity', 'Exp1', 'ImaginaryUnit', 'Pi', 'EulerGamma',
                    'Catalan', 'GoldenRatio', 'TribonacciConstant', 'mod_inverse']
        self.ops = [str(None), 'a', 'x'] + basis_functions[1] + basis_functions[2]
        #self.code = [(t,) for t in self.ops] + list(itertools.product(self.ops, repeat=2)) # USE WHEN WE DO SIBLINGS
        self.code = [t for t in self.ops]
        self.code = dict(zip(self.code, np.arange(len(self.code)).astype(str)))
        
        
    def equation2ntuples(self, n, eq, locs):

        expr, nodes, c = generator.string_to_node(eq, self.basis_functions, locs=locs)
        lin = nodes.get_lineage()
        
        # USE WHEN WE DO SIBLINGS
        #lin = nodes.get_sibling_lineage()
        #sib = nodes.get_siblings()

        ntuples = [None] * len(lin)
        for i, t in enumerate(lin):
            if len(t) >= n:
                x = t[-n:]
            else:
                x = tuple([None]*(n-len(t)) + list(t))

            ntuples[i] = tuple([self.op2codeword(tt) for tt in x])
            #for i in range(n-1):
            #    x[i] = (x[i],)
            #x[-1] = (x[-1],)

        return ntuples
        
    def process_all_equations(self, n, all_eq, maxvar):

        x = sympy.symbols([f'x{i}' for i in range(maxvar)], real=True)
        locs = {f'x{i}':x[i] for i in range(len(x))}
        
        ntuples = []
        for eq in all_eq:
            ntuples += self.equation2ntuples(n, eq, locs)

        return ntuples
    
        
    def op2str(self, op):
        if op is None:
            return str(None)
        elif op in self.sympy_numerics:
            return 'a'
        elif op == 'Symbol':
            return 'x'
        elif op == 'Add' and '+' in self.basis_functions[2]:
            return '+'
        elif op == 'Mul' and '*' in self.basis_functions[2]:
            return '*'
        elif op == 'Div' and '/' in self.basis_functions[2]:
            return '/'
        elif (op.lower() in self.basis_functions[1]) or (op.lower() in self.basis_functions[2]):
            return op.lower()
        raise Exception("Unknown operator type:" + op)
        
    def op2codeword(self, op):
        return self.code[self.op2str(op)]
