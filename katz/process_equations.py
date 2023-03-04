import pandas as pd
import numpy as np
import string
import csv
import sympy
import itertools
from sympy.core.sympify import kernS

from esr.generation.generator import DecoratedNode

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
    
    
def process_all_equations(n, all_eq, maxvar, coder):

    x = sympy.symbols([f'x{i}' for i in range(maxvar)], real=True)
    locs = {f'x{i}':x[i] for i in range(len(x))}
    
    ntuples = []
    for eq in all_eq:
        ntuples += equation2ntuples(n, eq, locs, coder)
    print(f'NUMBER OF {n}-TUPLES:', len(ntuples))

    return ntuples
    
    
def string_to_expr(s, locs, kern=False, evaluate=False):
    """Convert a string giving function into a sympy object
    
    Args:
        :s (str): string representation of the function considered
        :locs (dict): dictionary of string:sympy objects
        :kern (bool): whether to use sympy's kernS function or sympify
        :evaluate (bool): whether to use powsimp, factor and subs
        
    Returns:
        :expr (sympy object): expression corresponding to s
    
    """
    
    s = s.replace('[', '(')
    s = s.replace(']', ')')
    s = s.replace('Sqrt', 'sqrt')
    s = s.replace('*^', '*10^')
    
    if kern:
        expr = kernS(s)
    else:
        expr = sympy.sympify(s, locals=locs)
        if evaluate:
            expr = expr.powsimp(expr)
            expr = expr.factor()
            expr = expr.subs(1.0, 1)
    
    return expr
    
    
def string_to_node(s, locs, basis_functions):
    """Convert a string giving function into a tree with labels
    
    Args:
        :s (str): string representation of the function considered
        :locs (dict): dictionary of string:sympy objects
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        
    Returns:
        :tree (list): list of Node objects corresponding to the tree
        :labels (list): list of strings giving node labels of tree
    
    """
    
    expr = [None] * 3
    nodes = [None] * 3
    c = np.ones(3, dtype=int)

    i = 0
    expr[i] = string_to_expr(s, locs, kern=False, evaluate=True)
    nodes[i] = DecoratedNode(expr[i], basis_functions)
    c[i] = nodes[i].count_nodes(basis_functions)
    
    i = 1
    expr[i] = string_to_expr(s, locs, kern=False, evaluate=False)
    nodes[i] = DecoratedNode(expr[i], basis_functions)
    c[i] = nodes[i].count_nodes(basis_functions)

    i = 2
    expr[i] = string_to_expr(s, locs, kern=True)
    nodes[i] = DecoratedNode(expr[i], basis_functions)
    c[i] = nodes[i].count_nodes(basis_functions)

    i = c.argmin()

    return expr[i], nodes[i], c[i]
    
    
def equation2ntuples(n, eq, locs, coder):

    expr, nodes, c = string_to_node(eq, locs, coder.basis_functions)
    
    lin = nodes.get_lineage()
    
    # USE WHEN WE DO SIBLINGS
#    lin = nodes.get_sibling_lineage()
#    sib = nodes.get_siblings()

    ntuples = [None] * len(lin)
    for i, t in enumerate(lin):
        if len(t) >= n:
            x = t[-n:]
        else:
            x = tuple([None]*(n-len(t)) + list(t))

        ntuples[i] = tuple([coder.op2codeword(tt) for tt in x])
#        for i in range(n-1):
#            x[i] = (x[i],)
#        x[-1] = (x[-1],)

    return ntuples
    
    
class SymbolCoder:

    def __init__(self, basis_functions):
        self.basis_functions = basis_functions
        
        self.sympy_numerics = ['Number', 'Float', 'Rational', 'Integer', 'AlgebraicNumber',
                    'NumberSymbol', 'RealNumber', 'igcd', 'ilcm', 'seterr', 'Zero',
                    'One', 'NegativeOne', 'Half', 'NaN', 'Infinity', 'NegativeInfinity',
                    'ComplexInfinity', 'Exp1', 'ImaginaryUnit', 'Pi', 'EulerGamma',
                    'Catalan', 'GoldenRatio', 'TribonacciConstant', 'mod_inverse']
        self.ops = [str(None), 'a', 'x'] + basis_functions[1] + basis_functions[2]
#        self.code = [(t,) for t in self.ops] + list(itertools.product(self.ops, repeat=2)) # USE WHEN WE DO SIBLINGS
        self.code = [t for t in self.ops]
        self.code = dict(zip(self.code, np.arange(len(self.code)).astype(str)))
    
        
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
        

"""
TO DO
- Change Feynman equations to identify variables and constants as different?
- Allow more than 10 input variables
"""
