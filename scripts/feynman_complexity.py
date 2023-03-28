import pandas as pd
import sympy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc

rc('font', size=12)

rcParams['text.usetex'] = True

import esr.generation.generator as generator

basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                ["+", "-", "*", "/", "pow"]]

in_name = '../data/NewFeynman.csv'
out_name = '../data/NewFeynman_Complexity.csv'
maxvar = 10

df = pd.read_csv(in_name)
all_eq = list(df['New Formula'])
number = np.array(df['Number'], dtype=int)

x = sympy.symbols([f'x{i}' for i in range(maxvar)], real=True)
a = sympy.symbols([f'a{i}' for i in range(maxvar)], real=True)
locs = {f'x{i}':x[i] for i in range(len(x))} | {f'a{i}':a[i] for i in range(len(x))}

all_c = np.empty(len(all_eq))
all_nodes = [None] * len(all_eq)

for i, eq in enumerate(all_eq):

    opts = [
            {'kern':False, 'evaluate':True},
            {'kern':False, 'evaluate':False},
            {'kern':True, 'evaluate':True},
            {'kern':True, 'evaluate':False},
           ]
    all_c[i] = 1e10
    
    for j in range(len(opts)):
            
        expr = generator.string_to_expr(eq, locs=locs, **opts[j]) #kern=True, evaluate=True)
        expr = expr.evalf()
        nodes = generator.DecoratedNode(expr, basis_functions)
        c = nodes.count_nodes(basis_functions)
        if c < all_c[i]:
            all_nodes[i] = nodes
            all_c[i] = c
    
        if number[i] == 5:#90:
            print(c, all_c[i], '\t', expr)
        
print('Top 5 hardest:')
m = np.argsort(-all_c)
all_c = all_c[m]
all_eq = [all_eq[i] for i in m]
all_nodes = [all_nodes[i] for i in m]
number = number[m]
for i in range(5):
    print(all_c[i], '\t', number[i], '\t', all_eq[i])
#    print('\t', all_nodes[i].to_list(basis_functions))

    
plt.hist(all_c, bins=20, histtype='step')
plt.xlabel('Complexity')
plt.ylabel('Number of AI Feynman Equations')
plt.title(f'Number up complexity 10: {np.sum(all_c<=10)} of {len(all_c)}')
plt.savefig('feynman_complexity.png')
plt.clf()
