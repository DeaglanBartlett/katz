import pandas as pd
import numpy as np
import string
import csv

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

    return

"""
TO DO
- Change Feynman equations to identify variables and constants as different?
- Allow more than 10 input variables
"""
