import csv
import re
import string

import numpy as np
import pandas as pd
import sympy
from esr.generation import generator


def split_by_punctuation(s):
    """
    Split a string s into a list, where each instance of punctuation or a space causes a split.
    E.g. the string s = 'Hello, how are you?' becomes ['Hello', ',', ' ', 'how', ' ', 'are', ' ', 'you', '?']

    Args:
        :s (str): String we wish to split

    Returns:
        :split_str (list): List of strings split by punctuation

    """
    pun = string.punctuation.replace("_", "")  # allow underscores in variable names
    pun = pun + " "
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    if len(where_pun) == 0:  # e.g. a bare variable 'x0'
        return [s]
    split_str = [s[: where_pun[0]]]
    for i in range(len(where_pun) - 1):
        split_str += [s[where_pun[i]]]
        split_str += [s[where_pun[i] + 1 : where_pun[i + 1]]]
    split_str += [s[where_pun[-1]]]
    if where_pun[-1] != len(s) - 1:
        split_str += [s[where_pun[-1] + 1 :]]
    return split_str


def count_variables(all_eq):
    """
    Number of variables x0, x1, ... needed to parse a list of equation strings, i.e. one
    more than the largest variable index (at least 1)

    Args:
        :all_eq (list of str): Equations with variables named x0, x1, ...

    Returns:
        :maxvar (int): Largest variable index + 1
    """
    idx = [int(m) for eq in all_eq for m in re.findall(r"(?<![A-Za-z0-9_])x(\d+)", eq)]
    return max(idx) + 1 if len(idx) > 0 else 1


def standardise_file(in_name, out_name, input_delimiter):
    """
    Standardise the input equations used so that variables are named x0, x1, ..., x9

    Args:
        :in_name (str): Name of file containing the equations to study. If None, then equations read from out_eqfile
        :out_name (str): Name of file to output the new equations to
        :input_delimiter (str): The delimiter used in the input csv file

    Returns:
        :all_eq (list): List of equations as strings with the standardised variable names
        :max_var (int): The maximum number of variables appearing in any of the equations

    """

    if in_name is None:

        df = pd.read_csv(out_name, delimiter=input_delimiter)
        all_eq = df["New Formula"].tolist()
        maxvar = count_variables(all_eq)
    else:

        df = pd.read_csv(in_name, delimiter=input_delimiter)
        maxvar = int(df["# variables"].max())  # + 1

        all_eq = []

        with open(out_name, "w") as f:
            csvwriter = csv.writer(f)

            csvwriter.writerow(["Filename", "Number", "Old Formula", "New Formula"])

            for index, row in df.iterrows():

                if not np.isfinite(row["# variables"]):
                    continue
                eq = row["Formula"].replace(" ", "")

                # If equation already has variable 'x{i}" then we don't need to replace it
                # Note: At least Eqs I.18.12, I.18.14 and II.37.1 in AIFeynman has wrong number of vars
                # The next two lines fix this
                vars = [row[f"v{i+1}_name"] for i in range(maxvar)]
                vars = [v.replace(" ", "") for v in vars if isinstance(v, str)]

                names = [f"x{i}" for i in range(len(vars))]
                to_change = sorted(set(vars) - set(names), key=vars.index)
                to_sub = sorted(
                    set(names) - (set(vars) - set(to_change)), key=names.index
                )

                # Must split by punctuation to avoid replacing e.g. "t" in "sqrt" if we have a variable "t"
                sub_dict = dict(zip(to_change, to_sub))
                split_eq = split_by_punctuation(eq)
                for i, s in enumerate(split_eq):
                    if s in to_change:
                        split_eq[i] = sub_dict[s]
                eq = "".join(split_eq)

                all_eq.append(eq)

                csvwriter.writerow([row["Filename"], row["Number"], row["Formula"], eq])

    return all_eq, maxvar


class SymbolCoder:

    def __init__(self, basis_functions, root_context=False):
        """Class to encode equations as tuples of strings to be used by a back-off model

        Args:
            :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
            :root_context (bool, default=False): If True, lineages which reach the root are left-padded with the "None" codeword, so that the root and nodes near it are conditioned on an explicit start-of-tree context. If False (original behaviour), such tuples are simply shorter.

        Returns:
            SymbolCoder: A coder to encode equations
        """
        self.basis_functions = basis_functions
        self.root_context = root_context

        self.sympy_numerics = [
            "Number",
            "Float",
            "Rational",
            "Integer",
            "AlgebraicNumber",
            "NumberSymbol",
            "RealNumber",
            "igcd",
            "ilcm",
            "seterr",
            "Zero",
            "One",
            "NegativeOne",
            "Half",
            "NaN",
            "Infinity",
            "NegativeInfinity",
            "ComplexInfinity",
            "Exp1",
            "ImaginaryUnit",
            "Pi",
            "EulerGamma",
            "Catalan",
            "GoldenRatio",
            "TribonacciConstant",
            "mod_inverse",
        ]
        self.sympy_numerics = [s.lower() for s in self.sympy_numerics]
        self.ops = [str(None), "a", "x", "y"] + basis_functions[1] + basis_functions[2]
        self.code = self.ops
        self.code = dict(zip(self.code, np.arange(len(self.code)).astype(str)))
        self.ignore_ops = [
            "Abs",
            "re",
            "im",
        ]  # do not attempt to find probability of these operators

    def nodes2ntuples(self, n, nodes):
        """
        Convert a node object giving the tree representation of an equation into
        n-tuples describing the tree structure of the function

        Args:
            :n (int): The length of the n-tuples to produce
            :nodes (esr.generation.generator.DecoratedNode): Node object corresponding to the tree

        Returns:
            :ntuples (list); List of n-tuples which describe tree structure of function

        """

        lin, val = nodes.get_sibling_lineage()

        ntuples = []
        for t, v in zip(lin, val):

            # Check for ignored operators
            idx = [i for i, tt in enumerate(t) if tt not in self.ignore_ops]
            tnew = [t[i] for i in idx]
            vnew = [v[i] for i in idx]
            if tnew[-1][0] in self.ignore_ops or (
                len(tnew[-1]) > 1 and tnew[-1][1] in self.ignore_ops
            ):
                continue

            # Get codeword of ancestors
            if len(tnew) >= n:
                x = tnew[-n:-1]
            else:
                x = tuple([None] * (n - len(tnew)) + list(tnew[:-1]))
            nt = [self.op2codeword(tt) for tt in x]
            # Deal with sibling at end of tree
            if isinstance(t[-1], tuple):
                sib = [self.op2str(tt) for tt in t[-1]]
            else:
                sib = [self.op2str(tt) for tt in (t[-1], None)]
            if sib[0] == "x" and sib[1] == "x" and (vnew[-1][0] != vnew[-1][1]):
                sib[1] = "y"
            tup = list(nt + [self.code[s] for s in sib])

            # Remove "None" at start of lineage
            idx = [i for i in range(len(tup)) if tup[i] != self.code["None"]]
            tup = tuple(tup[idx[0] :])
            if self.root_context:
                # Explicit start-of-tree context: pad back to the full length
                tup = (self.code["None"],) * (n + 1 - len(tup)) + tup
                tup = tup[-(n + 1) :]
            ntuples.append(tup)

        # The root event, taken from the tree itself (the first lineage tuple is skipped
        # when the root's children include an ignored operator); an ignored operator at the
        # root is skipped like ignored ancestors in the lineages
        root_node = nodes
        while root_node.op in self.ignore_ops and len(root_node.children) > 0:
            root_node = root_node.children[0]
        root = self.op2codeword(root_node.op)
        if self.root_context:
            ntuples = [(self.code["None"],) * (n - 1) + (root,)] + ntuples
        else:
            ntuples = [(root,)] + ntuples

        # A single-leaf tree has only the root event: the loop above also produced
        # a (leaf, None) sibling tuple for it, which would count the root twice
        if len(nodes.children) == 0:
            ntuples = ntuples[:1]

        return ntuples

    def labels2ntuples(self, n, labels):
        """
        Convert a list of labels giving the tree representation of an equation into
        n-tuples describing the tree structure of the function

        Args:
            :n (int): The length of the n-tuples to produce
            :labels (list): The list giving the equation to convert to an n-tuple

        Returns:
            :ntuples (list); List of n-tuples which describe tree structure of function

        """

        labels_changed = labels.copy()
        for i, lab in enumerate(labels):
            if lab.lower() in self.sympy_numerics or generator.is_float(lab):
                labels_changed[i] = "a"

        # Get parent operators. Every x_i/a_i is a leaf, including variables beyond the
        # corpus's range (the nullary basis lists only x0..x{maxvar-1}).
        leaves = [lab for lab in labels_changed if re.fullmatch(r"[xa]\d+", lab)]
        shape_basis = [list(self.basis_functions[0]) + leaves] + list(
            self.basis_functions[1:]
        )
        s = generator.labels_to_shape(labels_changed, shape_basis)
        success, _, tree = generator.check_tree(s)
        assert success

        for i, lab in enumerate(labels_changed):
            tree[i].assign_op(lab)

        nodes = generator.DecoratedNode(None, self.basis_functions)
        nodes.from_node_list(0, tree, self.basis_functions)
        ntuples = self.nodes2ntuples(n, nodes)

        return ntuples

    def equation2ntuples(self, n, eq, locs):
        """
        Convert an equation into n-tuples describing the tree structure of the function

        Args:
            :n (int): The length of the n-tuples to produce
            :eq (str): The equation to convert to an n-tuple
            :locs (dict): dictionary of string:sympy objects describing variables

        Returns:
            :ntuples (list); List of n-tuples which describe tree structure of function

        """
        return self.labels2ntuples(n, self.equation2labels(eq, locs))

    def equation2labels(self, eq, locs):
        """
        Convert an equation into the list of labels (prefix notation) of its tree

        Args:
            :eq (str): The equation to convert
            :locs (dict): dictionary of string:sympy objects describing variables

        Returns:
            :labels (list): The labels of the tree nodes in prefix order, with numerical constants replaced by "a"

        """

        expr, nodes, _ = generator.string_to_node(
            eq,
            self.basis_functions,
            locs=locs,
            evalf=True,
            allow_eval=False,
            check_ops=True,
        )

        # Remove any operators we want to ignore ('Abs'). Look for them in the parsed
        # expression as well as in the input, which may spell them differently ('abs')
        redo = any(op in eq for op in self.ignore_ops) or any(
            type(e).__name__ in self.ignore_ops for e in sympy.preorder_traversal(expr)
        )
        if redo:
            s = str(expr)
            s = split_by_punctuation(s)
            s = [ss for ss in s if ss not in self.ignore_ops]
            s = "".join(s)
            expr, nodes, _ = generator.string_to_node(
                s,
                self.basis_functions,
                locs=locs,
                evalf=True,
                allow_eval=False,
                check_ops=True,
            )

        labels = nodes.to_list(self.basis_functions)
        for i in range(len(labels)):
            if labels[i] == "Add" and "+" in self.basis_functions[2]:
                labels[i] = "+"
            elif labels[i] == "Sub" and "-" in self.basis_functions[2]:
                labels[i] = "-"
            elif labels[i] == "Mul" and "*" in self.basis_functions[2]:
                labels[i] = "*"
            elif labels[i] == "Div" and "/" in self.basis_functions[2]:
                labels[i] = "/"
            elif labels[i].lower() in self.sympy_numerics or generator.is_float(
                labels[i]
            ):
                labels[i] = "a"
            else:
                labels[i] = labels[i].lower()

        return labels

    def process_all_equations(self, n, all_eq, maxvar):
        """
        Turn all equations into n-tuples describing the tree structures of their functions

        Args:
            :n (int): The length of the n-tuples to produce
            :all_eq (list): List of equations as strings to convert to n-tuples
            :maxvar (int): The maximum number of variables appearing in any of the equations

        Returns:
            :ntuples (list): List of n-tuples which describe tree structures of the functions
        """

        # Real symbols for every x_i and a_i used, not just the corpus's variable range: an
        # undeclared symbol would be parsed without assumptions (e.g. sqrt(a5**2) does not
        # simplify to |a5|) and can fail to convert to a tree.
        nsym = max(
            [int(maxvar)]
            + [
                int(k) + 1
                for eq in all_eq
                if isinstance(eq, str)
                for k in re.findall(r"(?<![A-Za-z_0-9])[xa](\d+)", eq)
            ]
        )
        x = sympy.symbols([f"x{i}" for i in range(nsym)], real=True)
        a = sympy.symbols([f"a{i}" for i in range(nsym)], real=True)
        locs = {
            **{f"x{i}": x[i] for i in range(nsym)},
            **{f"a{i}": a[i] for i in range(nsym)},
        }

        ntuples = []
        for eq in all_eq:
            if isinstance(eq, str):
                ntuples += self.equation2ntuples(n, eq, locs)
            else:
                ntuples += self.labels2ntuples(n, eq)

        return ntuples

    def op2str(self, op):
        """
        Convert operator names defined by sympy into symbols used here

        Args:
            :op (str): Operator name of sympy class

        Returns:
            str: The equivalent symbol used here
        """
        if op is None:
            return str(None)
        elif op.lower() in self.sympy_numerics or generator.is_float(op):
            return "a"
        elif (
            op == "Symbol"
            or op.startswith(("x", "a"))
            and (op[1:].isdigit() or len(op) == 1)
        ):
            return "x"
        elif op == "Add" and "+" in self.basis_functions[2]:
            return "+"
        elif op == "Sub" and "-" in self.basis_functions[2]:
            return "-"
        elif op == "Mul" and "*" in self.basis_functions[2]:
            return "*"
        elif op == "Div" and "/" in self.basis_functions[2]:
            return "/"
        elif (op.lower() in self.basis_functions[1]) or (
            op.lower() in self.basis_functions[2]
        ):
            return op.lower()
        raise ValueError("Unknown operator type:" + op)

    def op2codeword(self, op):
        """
        Convert an operator name as defined by sympy into the codeword assigned to it

        Args:
            :op (str): Operator name of sympy class

        Returns:
            str: The codeword used to represent this symbol
        """
        return self.code[self.op2str(op)]
