"""Conversion of equations (sympy strings or prefix label lists) into the binary/unary
expression trees used by the function prior, using sympy only.

The classes and functions below are copied from esr/generation/generator.py of the ESR
package (https://github.com/DeaglanBartlett/ESR, MIT licence; Bartlett, Desmond & Ferreira
2022), without the parts that need MPI or the rest of ESR, so that katz gives the same trees
as ESR without requiring it. The only change is that string_to_expr uses an empty namespace
when no `locs` are given. tests/test_katz.py checks that the n-tuples agree with those
obtained with ESR when ESR is installed.
"""

# The code is kept as in ESR, so that it can be compared line by line with its source.
# ruff: noqa: BLE001, C419, D419, SIM114

import numpy as np
import sympy
from sympy.core.sympify import kernS


def is_float(string):
    """Determine whether a string is a float or not

    Args:
        :string (str): The string to check

    Returns:
        bool: Whether the string is a float (True) or not (False).

    """
    try:
        float(eval(string))
        return True
    except Exception:
        return False


class Node:
    def __init__(self, t):
        self.type = t
        self.parent = None
        self.left = None
        self.right = None
        self.op = None
        self.val = None
        self.node_name = None
        self.tree = None

    def copy(self):
        new_node = Node(self.type)
        new_node.parent = self.parent
        new_node.left = self.left
        new_node.right = self.right
        return new_node

    def is_used(self):
        if (self.type == 0) and (self.parent is None):
            return False
        elif (self.type == 1) and (self.left is None):
            return False
        elif (self.type == 2) and (self.left is None) and (self.right is None):
            return False
        return True

    def assign_op(self, op):
        self.op = op
        v = op
        if v.lstrip("-").isdigit():
            self.val = int(v)
        elif v.lstrip("-").lstrip("/").isnumeric():
            self.val = float(v)
        else:
            self.val = v


class DecoratedNode:

    def __init__(self, fun, basis_functions, parent_op=None, parent=None):

        if fun is not None:

            self.expr = fun
            self.type = type(fun)
            self.constant = fun.is_number
            self.degree = len(fun.args)
            self.op = fun.__class__.__name__
            self.parent_op = parent_op
            self.parent = parent
            self.tree = None

            if self.constant:
                self.val = str(fun)
            elif fun.is_symbol:
                self.val = fun.name
            else:
                self.val = None

            if self.op == "Pow" and fun.args[1] == 2 and "square" in basis_functions[1]:
                self.op = "Square"
                self.children = [
                    DecoratedNode(
                        fun.args[0], basis_functions, parent_op=self.op, parent=self
                    )
                ]
            elif self.op == "Pow" and fun.args[1] == 3 and "cube" in basis_functions[1]:
                self.op = "Cube"
                self.children = [
                    DecoratedNode(
                        fun.args[0], basis_functions, parent_op=self.op, parent=self
                    )
                ]
            elif (
                self.op == "Pow"
                and fun.args[1] == 1 / 2
                and ("sqrt" in basis_functions[1] or "sqrt_abs" in basis_functions[1])
            ):
                self.op = "Sqrt"
                self.children = [
                    DecoratedNode(
                        fun.args[0], basis_functions, parent_op=self.op, parent=self
                    )
                ]
            elif (
                self.op == "Mul"
                and len(fun.args) == 2
                and fun.args[1].__class__.__name__ == "Pow"
                and fun.args[1].args[1] == -1
            ):
                self.op = "Div"
                self.children = [
                    DecoratedNode(
                        fun.args[0], basis_functions, parent_op=self.op, parent=self
                    ),
                    DecoratedNode(
                        fun.args[1].args[0],
                        basis_functions,
                        parent_op=self.op,
                        parent=self,
                    ),
                ]
            elif self.op == "Pow" and fun.args[1] == -1 and "inv" in basis_functions[1]:
                self.op = "Inv"
                self.children = [
                    DecoratedNode(
                        fun.args[0], basis_functions, parent_op=self.op, parent=self
                    )
                ]
            else:
                if len(fun.args) > 2:
                    f = fun.as_two_terms()
                    self.children = [
                        DecoratedNode(
                            f[0], basis_functions, parent_op=self.op, parent=self
                        ),
                        DecoratedNode(
                            f[1], basis_functions, parent_op=self.op, parent=self
                        ),
                    ]
                else:
                    self.children = [
                        DecoratedNode(
                            a, basis_functions, parent_op=self.op, parent=self
                        )
                        for a in fun.args
                    ]

    def from_node_list(self, idx, nodes, basis_functions, parent_op=None, parent=None):

        self.expr = nodes[idx].op
        self.type = nodes[idx].type
        self.constant = is_float(nodes[idx].val)
        self.degree = nodes[idx].type
        self.op = nodes[idx].op
        self.parent_op = parent_op
        self.parent = parent
        self.tree = None

        if self.constant:
            self.val = str(nodes[idx].val)
        else:
            self.val = nodes[idx].val

        if nodes[idx].right is not None:
            self.children = [
                DecoratedNode(None, basis_functions),
                DecoratedNode(None, basis_functions),
            ]
            self.children[0].from_node_list(
                nodes[idx].left,
                nodes,
                basis_functions,
                parent_op=nodes[idx].op,
                parent=self,
            )
            self.children[1].from_node_list(
                nodes[idx].right,
                nodes,
                basis_functions,
                parent_op=nodes[idx].op,
                parent=self,
            )
        elif nodes[idx].left is not None:
            self.children = [DecoratedNode(None, basis_functions)]
            self.children[0].from_node_list(
                nodes[idx].left,
                nodes,
                basis_functions,
                parent_op=nodes[idx].op,
                parent=self,
            )
        else:
            self.children = []

    def is_unity(self):
        try:
            f = float(self.val)
            return f == float(1)
        except Exception:
            return False

    def count_nodes(self, basis_functions):
        """
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        """
        return len(self.to_list(basis_functions))

    def to_list(self, basis_functions):
        """ """
        if self.degree == 0:
            return [str(self.val)]
        elif self.degree == 1:
            return [self.op] + self.children[0].to_list(basis_functions)
        #  Sqrt(x) instead of pow(x, 1/2)
        elif (
            self.op == "Pow"
            and (self.children[1].type == sympy.core.numbers.Half)
            and (("sqrt" in basis_functions[1]) or ("sqrt_abs" in basis_functions[1]))
        ):
            if "sqrt" in basis_functions[1]:
                return ["sqrt"] + self.children[0].to_list(basis_functions)
            else:
                return ["sqrt_abs"] + self.children[0].to_list(basis_functions)
        #  Square(x) instead of pow(x, 2) if possible
        elif (
            self.op == "Pow"
            and (self.children[1].val == str(2))
            and "square" in basis_functions[1]
        ):
            return ["square"] + self.children[0].to_list(basis_functions)
        # pow(x,2) instead of Square(x) if necessary
        elif self.op == "Square" and "square" not in basis_functions[1]:
            return ["pow"] + self.children[0].to_list(basis_functions) + ["2"]
        #  Cube(x) instead of pow(x, 3)
        elif (
            self.op == "Pow"
            and (self.children[1].val == str(3))
            and "cube" in basis_functions[1]
        ):
            return ["cube"] + self.children[0].to_list(basis_functions)
        # pow(x,2) instead of Square(x) if necessary
        elif self.op == "Cube" and "cube" not in basis_functions[1]:
            return ["pow"] + self.children[0].to_list(basis_functions) + ["3"]
        #  Inv(x) instead of pow(x, -1)
        elif (
            self.op == "Pow"
            and (self.children[1].type == sympy.core.numbers.NegativeOne)
            and ("inv" in basis_functions[1])
        ):
            return ["Inv"] + self.children[0].to_list(basis_functions)
        # Deal with * inv = /
        elif (
            self.op == "Mul"
            and self.children[0].op == "Pow"
            and (self.children[1].type == sympy.core.numbers.NegativeOne)
            and ("/" in basis_functions[2])
        ):
            return ["Mul"] + self.children[1].to_list(basis_functions)
        # Deal with / inv = *
        elif (
            self.op == "Div"
            and self.children[0].op == "Pow"
            and (self.children[1].type == sympy.core.numbers.NegativeOne)
            and ("*" in basis_functions[2])
        ):
            return ["Mul"] + self.children[1].to_list(basis_functions)
        #  Multiply or divide by one doesn't do anything
        elif self.op == "Mul" and (
            self.children[0].is_unity() or self.children[1].is_unity()
        ):
            if self.children[0].is_unity():
                return self.children[1].to_list(basis_functions)
            else:
                return self.children[0].to_list(basis_functions)
        # Don't keep abs after pow or sqrt
        elif self.op == "Abs" and self.parent.op in ["Sqrt", "Pow"]:
            return self.children[0].to_list(basis_functions)
        elif self.op == "Div" and (self.children[0] == 1 or self.children[1] == 1):
            pass
        elif (
            self.op == "Add"
            and self.children[1].op == "Mul"
            and (
                self.children[1].children[0].op == "NegativeOne"
                or self.children[1].children[1].op == "NegativeOne"
            )
        ):
            if self.children[1].children[0].op == "NegativeOne":
                return (
                    ["Sub"]
                    + self.children[0].to_list(basis_functions)
                    + self.children[1].children[1].to_list(basis_functions)
                )
            else:
                return (
                    ["Sub"]
                    + self.children[0].to_list(basis_functions)
                    + self.children[1].children[0].to_list(basis_functions)
                )
        else:
            r = [self.op]
            for c in self.children:
                r = r + c.to_list(basis_functions)
            return r

    def get_lineage(self):
        p = [self.op, self.parent_op]
        q = self.parent
        while q is not None:
            p += [q.parent_op]
            q = q.parent
        p.reverse()
        p = [tuple(p)]
        for c in self.children:
            p += c.get_lineage()
        return p

    def get_sibling_lineage(self):
        #  First get direct lineage of nodes
        p = [self.op, self.parent_op]
        v = [self.val]
        if self.parent_op is None:
            v += [None]
        else:
            v += [self.parent.val]
        q = self.parent
        while q is not None:
            p += [q.parent_op]
            if q.parent_op is None:
                v += [None]
            else:
                v += [q.parent.val]
            q = q.parent
        p.reverse()
        v.reverse()
        # Now add the siblings
        if len(self.children) == 1:
            p += [(self.children[0].op, None)]
            v += [(self.children[0].val, None)]
        elif len(self.children) > 1:
            p += [tuple([c.op for c in self.children])]
            v += [tuple([c.val for c in self.children])]

        p = [tuple(p)]
        v = [tuple(v)]
        for c in self.children:
            if len(c.children) > 0:
                pp, vv = c.get_sibling_lineage()
                p += pp
                v += vv
        return p, v

    def get_siblings(self):
        if self.parent is not None and len(self.parent.children) > 1:
            p = [tuple([c.op for c in self.parent.children])]
        else:
            p = [(self.op, "None")]
        for c in self.children:
            p += c.get_siblings()
        return p


def check_tree(s):
    """Given a candidate string of 0, 1 and 2s, see whether one can make a function out of this

    Args:
        :s (str): string comprised of 0, 1 and 2 representing tree of nullary, unary and binary nodes

    Returns:
        :success (bool): whether candidate string can form a valid tree (True) or not (False)
        :part_considered (str): string of length <= s, where s[:len(part_considered)] = part_considered
        :tree (list): list of Node objects corresponding to string s
    """

    tree = [Node(t) for t in s]

    for i in range(len(s) - 1):

        success = False

        if (tree[i].type == 2) or (tree[i].type == 1):
            # Add to the left if possible
            tree[i].left = i + 1
            tree[i + 1].parent = i
            success = True
        else:
            #  try to go up the tree
            j = tree[i].parent

            while not success:

                if (tree[j].type == 2) and (tree[j].right is None):
                    # Add to right of node if possible
                    tree[j].right = i + 1
                    tree[i + 1].parent = j
                    success = True
                elif tree[j].parent is None:
                    #  Check if can't move up the tree any higher
                    break

                # Go up the tree to this node's parent
                j = tree[j].parent

        if not success:
            break

    if len(s) > 1:

        #  Need to check for parents without left nodes which should have them
        if success:
            lefts = [t.left for t in tree if t.type == 1 or t.type == 2]
            if None in lefts:
                success = False

        #  Need to check for parents without right nodes which should have them
        if success:
            rights = [t.right for t in tree if t.type == 2]
            if None in rights:
                success = False

        # This will allow us to delete any trees which start with this
        part_considered = s[: i + 2]

    else:
        success = True
        part_considered = None

    return success, part_considered, tree


def string_to_expr(s, kern=False, evaluate=False, locs=None):
    """Convert a string giving function into a sympy object

    Args:
        :s (str): string representation of the function considered
        :kern (bool): whether to use sympy's kernS function or sympify
        :evaluate (bool): whether to use powsimp, factor and subs
        :locs (dict): dictionary of string:sympy objects. If None, will create here

    Returns:
        :expr (sympy object): expression corresponding to s

    """

    s = s.replace("[", "(")
    s = s.replace("]", ")")
    s = s.replace("Sqrt", "sqrt")
    s = s.replace("*^", "*10^")

    if locs is None:
        locs = {}

    if kern:
        expr = kernS(s)
    else:
        expr = sympy.sympify(s, evaluate=False, locals=locs)

        if evaluate:
            expr = expr.powsimp(expr)
            expr = expr.factor()
            expr = expr.subs(1.0, 1)

    return expr


def check_operators(nodes, basis_functions):
    """Check whether all operators in the tree are in the basis

    Args:
        :nodes (DecoratedNode): Node representation of the function tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators

    Returns:
        :all_in_basis (bool): Whether all functions in tree are in basis
    """

    sympy_numerics = [
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
    sympy_numerics = [s.lower() for s in sympy_numerics]

    labels = nodes.to_list(basis_functions)

    for i in range(len(labels)):
        if labels[i] == "Add" and "+" in basis_functions[2]:
            labels[i] = "+"
        elif labels[i] == "Sub" and "-" in basis_functions[2]:
            labels[i] = "-"
        elif labels[i] == "Mul" and "*" in basis_functions[2]:
            labels[i] = "*"
        elif labels[i] == "Div" and "/" in basis_functions[2]:
            labels[i] = "/"
        elif labels[i].lower() in sympy_numerics or is_float(labels[i]):
            labels[i] = "a"
        elif labels[i].startswith("a") and labels[i][1:].isdigit():
            labels[i] = "a"
        elif labels[i].startswith("x") and labels[i][1:].isdigit():
            labels[i] = "x"
        else:
            labels[i] = labels[i].lower()

    flat_basis = [item for sublist in basis_functions for item in sublist]
    all_in_basis = all([ll in flat_basis for ll in labels])

    return all_in_basis


def string_to_node(
    s, basis_functions, locs=None, evalf=False, allow_eval=True, check_ops=False
):
    """Convert a string giving function into a tree with labels

    Args:
        :s (str): string representation of the function considered
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        :locs (dict): dictionary of string:sympy objects. If None, will create here
        :evalf (bool): whether to run evalf() on function (default=False)
        :allow_eval (bool, default=True): whether to run the (kernS=False and evaluate=True) option
        :check_ops (bool, default=False): whether to check all operators appear in basis functions

    Returns:
        :tree (list): list of Node objects corresponding to the tree
        :labels (list): list of strings giving node labels of tree

    """

    expr = [None] * 4
    nodes = [None] * 4
    if check_ops:
        all_in_basis = [False] * 4
    c = np.full(4, np.nan)

    if allow_eval:
        i = 0
        try:
            expr[i] = string_to_expr(s, kern=False, evaluate=True, locs=locs)
            if evalf:
                expr[i] = expr[i].evalf()
            nodes[i] = DecoratedNode(expr[i], basis_functions)
            c[i] = nodes[i].count_nodes(basis_functions)
            if check_ops:
                all_in_basis[i] = check_operators(nodes[i], basis_functions)
        except Exception:
            c[i] = np.nan

    i = 1
    try:
        expr[i] = string_to_expr(s, kern=False, evaluate=False, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
        if check_ops:
            all_in_basis[i] = check_operators(nodes[i], basis_functions)
    except Exception:
        c[i] = np.nan

    i = 2
    try:
        expr[i] = string_to_expr(s, kern=True, evaluate=True, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
        if check_ops:
            all_in_basis[i] = check_operators(nodes[i], basis_functions)
    except Exception:
        c[i] = np.nan

    i = 3
    try:
        expr[i] = string_to_expr(s, kern=True, evaluate=False, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
        if check_ops:
            all_in_basis[i] = check_operators(nodes[i], basis_functions)
    except Exception:
        c[i] = np.nan

    if check_ops and any(all_in_basis):
        for i in range(len(all_in_basis)):
            if not all_in_basis[i]:
                c[i] = np.nan
    i = np.nanargmin(c)

    return expr[i], nodes[i], int(c[i])


def labels_to_shape(labels, basis_functions):
    """Find the representation of the shape of a tree given its labels

    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators

    Returns:
        :s (str): string comprised of 0, 1 and 2 representing tree of nullary, unary and binary nodes

    """

    basis_dict = {}
    for i in range(len(basis_functions)):
        for f in basis_functions[i]:
            basis_dict[f] = i

    s = [None] * len(labels)
    for i, t in enumerate(labels):
        try:
            s[i] = basis_dict[t]
        except Exception:
            if (t.startswith("a") and t[1:].isdigit()) or (is_float(t)):
                s[i] = 0
            else:
                raise ValueError
    return s
