import ast
import itertools
import os
import re
import sys
import warnings

import numpy as np
from mpi4py import MPI

from katz.prior import KatzPrior

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_indices(n):
    """
    Find the indices of an array of length n which should be considered
    by this rank. For output data_start, data_end, the rank considers the
    entries array[data_start:data_end] where len(array) = n.

    Args:
        :n (int): The length of the array we wish to split among ranks.

    Returns:
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank.

    """

    nLs = int(np.ceil(n / float(size)))  # Number of lines per file for given thread

    while nLs * (size - 1) > n:
        if rank == 0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank == 0:
        print("Total number of functions: ", n, flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank * nLs
    data_end = (rank + 1) * nLs

    if rank == size - 1:
        data_end = n

    return data_start, data_end


def get_functions(comp, dirname):
    """
    Load all functions from a given directory at a certain complexity
    and divide these among the ranks.

    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the function lists

    Returns:
        :fcn_list (list[str]): The list of functions considered by this rank
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank

    """

    recursion_limits = {8: 2000, 9: 2500, 10: 3000}
    sys.setrecursionlimit(recursion_limits.get(comp, sys.getrecursionlimit()))

    fname = dirname + f"/compl_{comp}/all_equations_{comp}.txt"

    if rank == 0:
        print("Number of cores:", size, flush=True)

    with open(fname, "r") as f:
        fcn_list = f.readlines()

    data_start, data_end = get_indices(len(fcn_list))

    return fcn_list[data_start:data_end], data_start, data_end


def get_trees(comp, dirname):
    """
    Load the tree files for ESR for a given complexity and convert to
    lists of strings. These are split among ranks.

    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the ESR functions

    Returns:
        :tree_list (list): The list of lists giving trees to be considered by this rank

    """

    fname = dirname + f"/compl_{comp}/trees_{comp}.txt"
    with open(fname, "r") as f:
        tree_list = f.read().splitlines()
    data_start, data_end = get_indices(len(tree_list))
    tree_list = tree_list[data_start:data_end]
    for i in range(len(tree_list)):
        # Remove np.str_(...) wrappers
        tree = re.sub(r"np\.str_\((['\"])(.*?)\1\)", r"\1\2\1", tree_list[i])
        # Add commas between adjacent quoted strings
        tree = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", tree)
        tree_list[i] = ast.literal_eval(tree)

    return tree_list


def _sha256(path):
    """SHA-256 of a file's contents"""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _savetxt_atomic(path, values):
    """np.savetxt to a temporary file, then rename: an interrupted job cannot leave a
    truncated or mixed file behind"""
    tmp = f"{path}.tmp{os.getpid()}"
    np.savetxt(tmp, values)
    os.replace(tmp, path)


def _dump_json_atomic(path, obj):
    import json

    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _code_sha256():
    """SHA-256 of the source files that determine the scores: the katz modules and ESR's
    equation parser (esr.generation.generator), whose trees the prior scores"""
    import hashlib

    from esr.generation import generator as esr_generator

    here = os.path.dirname(os.path.abspath(__file__))
    names = (
        "prior.py",
        "back_off.py",
        "good_turing.py",
        "process_equations.py",
        "esr_prior.py",
    )
    paths = [os.path.join(here, name) for name in sorted(names)] + [
        esr_generator.__file__
    ]
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + b"\0" + f.read())
    return h.hexdigest()


def _n_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def _on_root(fn):
    """Run fn() on rank 0 only and broadcast its result. An exception on rank 0 is raised on
    every rank (on the others as a RuntimeError), so that no rank is left waiting in a later
    collective operation."""
    result, err = None, None
    if rank == 0:
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001
            err = e
    result, msg = comm.bcast(
        (result, None if err is None else f"{type(err).__name__}: {err}"), root=0
    )
    if err is not None:
        raise err
    if msg is not None:
        raise RuntimeError(f"rank 0 failed: {msg}")
    return result


def get_logconst(comp, dirname, overwrite=False):
    """
    Determine the sum(log(c_i)) for constants c_i appearning in all equations
    and save results to file.

    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the function lists
        :overwrite (bool): Whether to overwrite the constants file if it already exists

    Returns:
        :None

    """

    outname = dirname + f"/compl_{comp}/logconst_{comp}.txt"
    treename = dirname + f"/compl_{comp}/trees_{comp}.txt"
    sidecar = outname[:-4] + ".json"

    def check_reuse():
        # reuse only if computed from the current tree file (older files have no sidecar)
        import json

        if overwrite or not os.path.isfile(outname):
            return False
        try:
            with open(sidecar) as f:
                meta = json.load(f)
            reuse = meta.get("trees_sha256") == _sha256(treename) and meta.get(
                "logconst_sha256"
            ) == _sha256(outname)
        except (OSError, ValueError):
            reuse = False
        if not reuse:
            print(
                f"Recomputing {outname}: tree file changed or no record of it",
                flush=True,
            )
        return reuse

    if _on_root(check_reuse):
        return

    tree_list = get_trees(comp, dirname)
    logconst = [None] * len(tree_list)
    for i, tree in enumerate(tree_list):
        n = np.array([int(tt) for tt in tree if tt.lstrip("-").isdigit()])  # Integers
        n[n == 0] = 1  # So we have log(1) for 0 instead of log(0)
        logconst[i] = np.sum(np.log(np.abs(n)))

    logconst = comm.gather(logconst, root=0)

    def save():
        values = np.array(list(itertools.chain(*logconst)))
        _savetxt_atomic(outname, values)
        _dump_json_atomic(
            sidecar,
            {
                "trees_sha256": _sha256(treename),
                "logconst_sha256": _sha256(outname),
                "n": len(values),
            },
        )

    _on_root(save)

    return


def _prior_config(n, basis_functions, prior_kwargs):
    """Complete (JSON-serialisable) configuration of the KatzPrior built by compute_logprior"""
    import inspect

    defaults = {
        k: v.default
        for k, v in inspect.signature(KatzPrior.__init__).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    cfg = {
        k: bool(prior_kwargs.get(k, defaults[k])) for k in ("root_context", "normalise")
    }
    cfg["n"] = int(n)
    cfg["basis_functions"] = [list(b) for b in basis_functions]
    return cfg


def default_label(prior_kwargs):
    """Output-file prefix for a prior configuration: 'katz' plus the non-default options,
    e.g. 'katz', 'katz_noroot', 'katz_noroot_unnorm' (the model of version 0.2.1; the priors of
    Bartlett et al. 2023 need version 0.1.0)"""
    label = "katz"
    if not prior_kwargs.get("root_context", True):
        label += "_noroot"
    if not prior_kwargs.get("normalise", True):
        label += "_unnorm"
    return label


def compute_logprior(
    comp,
    n,
    basis_functions,
    dirname,
    in_eqfile,
    out_eqfile,
    overwrite=False,
    input_delimiter=",",
    use_tree=False,
    prior_kwargs=None,
    label=None,
):
    """
    Compute the log of the function prior for all functions at a give complexity
    given a corpus of equations. Saves results to two files: {label}_logprior_{n}_{comp}.txt
    contains just -log(prior) values, and {label}_codelen_{n}_{comp}.txt also contains
    the sum(log(c_i)) terms for constants c_i which appear in the equations. A manifest
    {label}_prior_{n}_{comp}.json records the prior configuration; hashes of the corpus, of
    the candidate functions (all_equations_{comp}.txt or trees_{comp}.txt), of the
    constant-penalty file logconst_{comp}.txt, of the scoring source code (the katz modules and
    ESR's equation parser) and of both outputs; the katz and sympy versions; and the number of functions that could not be
    scored (NaN in the output; a function with zero prior probability has -log(prior) = inf).
    Existing output is reused only if all of these match. Outputs are written atomically and
    the manifest last.

    Args:
        :comp (int): The complexity of function to consider
        :n (int): The length of the n-tuples to use for back-off model
        :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
        :dirname (str): The directory containing the function lists
        :in_eqfile (str): Name of file containing the equations to use as corpus
        :out_eqfile (str): Name of file to output standardised corpus equations to
        :overwrite (bool): Whether to overwrite files if they already exist
        :input_delimiter (str): The delimiter used in the input csv file
        :use_tree (bool, default=False): Whether to compute the prior using the tree representation from file (True) or using the string representation (False) and convert this to a tree automatically.
        :prior_kwargs (dict, default=None): Extra keyword arguments passed to KatzPrior (e.g. {'root_context': False, 'normalise': False} for the model of version 0.2.1; the priors of Bartlett et al. 2023 need version 0.1.0, see README)
        :label (str, default=None): Prefix of the output files, so that priors from different models are not overwritten. If None, default_label(prior_kwargs) is used: 'katz' followed by '_noroot' and '_unnorm' for non-default options.

    Returns:
        :None

    """
    import hashlib
    import json

    from katz.prior import __version__
    from katz.process_equations import standardise_file

    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    if label is None:
        label = default_label(prior_kwargs)
    suffix = f"_{n}_{comp}" + ("_tree" if use_tree else "")
    outname_code = dirname + f"/compl_{comp}/{label}_codelen{suffix}.txt"
    outname_prior = dirname + f"/compl_{comp}/{label}_logprior{suffix}.txt"
    outname_manifest = dirname + f"/compl_{comp}/{label}_prior{suffix}.json"

    # The corpus is read (and the standardised file written) by rank 0 only
    if "equations" in prior_kwargs:  # rank 0's list is the corpus on every rank
        equations = prior_kwargs.pop("equations")
        all_eq = _on_root(lambda: list(equations))
    else:
        all_eq = _on_root(
            lambda: standardise_file(in_eqfile, out_eqfile, input_delimiter)[0]
        )

    import sympy

    candname = dirname + (
        f"/compl_{comp}/trees_{comp}.txt"
        if use_tree
        else f"/compl_{comp}/all_equations_{comp}.txt"
    )
    logconst_name = dirname + f"/compl_{comp}/logconst_{comp}.txt"

    def manifest_and_reuse():
        manifest = {
            "config": _prior_config(n, basis_functions, prior_kwargs),
            "corpus_sha256": hashlib.sha256("\n".join(all_eq).encode()).hexdigest(),
            "corpus_size": len(all_eq),
            "comp": int(comp),
            "use_tree": bool(use_tree),
            "candidates_sha256": _sha256(candname),
            "n_candidates": _n_lines(candname),
            "logconst_sha256": _sha256(logconst_name),
            "code_sha256": _code_sha256(),
            "katz_version": __version__,
            "sympy_version": sympy.__version__,
        }
        reuse = False
        if (
            not overwrite
            and os.path.isfile(outname_code)
            and os.path.isfile(outname_prior)
        ):
            old = None
            if os.path.isfile(outname_manifest):
                with open(outname_manifest) as f:
                    old = json.load(f)
            keys = [k for k in manifest if k != "corpus_size"]
            why = None
            if old is None:
                why = "no manifest"
            elif any(old.get(k) != manifest[k] for k in keys):
                why = "different inputs: " + ", ".join(
                    k for k in keys if old.get(k) != manifest[k]
                )
            elif old.get("prior_sha256") != _sha256(outname_prior) or old.get(
                "codelen_sha256"
            ) != _sha256(outname_code):
                why = "output files changed since they were written"
            if why is None:
                reuse = True
            else:
                print(
                    f"Recomputing {outname_prior}: existing output has {why}",
                    flush=True,
                )
        return manifest, reuse

    manifest, reuse = _on_root(manifest_and_reuse)
    if reuse:
        return

    if use_tree:
        fcn_list_proc = get_trees(comp, dirname)
    else:
        fcn_list_proc, _, _ = get_functions(comp, dirname)

    kp = KatzPrior(n, basis_functions, equations=all_eq, **prior_kwargs)
    logprior_proc = np.empty(len(fcn_list_proc))
    nbad = np.zeros(2, dtype=int)  # zoo (complex infinity), failed to parse/score

    for i, eq in enumerate(fcn_list_proc):
        if "zoo" in eq:
            logprior_proc[i] = np.nan
            nbad[0] += 1
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value = kp.logprior(eq) if use_tree else kp.logprior(eq.strip())
                if np.isnan(value):
                    raise ValueError("the prior is NaN")
                logprior_proc[i] = value
            except Exception as e:  # noqa: BLE001
                if use_tree:
                    print("BAD EQ:", eq, e)
                else:
                    print("BAD EQ:", eq.strip(), e)
                logprior_proc[i] = np.nan
                nbad[1] += 1

    gathered = comm.gather(logprior_proc, root=0)
    nbad = comm.reduce(nbad, op=MPI.SUM, root=0)

    def write():
        logprior = np.array(list(itertools.chain(*gathered)))
        logconst = np.atleast_1d(np.loadtxt(logconst_name))
        if len(logconst) != len(logprior):
            raise ValueError(
                f"{logconst_name} has {len(logconst)} entries for {len(logprior)} "
                "functions: rerun get_logconst for the current library"
            )
        if os.path.isfile(outname_manifest):
            os.remove(outname_manifest)  # no manifest describes a partial update
        _savetxt_atomic(outname_prior, -logprior)
        # will use as replacement to aifeyn term, so need to sum these
        _savetxt_atomic(outname_code, -logprior + logconst)
        manifest.update(
            n_functions=len(logprior),
            n_zoo=int(nbad[0]),
            n_failed=int(nbad[1]),
            n_zero_prior=int(np.sum(np.isneginf(logprior))),
            prior_sha256=_sha256(outname_prior),
            codelen_sha256=_sha256(outname_code),
        )
        _dump_json_atomic(outname_manifest, manifest)
        if nbad[1] > 0:
            print(
                f"WARNING: {nbad[1]} of {len(logprior)} functions at complexity {comp} "
                "could not be scored (NaN in the output)",
                flush=True,
            )

    _on_root(write)

    return
