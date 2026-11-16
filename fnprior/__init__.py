"""fnprior: priors on functions for symbolic regression from n-gram language models of
their expression trees (called katz before version 0.5.0)"""

import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def data_path(name):
    """Path of a corpus file included in the package

    Args:
        :name (str): File name: 'PhysicsEquations.csv' (the physics corpus, delimiter ';')
            or 'FeynmanEquations.csv' (the AI Feynman equations, delimiter ',')

    Returns:
        :path (str): The path of the file
    """
    path = os.path.join(_DATA_DIR, name)
    if not os.path.isfile(path):
        available = ", ".join(sorted(os.listdir(_DATA_DIR)))
        raise FileNotFoundError(
            f"No corpus file {name!r} in fnprior; the files are: {available}"
        )
    return path
