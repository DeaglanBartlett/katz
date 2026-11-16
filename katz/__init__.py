"""Deprecated name of fnprior, kept for backwards compatibility.

The package was called katz before version 0.5.0. ``import katz`` and ``katz.<module>`` now
give the fnprior modules themselves (e.g. ``katz.prior`` is ``fnprior.prior``), with a
FutureWarning. This package will be removed in version 0.8.0; import fnprior instead.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

warnings.warn(
    "The katz package has been renamed fnprior; 'import katz' will stop working in "
    "version 0.8.0. Import fnprior instead (e.g. 'from fnprior.prior import FunctionPrior').",
    FutureWarning,
    stacklevel=2,
)


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import katz.<name> as the module fnprior.<name>, so that both names give the same
    module (and the same classes)"""

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("katz."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("fnprior" + spec.name[len("katz") :])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

from fnprior import data_path as data_path
