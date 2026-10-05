"""Proofread — spellcheck for tables, powered by TabPFN-3.5.

    from proofread import Proofreader
    report = Proofreader(device="cuda:0").check(df, label="target")
    report.issues            # one row per suspicious cell or label, most suspicious first
    report.highlight()       # pandas Styler with flagged cells marked

Every cell is predicted out-of-fold from the rest of its row by TabPFN-3.5; a value is suspicious when it
falls far in the tail of TabPFN's full predictive distribution. Importing this package does not load Torch.
"""
from .core import Proofreader, Report
from . import notebook as _notebook          # registers the pandas accessor df.proofread (no Torch import)
from .notebook import load_ipython_extension  # `%load_ext proofread`

__all__ = ["Proofreader", "Report", "load_ipython_extension"]
__version__ = "0.1.0"
