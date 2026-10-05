"""Chain of Custody — forensics for tabular AI, built on TabPFN-3.5.

    from chainofcustody import Case

    case = Case(X, y, feature_names=names, device="cuda:0")
    finding = case.suspect(x_query, k=3)      # 1. find rows that could flip this decision
    proof   = case.prove(finding)             # 2. verify on the deployed model + controls
    suspects = case.catch(finding)            # 3. rank context rows by exact leave-one-out
    print(case.report(finding, proof, suspects))

Importing this package does not load Torch or TabPFN.
"""
from .case import Case, Finding, Proof, Suspects

__all__ = ["Case", "Finding", "Proof", "Suspects"]
__version__ = "0.2.0"
