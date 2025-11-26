import numpy as np

def final_decision(safety, ml_pred, ml_probs):
    """Override ML predictions if rules forbid the surgery."""
    if not safety["LASIK_allowed"] and ml_pred == "LASIK":
        return "PRK", "LASIK forbidden by safety rules"

    if not safety["IntraLase_allowed"] and ml_pred == "IntraLase":
        return "PRK", "IntraLase forbidden by safety rules"

    if not safety["PRK_allowed"] and ml_pred == "PRK":
        return "NO_SURGERY", "PRK forbidden by safety rules"

    return ml_pred, None
