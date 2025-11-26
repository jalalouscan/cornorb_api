def munnerlyn_ablation(diopters, oz_mm=6.0):
    """Ablation depth in microns."""
    return (diopters * (oz_mm**2)) / 3.0

def compute_rsb(cct, ablation, flap=110):
    return cct - (flap + ablation)

def evaluate_safety(cct, kmax, kc_prob, rsb):
    safe = {
        "RSB": rsb,
        "LASIK_allowed": True,
        "PRK_allowed": True,
        "IntraLase_allowed": True,
        "warnings": []
    }

    if kc_prob > 0.5:
        safe["LASIK_allowed"] = False
        safe["IntraLase_allowed"] = False
        safe["warnings"].append("Keratoconus detected")

    if cct < 480:
        safe["LASIK_allowed"] = False
        safe["PRK_allowed"] = False
        safe["IntraLase_allowed"] = False
        safe["warnings"].append("Cornea too thin (<480µm)")

    if rsb < 300:
        safe["LASIK_allowed"] = False
        safe["IntraLase_allowed"] = False
        safe["warnings"].append("RSB < 300µm")

    if 300 <= rsb < 330:
        safe["LASIK_allowed"] = False
        safe["warnings"].append("Borderline RSB (300–330µm)")

    return safe
