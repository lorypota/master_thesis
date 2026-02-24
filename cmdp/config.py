R_MAX_VALUES = [0.05, 0.0625, 0.075, 0.0875, 0.10, 0.125, 0.15, 0.20, 1.0]


def fmt_token(value):
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def compute_failure_thresholds(
    r_max, demand_params, active_cats, constrained_cats=None
):
    """Compute per-category failure thresholds for CMDP constraints.

    Args:
        r_max: Maximum allowed failure rate.
        demand_params: List of (lambda_a, lambda_d) tuples per time slot, per category.
        active_cats: List of active category indices.
        constrained_cats: Set of category indices to constrain (default: all active).

    Returns:
        Dict mapping constrained category -> [morning_threshold, evening_threshold].
    """
    if constrained_cats is None:
        constrained_cats = set(active_cats)
    thresholds = {}
    for cat_idx, cat in enumerate(active_cats):
        if cat in constrained_cats:
            thresholds[cat] = [
                r_max * 12 * demand_params[cat_idx][0][1],  # morning
                r_max * 12 * demand_params[cat_idx][1][1],  # evening
            ]
    return thresholds
