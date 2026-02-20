R_MAX_VALUES = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 1.0]

TRAIN_UNTIL = {0: 50, 1: 50, 2: 37, 3: 55, 4: 110}


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
