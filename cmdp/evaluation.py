"""
CMDP Evaluation Script
========================================================

Evaluates trained CMDP Q-learning policies for any category scenario (2-5).
Computes failure rates, Gini coefficient, global service cost,
and constraint satisfaction for constrained categories.

Usage:
    uv run evaluation.py --categories 2
    uv run evaluation.py --categories 2 --r-max-values 0.05 0.10 0.15 0.20 0.25
    uv run evaluation.py --categories 5 --seeds 100 110 --save-detailed

Output (saved to results/):
    results/gini_{M}_cat_{N}seeds.npy
    results/cost_{M}_cat_{N}seeds.npy
    results/constraint_sat_{M}_cat_{N}seeds.npy
    (with --save-detailed):
    results/cost_reb_{M}_cat_{N}seeds.npy
    results/cost_fail_{M}_cat_{N}seeds.npy
    results/cost_bikes_{M}_cat_{N}seeds.npy
    results/initial_bikes_{M}_cat_{N}seeds.npy
"""

import argparse
import os
import pickle
import random

import inequalipy as ineq
import numpy as np

from cmdp.environment import CMDPEnv
from common.agent import RebalancingAgent
from common.config import GAMMA, NUM_EVAL_DAYS, PHI, TIME_SLOTS, get_scenario
from common.demand import generate_global_demand
from common.network import generate_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
R_MAX_VALUES = [round(r * 0.1, 1) for r in range(11)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", required=True, type=int, choices=[2, 3, 4, 5])
    parser.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        default=[100, 110],
        help="Seed range [start, end) (default: 100 110)",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed cost breakdowns (rebalancing, failures, bikes, initial_bikes)",
    )
    parser.add_argument(
        "--constrained-cats",
        nargs="+",
        type=int,
        default=[0],
        help="Category indices that were constrained during training",
    )
    parser.add_argument(
        "--r-max-values",
        nargs="+",
        type=float,
        default=None,
        help="r_max values to evaluate (default: 0.0 to 1.0 in 0.1 steps)",
    )
    args = parser.parse_args()

    M = args.categories
    scenario = get_scenario(M)
    node_list = scenario["node_list"]
    active_cats = scenario["active_cats"]
    demand_params = scenario["demand_params"]
    station_params = scenario["station_params"]
    constrained_cats = set(args.constrained_cats)

    r_max_values = args.r_max_values if args.r_max_values else R_MAX_VALUES

    num_stations = sum(node_list)
    boundaries = np.cumsum([0] + node_list)

    seeds = list(range(args.seeds[0], args.seeds[1]))
    num_seeds = len(seeds)

    # Result storage
    num_r_max = len(r_max_values)
    gini_values_tot = [[] for _ in range(num_r_max)]
    costs_tot = [[] for _ in range(num_r_max)]
    costs_rebalancing = [[] for _ in range(num_r_max)]
    costs_failures = [[] for _ in range(num_r_max)]
    costs_bikes = [[] for _ in range(num_r_max)]
    initial_bikes = [[] for _ in range(num_r_max)]
    constraint_satisfaction = [[] for _ in range(num_r_max)]

    for r_idx, r_max in enumerate(r_max_values):
        print(f"\nEvaluating r_max = {r_max}")

        # Pre-compute failure thresholds for constraint checking
        failure_thresholds = {}
        for cat_idx, cat in enumerate(active_cats):
            if cat in constrained_cats:
                failure_thresholds[cat] = [
                    r_max * 12 * demand_params[cat_idx][0][1],  # morning
                    r_max * 12 * demand_params[cat_idx][1][1],  # evening
                ]

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)

            n_bikes = np.load(
                os.path.join(SCRIPT_DIR, f"results/bikes_{M}_cat_{r_max}_{seed}.npy")
            )

            G = generate_network(node_list)
            all_days_demand, transformed_demand = generate_global_demand(
                node_list, NUM_EVAL_DAYS, demand_params, TIME_SLOTS
            )

            # Load trained agents
            agents = {}
            for cat in active_cats:
                agent = RebalancingAgent(cat)
                with open(
                    os.path.join(
                        SCRIPT_DIR, f"q_tables/q_table_{r_max}_{M}_{seed}_cat{cat}.pkl"
                    ),
                    "rb",
                ) as f:
                    agent.q_table = pickle.load(f)
                agent.set_epsilon(0.0)
                agents[cat] = agent

            # Initialize environment (lambdas={} — rewards don't matter for greedy eval)
            eval_env = CMDPEnv(G, transformed_demand, {}, GAMMA, station_params)
            state = eval_env.reset()

            # Per-category daily tracking
            daily_cat_failures = {cat: [] for cat in active_cats}
            daily_global_failures = []
            daily_global_costs = []

            # Per-category per-period failure tracking for constraint check
            period_cat_failures = {}
            for cat in constrained_cats:
                if cat in active_cats:
                    period_cat_failures[cat] = {0: [], 1: []}

            for day in range(NUM_EVAL_DAYS):
                cat_fails = {cat: 0 for cat in active_cats}
                global_fails = 0
                costs = 0

                # Per-period tracking within this day
                period_fails_today = {
                    cat: {0: 0.0, 1: 0.0} for cat in period_cat_failures
                }

                for _time_period in (0, 1):
                    actions = np.zeros(num_stations, dtype=np.int64)
                    if day > 0:
                        for i in range(num_stations):
                            cat = G.nodes[i]["station"]
                            actions[i] = agents[cat].decide_action(state[i])

                    next_state, reward, failures = eval_env.step(actions)
                    period = eval_env.current_period

                    # Accumulate failures per category
                    for idx, cat in enumerate(active_cats):
                        cat_fails[cat] += np.sum(
                            failures[boundaries[idx] : boundaries[idx + 1]]
                        )
                    global_fails += np.sum(failures)

                    # Per-period per-category accumulation for constrained cats
                    for idx, cat in enumerate(active_cats):
                        if cat in period_cat_failures:
                            pf = (
                                np.sum(failures[boundaries[idx] : boundaries[idx + 1]])
                                / node_list[idx]
                            )
                            period_fails_today[cat][period] += pf

                    # Accumulate rebalancing costs
                    for station, action in enumerate(actions):
                        if action != 0:
                            cat = G.nodes[station]["station"]
                            costs += PHI[cat]

                    state = next_state

                if day > 0:
                    for idx, cat in enumerate(active_cats):
                        daily_cat_failures[cat].append(cat_fails[cat] / node_list[idx])
                    daily_global_failures.append(global_fails)
                    daily_global_costs.append(costs)

                    for cat in period_cat_failures:
                        for p in (0, 1):
                            period_cat_failures[cat][p].append(
                                period_fails_today[cat][p]
                            )

                if day == 0 and args.save_detailed:
                    initial_bikes[r_idx].append(
                        sum(G.nodes[i]["bikes"] for i in range(num_stations))
                    )

            # Compute requests per category
            cat_requests = {cat: 0 for cat in active_cats}
            global_requests = 0

            for day in range(NUM_EVAL_DAYS):
                for hour in range(24):
                    for station in range(num_stations):
                        demand = all_days_demand[day][station][hour]
                        if demand < 0:
                            for idx, cat in enumerate(active_cats):
                                if boundaries[idx] <= station < boundaries[idx + 1]:
                                    cat_requests[cat] += abs(demand)
                                    break
                            global_requests += abs(demand)

            # Normalize requests
            for idx, cat in enumerate(active_cats):
                cat_requests[cat] = cat_requests[cat] / NUM_EVAL_DAYS / node_list[idx]
            global_requests = global_requests / NUM_EVAL_DAYS

            # Compute failure rates
            cat_failure_rates = {}
            for cat in active_cats:
                cat_failure_rates[cat] = (
                    np.mean(daily_cat_failures[cat]) / cat_requests[cat] * 100
                )
            failure_rate_global = np.mean(daily_global_failures) / global_requests * 100

            # Gini coefficient (ordered from most central to most remote)
            failure_rates_list = [
                cat_failure_rates[cat] for cat in reversed(active_cats)
            ]
            gini = np.round(ineq.gini(failure_rates_list), 3)

            # Total cost
            total_cost = (
                np.mean(daily_global_costs) + n_bikes / 100 + failure_rate_global / 10
            )

            # Constraint satisfaction check
            satisfied = True
            for cat in period_cat_failures:
                for p in (0, 1):
                    avg_fail = np.mean(period_cat_failures[cat][p])
                    threshold = failure_thresholds[cat][p]
                    if avg_fail > threshold:
                        satisfied = False

            gini_values_tot[r_idx].append(gini)
            costs_tot[r_idx].append(total_cost)
            constraint_satisfaction[r_idx].append(satisfied)
            if args.save_detailed:
                costs_rebalancing[r_idx].append(np.mean(daily_global_costs))
                costs_failures[r_idx].append(failure_rate_global)
                costs_bikes[r_idx].append(n_bikes)

            constraint_str = "SAT" if satisfied else "VIOL"
            print(
                f"Gini={gini:.3f}, Cost={total_cost:.2f}, Constraint={constraint_str}"
            )

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"gini_{M}_cat_{num_seeds}seeds.npy"),
        gini_values_tot,
    )
    np.save(os.path.join(results_dir, f"cost_{M}_cat_{num_seeds}seeds.npy"), costs_tot)
    np.save(
        os.path.join(results_dir, f"constraint_sat_{M}_cat_{num_seeds}seeds.npy"),
        constraint_satisfaction,
    )

    if args.save_detailed:
        np.save(
            os.path.join(results_dir, f"cost_reb_{M}_cat_{num_seeds}seeds.npy"),
            costs_rebalancing,
        )
        np.save(
            os.path.join(results_dir, f"cost_fail_{M}_cat_{num_seeds}seeds.npy"),
            costs_failures,
        )
        np.save(
            os.path.join(results_dir, f"cost_bikes_{M}_cat_{num_seeds}seeds.npy"),
            costs_bikes,
        )
        np.save(
            os.path.join(results_dir, f"initial_bikes_{M}_cat_{num_seeds}seeds.npy"),
            initial_bikes,
        )


if __name__ == "__main__":
    main()
