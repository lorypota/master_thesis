"""
Evaluation Script
========================================================

Evaluates trained Q-learning policies for any category scenario (2-5).
Computes failure rates, Gini coefficient, and global service cost.

Usage:
    python evaluation.py --categories 2
    python evaluation.py --categories 5 --seeds 100 110 --save-detailed

Output (saved to results/):
    results/gini_{M}_cat_{N}seeds.npy
    results/cost_{M}_cat_{N}seeds.npy
    (with --save-detailed):
    results/cost_reb_{M}_cat_{N}seeds.npy
    results/cost_fail_{M}_cat_{N}seeds.npy
    results/cost_bikes_{M}_cat_{N}seeds.npy
    results/initial_bikes_{M}_cat_{N}seeds.npy
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

from environment import FairEnv
from common.agent import RebalancingAgent
from common.network import generate_network
from common.demand import generate_global_demand
from common.config import get_scenario, PHI, GAMMA, NUM_EVAL_DAYS, TIME_SLOTS, BETAS
import numpy as np
import random
import pickle
import argparse
import inequalipy as ineq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", required=True, type=int, choices=[2, 3, 4, 5])
    parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110],
                        help="Seed range [start, end) (default: 100 110)")
    parser.add_argument("--save-detailed", action="store_true",
                        help="Save detailed cost breakdowns (rebalancing, failures, bikes, initial_bikes)")
    args = parser.parse_args()

    M = args.categories
    scenario = get_scenario(M)
    node_list = scenario['node_list']
    active_cats = scenario['active_cats']
    demand_params = scenario['demand_params']
    station_params = scenario['station_params']

    num_stations = sum(node_list)

    # Precompute cumulative boundaries for slicing failures by category
    # e.g. node_list=[60, 40, 30, 20, 10] -> boundaries=[0, 60, 100, 130, 150, 160]
    boundaries = np.cumsum([0] + node_list)

    seeds = list(range(args.seeds[0], args.seeds[1]))
    num_seeds = len(seeds)

    # Result storage
    gini_values_tot = [[] for _ in range(len(BETAS))]
    costs_tot = [[] for _ in range(len(BETAS))]
    costs_rebalancing = [[] for _ in range(len(BETAS))]
    costs_failures = [[] for _ in range(len(BETAS))]
    costs_bikes = [[] for _ in range(len(BETAS))]
    initial_bikes = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        index = int(beta * 10)
        print(f"\nEvaluating beta = {beta}")

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)

            n_bikes = np.load(os.path.join(SCRIPT_DIR, f'results/bikes_{M}_cat_{beta}_{seed}.npy'))

            G = generate_network(node_list)
            all_days_demand, transformed_demand = generate_global_demand(
                node_list, NUM_EVAL_DAYS, demand_params, TIME_SLOTS
            )

            # Load trained agents
            agents = {}
            for cat in active_cats:
                agent = RebalancingAgent(cat)
                with open(os.path.join(SCRIPT_DIR, f"q_tables/q_table_{beta}_{M}_{seed}_cat{cat}.pkl"), "rb") as f:
                    agent.q_table = pickle.load(f)
                agent.set_epsilon(0.0)
                agents[cat] = agent

            # Initialize environment
            eval_env = FairEnv(G, transformed_demand, beta, GAMMA, station_params)
            state = eval_env.reset()

            # Per-category daily tracking
            # cat_index maps active_cats to their position in node_list order (0, 1, 2, ...)
            daily_cat_failures = {cat: [] for cat in active_cats}
            daily_global_failures = []
            daily_global_costs = []

            for day in range(NUM_EVAL_DAYS):
                cat_fails = {cat: 0 for cat in active_cats}
                global_fails = 0
                costs = 0

                for time_period in (0, 1):
                    actions = np.zeros(num_stations, dtype=np.int64)
                    if day > 0:
                        for i in range(num_stations):
                            cat = G.nodes[i]['station']
                            actions[i] = agents[cat].decide_action(state[i])

                    next_state, reward, failures = eval_env.step(actions)

                    # Accumulate failures per category using boundaries
                    for idx, cat in enumerate(active_cats):
                        cat_fails[cat] += np.sum(failures[boundaries[idx]:boundaries[idx + 1]])
                    global_fails += np.sum(failures)

                    # Accumulate rebalancing costs using phi
                    for station, action in enumerate(actions):
                        if action != 0:
                            cat = G.nodes[station]['station']
                            costs += PHI[cat]

                    state = next_state

                if day > 0:
                    for idx, cat in enumerate(active_cats):
                        daily_cat_failures[cat].append(cat_fails[cat] / node_list[idx])
                    daily_global_failures.append(global_fails)
                    daily_global_costs.append(costs)

                if day == 0 and args.save_detailed:
                    initial_bikes[index].append(
                        sum(G.nodes[i]['bikes'] for i in range(num_stations))
                    )

            # Compute requests per category
            cat_requests = {cat: 0 for cat in active_cats}
            global_requests = 0

            for day in range(NUM_EVAL_DAYS):
                for hour in range(24):
                    for station in range(num_stations):
                        demand = all_days_demand[day][station][hour]
                        if demand < 0:
                            # Find which category this station belongs to
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
                cat_failure_rates[cat] = np.mean(daily_cat_failures[cat]) / cat_requests[cat] * 100
            failure_rate_global = np.mean(daily_global_failures) / global_requests * 100

            # Gini coefficient (ordered from most central to most remote)
            failure_rates_list = [cat_failure_rates[cat] for cat in reversed(active_cats)]
            gini = np.round(ineq.gini(failure_rates_list), 3)

            # Total cost
            total_cost = np.mean(daily_global_costs) + n_bikes / 100 + failure_rate_global / 10

            gini_values_tot[index].append(gini)
            costs_tot[index].append(total_cost)
            if args.save_detailed:
                costs_rebalancing[index].append(np.mean(daily_global_costs))
                costs_failures[index].append(failure_rate_global)
                costs_bikes[index].append(n_bikes)

            print(f"Gini={gini:.3f}, Cost={total_cost:.2f}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    np.save(os.path.join(results_dir, f'gini_{M}_cat_{num_seeds}seeds.npy'), gini_values_tot)
    np.save(os.path.join(results_dir, f'cost_{M}_cat_{num_seeds}seeds.npy'), costs_tot)

    if args.save_detailed:
        np.save(os.path.join(results_dir, f'cost_reb_{M}_cat_{num_seeds}seeds.npy'), costs_rebalancing)
        np.save(os.path.join(results_dir, f'cost_fail_{M}_cat_{num_seeds}seeds.npy'), costs_failures)
        np.save(os.path.join(results_dir, f'cost_bikes_{M}_cat_{num_seeds}seeds.npy'), costs_bikes)
        np.save(os.path.join(results_dir, f'initial_bikes_{M}_cat_{num_seeds}seeds.npy'), initial_bikes)


if __name__ == "__main__":
    main()