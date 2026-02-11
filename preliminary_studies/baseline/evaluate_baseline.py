"""
Baseline Evaluation
========================================================

This script evaluates the trained Q-learning policies and computes:
1. Failure rates per station category (remote vs central)
2. Gini coefficient measuring inequality between categories
3. Global service cost (rebalancing + bikes + failure penalty)

The evaluation runs each trained policy for 100 days and aggregates metrics.

Configuration:
- Uses Q-tables trained with run_baseline_reproduction.py
- 3 seeds x 11 beta values = 33 evaluations
- 2-category case (remote + central)

Output (saved to results/):
    results/gini_2_cat_3seeds.npy  - Gini coefficients (11 betas x 3 seeds)
    results/cost_2_cat_3seeds.npy  - Service costs (11 betas x 3 seeds)
"""

import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add FairMSS root directory to path to import modules
FAIRMSS_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, FAIRMSS_ROOT)

from environment import FairEnv
from agent import RebalancingAgent
from network import generate_network
from demand import generate_global_demand
import numpy as np
import random
import pickle
import inequalipy as ineq

# =============================================================================
# CONFIGURATION
# =============================================================================

# Seeds used in training
SEEDS = [100, 101, 102]

# Beta values
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Network configuration
NUM_REMOTE = 60      # Category 0 stations (underserved areas)
NUM_CENTRAL = 10     # Category 4 stations (well-served areas)
NUM_STATIONS = NUM_REMOTE + NUM_CENTRAL

# Evaluation parameters
NUM_EVAL_DAYS = 101  # Days to run evaluation (first is skipped)
GAMMA = 20           # Rebalancing cost coefficient

# Demand parameters (Skellam distribution: difference of two Poisson)
# Format: [(mu1_morning, mu2_morning), (mu1_evening, mu2_evening)]
REMOTE_DEMAND_PARAMS = [(0.3, 2), (1.5, 0.3)]    # Low demand, outflow in morning
CENTRAL_DEMAND_PARAMS = [(13.8, 3.6), (6.6, 13.8)]  # High demand, inflow in morning

# Time slots for rebalancing decisions
TIME_SLOTS = [(0, 12), (12, 24)]  # Morning and evening windows

# Cost parameters for rebalancing
REBALANCING_COST_REMOTE = 1.0   # Higher cost to rebalance remote stations
REBALANCING_COST_CENTRAL = 0.1  # Lower cost to rebalance central stations

# Station reward parameters for 2-category scenario
# Derived from Skellam demand params: target = expected occupancy, threshold = 0.5 * expected arrivals
STATION_PARAMS = {
    0: {
        'chi': 1, 'phi': 1,
        'evening_target': 22, 'evening_threshold': 0.4,
        'morning_target': 2, 'morning_threshold': 8,
    },
    4: {
        'chi': -1, 'phi': 0.1,
        'evening_target': 0, 'evening_threshold': 61,
        'morning_target': 88, 'morning_threshold': 1,
    },
}

# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def main():
    # Storage for results: gini_values_tot[beta_index] = [seed0, seed1, seed2]
    gini_values_tot = [[] for _ in range(len(BETAS))]
    costs_tot = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        beta_index = int(beta * 10)
        print(f"\nEvaluating beta = {beta}")

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")

            # Set random seeds for reproducibility
            np.random.seed(seed)
            random.seed(seed)

            # Load number of bikes from training results (local path)
            bikes_file = os.path.join(SCRIPT_DIR, 'results', f'bikes_2_cat_{beta}_{seed}.npy')
            n_bikes = np.load(bikes_file)

            # Generate network and demand
            G = generate_network([NUM_REMOTE, NUM_CENTRAL])
            all_days_demand, transformed_demand = generate_global_demand(
                [NUM_REMOTE, NUM_CENTRAL],
                NUM_EVAL_DAYS,
                [REMOTE_DEMAND_PARAMS, CENTRAL_DEMAND_PARAMS],
                TIME_SLOTS
            )

            # Load trained Q-tables (local path)
            agent_remote = RebalancingAgent(0)
            agent_central = RebalancingAgent(4)

            q_table_remote = os.path.join(SCRIPT_DIR, 'q_tables', f'q_table_{beta}_2_{seed}_cat0.pkl')
            q_table_central = os.path.join(SCRIPT_DIR, 'q_tables', f'q_table_{beta}_2_{seed}_cat4.pkl')
            with open(q_table_remote, "rb") as f:
                agent_remote.q_table = pickle.load(f)
            with open(q_table_central, "rb") as f:
                agent_central.q_table = pickle.load(f)

            # Set to greedy policy (no exploration during evaluation)
            agent_remote.set_epsilon(0.0)
            agent_central.set_epsilon(0.0)

            # Initialize environment
            eval_env = FairEnv(G, transformed_demand, beta, GAMMA, STATION_PARAMS)
            state = eval_env.reset()

            # Tracking metrics
            daily_central_failures = []
            daily_remote_failures = []
            daily_global_failures = []
            daily_global_costs = []

            # Run evaluation for NUM_EVAL_DAYS days
            for day in range(NUM_EVAL_DAYS):
                central_fails = 0
                remote_fails = 0
                global_fails = 0
                costs = 0

                for time_period in (0, 1):  # Morning and evening
                    # Determine actions using trained policy
                    actions = np.zeros(NUM_STATIONS, dtype=np.int64)
                    if day > 0:  # Skip first day (initialization)
                        for station in range(NUM_STATIONS):
                            if G.nodes[station]['station'] == 0:  # Remote
                                actions[station] = agent_remote.decide_action(state[station])
                            else:  # Central
                                actions[station] = agent_central.decide_action(state[station])

                    # Execute actions and observe results
                    next_state, reward, failures = eval_env.step(actions)

                    # Accumulate failures
                    remote_fails += np.sum(failures[:NUM_REMOTE])
                    central_fails += np.sum(failures[NUM_REMOTE:])
                    global_fails += np.sum(failures)

                    # Calculate rebalancing costs
                    for station, action in enumerate(actions):
                        if action != 0:  # Rebalancing occurred
                            if station < NUM_REMOTE:
                                costs += REBALANCING_COST_REMOTE
                            else:
                                costs += REBALANCING_COST_CENTRAL

                    state = next_state

                # Record daily metrics (skip day 0)
                if day > 0:
                    daily_central_failures.append(central_fails / NUM_CENTRAL)
                    daily_remote_failures.append(remote_fails / NUM_REMOTE)
                    daily_global_failures.append(global_fails)
                    daily_global_costs.append(costs)

            # Calculate total requests for failure rate computation
            central_requests = 0
            remote_requests = 0
            global_requests = 0

            # Skip day 0 to match the failure calculation (day 0 is warm-up)
            for day in range(1, NUM_EVAL_DAYS):
                for hour in range(24):
                    for station in range(NUM_STATIONS):
                        demand = all_days_demand[day][station][hour]
                        if demand < 0:  # Negative = bike request (outflow)
                            if station < NUM_REMOTE:
                                remote_requests += abs(demand)
                            else:
                                central_requests += abs(demand)
                            global_requests += abs(demand)

            # Normalize by days and stations (NUM_EVAL_DAYS - 1 because day 0 is skipped)
            num_evaluated_days = NUM_EVAL_DAYS - 1
            central_requests = central_requests / num_evaluated_days / NUM_CENTRAL
            remote_requests = remote_requests / num_evaluated_days / NUM_REMOTE
            global_requests = global_requests / num_evaluated_days

            # Compute failure rates (percentage)
            failure_rate_central = np.mean(daily_central_failures) / central_requests * 100
            failure_rate_remote = np.mean(daily_remote_failures) / remote_requests * 100
            failure_rate_global = np.mean(daily_global_failures) / global_requests * 100

            # Compute Gini coefficient
            # Gini = 0 means perfect equality, Gini > 0 means inequality
            gini = np.round(ineq.gini([failure_rate_central, failure_rate_remote]), 3)

            # Compute total cost
            # cost = rebalancing_cost + bike_cost + failure_penalty
            total_cost = np.mean(daily_global_costs) + n_bikes / 100 + failure_rate_global / 10

            # Store results
            gini_values_tot[beta_index].append(gini)
            costs_tot[beta_index].append(total_cost)

            print(f"Gini={gini:.3f}, Cost={total_cost:.2f}")

    # Save results (local path)
    print("\n" + "=" * 60)
    print("Saving results...")
    gini_file = os.path.join(SCRIPT_DIR, 'results', 'gini_2_cat_3seeds.npy')
    cost_file = os.path.join(SCRIPT_DIR, 'results', 'cost_2_cat_3seeds.npy')
    np.save(gini_file, gini_values_tot)
    np.save(cost_file, costs_tot)
    print(f"Saved: {gini_file} and {cost_file}")


if __name__ == "__main__":
    main()
