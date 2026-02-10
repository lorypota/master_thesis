"""
Failure Rate Analysis
==========================================

This script evaluates trained Q-learning policies and extracts per-category failure rates.

Unlike the baseline evaluation which computes Gini coefficients, this script
saves the raw failure rates for remote and central areas separately.

Configuration:
- Uses Q-tables and bikes data from ../baseline/
- Same 3 seeds x 11 beta values = 33 evaluations
- 2-category case (remote + central)

Output:
    results/failure_rate_remote_2_cat_3seeds.npy  - Remote failure rates (11 betas x 3 seeds)
    results/failure_rate_central_2_cat_3seeds.npy - Central failure rates (11 betas x 3 seeds)
"""

import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add FairMSS root directory to path to import modules
FAIRMSS_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, FAIRMSS_ROOT)

from environment_2 import FairEnv
from agent import RebalancingAgent
from network import generate_network
from demand import generate_global_demand
import numpy as np
import random
import pickle

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
NUM_STATIONS = NUM_REMOTE + NUM_CENTRAL  # Total: 70

# Evaluation parameters
NUM_EVAL_DAYS = 101  # Days to run evaluation (1 discarded later)
GAMMA = 20           # Rebalancing cost coefficient

# Demand parameters (Skellam distribution: difference of two Poisson)
# Format: [(mu1_morning, mu2_morning), (mu1_evening, mu2_evening)]
REMOTE_DEMAND_PARAMS = [(0.3, 2), (1.5, 0.3)]    # Low demand, outflow in morning
CENTRAL_DEMAND_PARAMS = [(13.8, 3.6), (6.6, 13.8)]  # High demand, inflow in morning

# Time slots for rebalancing decisions
TIME_SLOTS = [(0, 12), (12, 24)]  # Morning and evening windows

# Path to baseline results (relative to this script)
BASELINE_DIR = os.path.join(SCRIPT_DIR, '..', 'baseline')

# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def main():
    # Storage for results: failure_rates_remote[beta_index] = [seed0, seed1, seed2]
    failure_rates_remote = [[] for _ in range(len(BETAS))]
    failure_rates_central = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        beta_index = int(beta * 10)
        print(f"\nEvaluating beta = {beta}")

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")

            # Set random seeds for reproducibility
            np.random.seed(seed)
            random.seed(seed)

            # Load number of bikes from baseline results
            bikes_file = os.path.join(BASELINE_DIR, 'results', f'bikes_2_cat_{beta}_{seed}.npy')
            n_bikes = np.load(bikes_file)

            # Generate network and demand
            G = generate_network([NUM_REMOTE, NUM_CENTRAL])
            all_days_demand, transformed_demand = generate_global_demand(
                [NUM_REMOTE, NUM_CENTRAL],
                NUM_EVAL_DAYS,
                [REMOTE_DEMAND_PARAMS, CENTRAL_DEMAND_PARAMS],
                TIME_SLOTS
            )

            # Load trained Q-tables from baseline
            agent_remote = RebalancingAgent(0)
            agent_central = RebalancingAgent(4)

            q_table_remote = os.path.join(BASELINE_DIR, 'q_tables', f'q_table_{beta}_2_{seed}_cat0.pkl')
            q_table_central = os.path.join(BASELINE_DIR, 'q_tables', f'q_table_{beta}_2_{seed}_cat4.pkl')
            with open(q_table_remote, "rb") as f:
                agent_remote.q_table = pickle.load(f)
            with open(q_table_central, "rb") as f:
                agent_central.q_table = pickle.load(f)

            # Set to greedy policy (no exploration during evaluation)
            agent_remote.set_epsilon(0.0)
            agent_central.set_epsilon(0.0)

            # Initialize environment
            eval_env = FairEnv(G, transformed_demand, beta, GAMMA)
            state = eval_env.reset()

            # Tracking metrics
            daily_central_failures = []
            daily_remote_failures = []

            # Run evaluation for NUM_EVAL_DAYS days
            for day in range(NUM_EVAL_DAYS):
                central_fails = 0
                remote_fails = 0

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

                    state = next_state

                # Record daily metrics (skip day 0)
                if day > 0:
                    daily_central_failures.append(central_fails / NUM_CENTRAL)
                    daily_remote_failures.append(remote_fails / NUM_REMOTE)

            # Calculate total requests for failure rate computation
            central_requests = 0
            remote_requests = 0

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

            # Normalize by days and stations (NUM_EVAL_DAYS - 1 because day 0 is skipped)
            num_evaluated_days = NUM_EVAL_DAYS - 1
            central_requests = central_requests / num_evaluated_days / NUM_CENTRAL
            remote_requests = remote_requests / num_evaluated_days / NUM_REMOTE

            # Compute failure rates (percentage)
            failure_rate_central = np.mean(daily_central_failures) / central_requests * 100
            failure_rate_remote = np.mean(daily_remote_failures) / remote_requests * 100

            # Store results
            failure_rates_remote[beta_index].append(failure_rate_remote)
            failure_rates_central[beta_index].append(failure_rate_central)

            print(f"Remote FR={failure_rate_remote:.2f}%, Central FR={failure_rate_central:.2f}%")

    # Save results
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    remote_file = os.path.join(results_dir, 'failure_rate_remote_2_cat_3seeds.npy')
    central_file = os.path.join(results_dir, 'failure_rate_central_2_cat_3seeds.npy')

    np.save(remote_file, failure_rates_remote)
    np.save(central_file, failure_rates_central)

    print(f"Saved: {remote_file} and {central_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table:")
    print("-" * 60)
    print(f"{'Beta':<6} {'Remote FR (%)':<15} {'Central FR (%)':<15}")
    print("-" * 60)

    for i, beta in enumerate(BETAS):
        remote_mean = np.mean(failure_rates_remote[i])
        remote_std = np.std(failure_rates_remote[i])
        central_mean = np.mean(failure_rates_central[i])
        central_std = np.std(failure_rates_central[i])

        print(f"{beta:<6.1f} {remote_mean:>6.2f} ± {remote_std:<5.2f}   {central_mean:>6.2f} ± {central_std:<5.2f}")


if __name__ == "__main__":
    main()
