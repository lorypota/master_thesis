"""
Compute Costs Without Failure Penalty
======================================

This script computes the rebalancing + bike costs WITHOUT the failure penalty.
Uses the same evaluation logic as evaluate_baseline.py but saves costs separately.

Output:
    results/costs_no_penalty_2_cat_3seeds.npy - (11 betas x 3 seeds)
"""

import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add FairMSS root directory to path to import modules
FAIRMSS_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, FAIRMSS_ROOT)

from beta.environment import FairEnv
from common.agent import RebalancingAgent
from common.network import generate_network
from common.demand import generate_global_demand
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
NUM_EVAL_DAYS = 101  # Days to run evaluation (first is skipped)
GAMMA = 20           # Rebalancing cost coefficient

# Demand parameters
REMOTE_DEMAND_PARAMS = [(0.3, 2), (1.5, 0.3)]
CENTRAL_DEMAND_PARAMS = [(13.8, 3.6), (6.6, 13.8)]

# Time slots for rebalancing decisions
TIME_SLOTS = [(0, 12), (12, 24)]

# Cost parameters for rebalancing
REBALANCING_COST_REMOTE = 1.0
REBALANCING_COST_CENTRAL = 0.1

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

# Path to baseline results
BASELINE_DIR = os.path.join(SCRIPT_DIR, '..', 'baseline')

# =============================================================================
# MAIN
# =============================================================================

def main():
    costs_no_penalty = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        beta_index = int(beta * 10)
        print(f"\nEvaluating beta = {beta}")

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)

            # Load number of bikes
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

            # Load trained Q-tables
            agent_remote = RebalancingAgent(0)
            agent_central = RebalancingAgent(4)

            q_table_remote = os.path.join(BASELINE_DIR, 'q_tables', f'q_table_{beta}_2_{seed}_cat0.pkl')
            q_table_central = os.path.join(BASELINE_DIR, 'q_tables', f'q_table_{beta}_2_{seed}_cat4.pkl')
            with open(q_table_remote, "rb") as f:
                agent_remote.q_table = pickle.load(f)
            with open(q_table_central, "rb") as f:
                agent_central.q_table = pickle.load(f)

            agent_remote.set_epsilon(0.0)
            agent_central.set_epsilon(0.0)

            # Initialize environment
            eval_env = FairEnv(G, transformed_demand, beta, GAMMA, STATION_PARAMS)
            state = eval_env.reset()

            daily_rebalancing_costs = []

            # Run evaluation
            for day in range(NUM_EVAL_DAYS):
                costs = 0

                for time_period in (0, 1):
                    actions = np.zeros(NUM_STATIONS, dtype=np.int64)
                    if day > 0:
                        for station in range(NUM_STATIONS):
                            if G.nodes[station]['station'] == 0:
                                actions[station] = agent_remote.decide_action(state[station])
                            else:
                                actions[station] = agent_central.decide_action(state[station])

                    next_state, reward, failures = eval_env.step(actions)

                    # Calculate rebalancing costs only
                    for station, action in enumerate(actions):
                        if action != 0:
                            if station < NUM_REMOTE:
                                costs += REBALANCING_COST_REMOTE
                            else:
                                costs += REBALANCING_COST_CENTRAL

                    state = next_state

                if day > 0:
                    daily_rebalancing_costs.append(costs)

            # Cost = rebalancing_cost + bike_cost (NO failure penalty)
            total_cost = np.mean(daily_rebalancing_costs) + n_bikes / 100

            costs_no_penalty[beta_index].append(total_cost)
            print(f"Cost (no penalty)={total_cost:.2f}")

    # Save results
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, 'costs_no_penalty_2_cat_3seeds.npy')
    np.save(output_file, costs_no_penalty)
    print(f"\nSaved: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary (Costs without failure penalty):")
    print("-" * 40)
    for i, beta in enumerate(BETAS):
        mean_cost = np.mean(costs_no_penalty[i])
        std_cost = np.std(costs_no_penalty[i])
        print(f"Beta {beta:.1f}: {mean_cost:.2f} ± {std_cost:.2f}")


if __name__ == "__main__":
    main()
