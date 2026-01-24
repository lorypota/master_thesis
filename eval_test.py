from environment_2 import FairEnv
from agent import RebalancingAgent
from network import generate_network
from demand import generate_global_demand
import numpy as np
import random
import pickle
import inequalipy as ineq

# Initialize lists for ALL metrics needed by boxplots.py
gini_values_tot = [[], [], [], [], [], [], [], [], [], [], []]
costs_tot = [[], [], [], [], [], [], [], [], [], [], []]
costs_reb = [[], [], [], [], [], [], [], [], [], [], []]   # NEW
costs_fail = [[], [], [], [], [], [], [], [], [], [], []]  # NEW
costs_bikes = [[], [], [], [], [], [], [], [], [], [], []] # NEW
init_bikes = [] # NEW

test_betas = [0.0, 0.5, 1.0]

print("Starting FULL evaluation for test run...")

for beta in test_betas:
    index = int(beta * 10)
    for seed in [100]: 
        print(f"Evaluating Beta: {beta}, Seed: {seed}...")
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Load bike count
        n_bikes = np.load(f'results/bikes_2_cat_{beta}_{seed}.npy')
        if beta == 0.0: init_bikes.append(n_bikes) # Save for reference

        gamma = 20
        num_central = 10
        num_remote = 60
        num_days = 101
        time_slots = [(0, 12), (12, 24)]
        remote_params = [(0.3, 2), (1.5, 0.3)]
        central_params = [(13.8, 3.6), (6.6, 13.8)]

        G = generate_network([num_remote, num_central])
        all_days_demand_vectors, transformed_demand_vectors = generate_global_demand(
            [num_remote, num_central], num_days, [remote_params, central_params], time_slots
        )

        agent_0 = RebalancingAgent(0)
        agent_4 = RebalancingAgent(4)

        with open(f"q_tables/q_table_{beta}_2_{seed}_cat0.pkl", "rb") as file:
            agent_0.q_table = pickle.load(file)
        with open(f"q_tables/q_table_{beta}_2_{seed}_cat4.pkl", "rb") as file:
            agent_4.q_table = pickle.load(file)

        num_stations = 70
        agent_0.set_epsilon(0.0)
        agent_4.set_epsilon(0.0)

        eval_env = FairEnv(G, transformed_demand_vectors, beta, gamma)
        state = eval_env.reset()

        # Temporary lists for this run
        run_costs = []
        run_reb_ops = []
        run_failures = []
        run_fails_central = []
        run_fails_remote = []

        for day in range(101):
            day_reb = 0
            day_fail = 0
            day_cost = 0
            central_fails = 0
            rem_fails = 0
            
            for time in (0, 1):
                actions = np.zeros(num_stations, dtype=np.int64)
                if day > 0:
                    for i in range(num_stations):
                        if G.nodes[i]['station'] == 0:
                            actions[i] = agent_0.decide_action(state[i])
                        else:
                            actions[i] = agent_4.decide_action(state[i])

                next_state, reward, failures = eval_env.step(actions)
                
                # Metric Accumulation
                central_fails += np.sum(failures[60:])
                rem_fails += np.sum(failures[:60])
                day_fail += np.sum(failures)

                for a in range(len(actions)):
                    if actions[a] != 0:
                        day_reb += 1 # Count operations
                        # Cost calculation per paper: 1 for remote, 0.1 for central
                        if a < 60: day_cost += 1
                        else: day_cost += 0.1

                state = next_state

            if day > 0:
                run_costs.append(day_cost)
                run_reb_ops.append(day_cost) # Paper uses weighted cost as "ops" in boxplot labels sometimes
                run_failures.append(day_fail)
                run_fails_central.append(central_fails/10)
                run_fails_remote.append(rem_fails/60)

        # --- CALCULATE FINAL METRICS FOR THIS SEED ---
        
        # 1. Gini
        central_req = 0; rem_req = 0
        for day in range(101):
            for i in range(24):
                for j in range(70):
                    val = abs(all_days_demand_vectors[day][j][i])
                    if j < 60: rem_req += val
                    else: central_req += val
        
        # Normalize requests
        central_req = central_req / 101 / 10
        rem_req = rem_req / 101 / 60
        
        fail_rate_cen = np.mean(run_fails_central) / central_req * 100
        fail_rate_rem = np.mean(run_fails_remote) / rem_req * 100
        
        gini = np.round(ineq.gini([fail_rate_cen, fail_rate_rem]), 3)
        
        # 2. Append to Global Lists
        gini_values_tot[index].append(gini)
        
        # Cost Components for Boxplots
        # Total Cost = Rebalancing + Vehicles + Failures
        c_reb = np.mean(run_costs)
        c_fail = np.mean(run_failures)/10
        c_bike = n_bikes/100
        
        costs_tot[index].append(c_reb + c_bike + c_fail)
        costs_reb[index].append(c_reb)
        costs_fail[index].append(np.mean(run_failures)) # Raw failure count or rate
        costs_bikes[index].append(n_bikes)

# Save ALL files needed for boxplots
print("Saving detailed results...")
np.save('results/gini_2_cat_TEST.npy', np.array(gini_values_tot, dtype=object))
np.save('results/cost_2_cat_TEST.npy', np.array(costs_tot, dtype=object))
np.save('results/cost_reb_2_cat_TEST.npy', np.array(costs_reb, dtype=object))
np.save('results/cost_fail_2_cat_TEST.npy', np.array(costs_fail, dtype=object))
np.save('results/cost_bikes_2_cat_TEST.npy', np.array(costs_bikes, dtype=object))
np.save('results/initial_bikes_2_cat_TEST.npy', np.array(init_bikes))