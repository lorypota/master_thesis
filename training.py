from agent import RebalancingAgent
from network import generate_network
from demand import generate_global_demand
from environment import FairEnv
import numpy as np
import random
import pickle
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--beta", default=0, type=float)
parser.add_argument("--categories", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--output-dir", default=".", type=str, help="Output directory for results")
args = parser.parse_args()

beta = args.beta / 10
gamma = 20
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, f'training_times_{args.categories}.txt')

num_days = 1000
time_slots = [(0, 12), (12, 24)]

# =============================================================================
# SCENARIO CONFIGURATIONS

# station_params per category: (chi, phi, eve_target, eve_thresh, morn_target, morn_thresh)
# chi = fairness penalty weight (multiplied by beta)
# phi = rebalancing cost weight (multiplied by gamma)
# eve/morn target/thresh = clutter penalty parameters
# =============================================================================

def _build_station_params(raw):
    """Convert compact tuples to the dict format expected by FairEnv."""
    return {
        cat: {
            'chi': chi, 'phi': phi,
            'evening_target': eve_t, 'evening_threshold': eve_th,
            'morning_target': morn_t, 'morning_threshold': morn_th,
        }
        for cat, (chi, phi, eve_t, eve_th, morn_t, morn_th) in raw.items()
    }
    
SCENARIOS = {
    2: {
        'demand_params': [
            [(0.3, 2), (1.5, 0.3)],       # cat 0 (peripheral)
            [(13.8, 3.6), (6.6, 13.8)],   # cat 4 (central)
        ],
        'node_list': [60, 10],
        'active_cats': [0, 4],
        #        chi    phi     eve_target  eve_thresh  morn_target  morn_thresh
        'station_params': {
            0: ( 1,     1,      22,         0.4,        2,           8),
            4: (-1,     0.1,    0,          61,         88,          1),
        },
    },
    3: {
        'demand_params': [
            [(0.3, 2), (1.5, 0.3)],       # cat 0
            [(3.3, 1.5), (1.5, 3.3)],     # cat 2
            [(13.8, 9), (12, 13.8)],      # cat 4
        ],
        'node_list': [60, 30, 10],
        'active_cats': [0, 2, 4],
        'station_params': {
            0: ( 1,   1,     22,       0.4,        2,          8),
            2: ( 0.4, 0.4,   3,        12,         25,         2),
            4: (-1,   0.1,   5,        30,         36,         7),
        },
    },
    4: {
        'demand_params': [
            [(0.3, 2), (1.5, 0.3)],        # cat 0
            [(0.45, 3), (2.25, 0.45)],     # cat 1
            [(9.2, 2.4), (4.4, 9.2)],      # cat 3
            [(13.8, 7), (9, 13.8)],        # cat 4
        ],
        'node_list': [60, 40, 20, 10],
        'active_cats': [0, 1, 3, 4],
        'station_params': {
            0: ( 1,   1,    22,       0.4,        2,          8),
            1: ( 0.5, 0.8,  32,       0.3,        1,          11),
            3: (-0.5, 0.3,  0.3,      41,         60,         1),
            4: (-1,   0.1,  2,        42,         64,         3),
        },
    },
    5: {
        'demand_params': [
            [(0.3, 2), (1.5, 0.3)],        # cat 0
            [(0.45, 3), (2.25, 0.45)],     # cat 1
            [(3.3, 1.5), (1.5, 3.3)],      # cat 2
            [(9.2, 5.1), (6.6, 9.2)],      # cat 3
            [(13.8, 7), (10, 13.8)],       # cat 4
        ],
        'node_list': [60, 40, 30, 20, 10],
        'active_cats': [0, 1, 2, 3, 4],
        'station_params': {
            0: ( 1,   1,    22,        0.4,        2,          8),
            1: ( 0.5, 0.8,  32,        0.3,        1,          11),
            2: ( 0.4, 0.4,   3,        12,         25,         2),
            3: (-0.5, 0.3,   3,        26,         39,         3),
            4: (-1,   0.1,   2,        42,         54,         3),
        },
    },
}

# =============================================================================
# SETUP
# =============================================================================
if args.categories not in SCENARIOS:
    raise ValueError("Wrong number of categories (can be 2, 3, 4 or 5).")

scenario = SCENARIOS[args.categories]
demand_params = scenario['demand_params']
node_list = scenario['node_list']
active_cats = scenario['active_cats']
station_params = _build_station_params(scenario['station_params'])

agents = {i: RebalancingAgent(i) for i in range(5)}

G = generate_network(node_list)
all_days_demand_vectors, transformed_demand_vectors = generate_global_demand(node_list, num_days,
                                                                             demand_params, time_slots)

num_stations = np.sum(node_list)
daily_returns = []
daily_failures = []
np.random.seed(args.seed)
random.seed(args.seed)

env = FairEnv(G, transformed_demand_vectors, beta, gamma, station_params)
state = env.reset()

# =============================================================================
# TRAINING LOOP
# =============================================================================

start = time.time()
for repeat in range(110):
    for day in range(num_days):
        ret = 0
        fails = 0
        for times in (0, 1):
            actions = np.zeros(num_stations, dtype=np.int64)
            if not (repeat == 0 and day == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]['station']
                    actions[i] = agents[cat].decide_action(state[i])

            next_state, reward, failures = env.step(actions)
            ret += np.sum(reward)

            fails += np.sum(failures)

            train_until = {0: 19, 1: 28, 2: 37, 3: 55, 4: 110}
            if not (day == 0 and repeat == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]['station']
                    if repeat < train_until[cat]:
                        agents[cat].update_q_table(state[i], actions[i], reward[i], next_state[i])
                        agents[cat].update_epsilon()

            state = next_state

        if repeat == 0 and day == 0:
            pass
        else:
            daily_returns.append(ret)
            daily_failures.append(fails)

end = time.time()
with open(file_path, 'a') as file:
    file.write(f'{end - start},\n')
    
# =============================================================================
# SAVE RESULTS
# =============================================================================

q_tables_dir = os.path.join(output_dir, 'q_tables')
results_dir = os.path.join(output_dir, 'results')
os.makedirs(q_tables_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

for cat in active_cats:
    with open(os.path.join(q_tables_dir, f"q_table_{beta}_{args.categories}_{args.seed}_cat{cat}.pkl"), "wb") as file:
        pickle.dump(agents[cat].q_table, file)

np.save(os.path.join(results_dir, f'learning_curve_{args.categories}_cat_{beta}_{args.seed}.npy'), daily_returns)

print(f'Finished simulation with seed: {args.seed}, categories: {args.categories} and beta: {beta}')

total_bikes = sum(G.nodes[i]['bikes'] for i in range(num_stations))
np.save(os.path.join(results_dir, f'bikes_{args.categories}_cat_{beta}_{args.seed}.npy'), total_bikes)
