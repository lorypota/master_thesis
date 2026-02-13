import argparse
import os
import pickle
import random
import time

import numpy as np
import psutil
import wandb

from beta.environment import FairEnv
from common.agent import RebalancingAgent
from common.config import (
    CPU_CORES,
    GAMMA,
    NUM_TRAIN_DAYS,
    TIME_SLOTS,
    TRAIN_UNTIL,
    get_scenario,
)
from common.demand import generate_global_demand
from common.network import generate_network

# Limit resource usage for app-reken12
# os.system(f"procgov64 --nowait --minws 10M --maxws {MAX_MEMORY_MB}M -p {os.getpid()}")
p = psutil.Process()
p.cpu_affinity(
    list(range(int(CPU_CORES.split("-")[0]), int(CPU_CORES.split("-")[1]) + 1))
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--beta", default=0, type=float)
parser.add_argument("--categories", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument(
    "--output-dir", default=SCRIPT_DIR, type=str, help="Output directory for results"
)
parser.add_argument(
    "--run-group", default=None, type=str, help="Wandb group ID for grouping runs"
)
args = parser.parse_args()

beta = args.beta
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# LOAD SCENARIO CONFIG
# =============================================================================

scenario = get_scenario(args.categories)
demand_params = scenario["demand_params"]
node_list = scenario["node_list"]
active_cats = scenario["active_cats"]
station_params = scenario["station_params"]

# =============================================================================
# WANDB
# =============================================================================

wandb.init(
    project="fairmss",
    group=f"beta-{args.categories}cat-{args.run_group}",
    name=f"beta{beta}_seed{args.seed}",
    config={
        "method": "beta",
        "beta": beta,
        "categories": args.categories,
        "seed": args.seed,
        "gamma": GAMMA,
        "num_train_days": NUM_TRAIN_DAYS,
        "node_list": node_list,
        "active_cats": active_cats,
    },
)

# =============================================================================
# SETUP
# =============================================================================

agents = {i: RebalancingAgent(i) for i in range(5)}

G = generate_network(node_list)
all_days_demand_vectors, transformed_demand_vectors = generate_global_demand(
    node_list, NUM_TRAIN_DAYS, demand_params, TIME_SLOTS
)

num_stations = np.sum(node_list)
boundaries = np.cumsum([0] + node_list)
daily_returns = []
daily_failures = []
np.random.seed(args.seed)
random.seed(args.seed)

env = FairEnv(G, transformed_demand_vectors, beta, GAMMA, station_params)
state = env.reset()

# =============================================================================
# TRAINING LOOP
# =============================================================================

global_step = 0
start = time.time()
for repeat in range(110):
    for day in range(NUM_TRAIN_DAYS):
        ret = 0
        fails = 0
        cat_daily_fails = {cat: 0.0 for cat in active_cats}
        for _times in (0, 1):
            actions = np.zeros(num_stations, dtype=np.int64)
            if not (repeat == 0 and day == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]["station"]
                    actions[i] = agents[cat].decide_action(state[i])

            next_state, reward, failures = env.step(actions)
            ret += np.sum(reward)
            fails += np.sum(failures)

            for cat_idx, cat in enumerate(active_cats):
                cat_daily_fails[cat] += np.sum(
                    failures[boundaries[cat_idx] : boundaries[cat_idx + 1]]
                )

            if not (day == 0 and repeat == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]["station"]
                    if repeat < TRAIN_UNTIL[cat]:
                        agents[cat].update_q_table(
                            state[i], actions[i], reward[i], next_state[i]
                        )
                        agents[cat].update_epsilon()

            state = next_state

        if not (repeat == 0 and day == 0):
            global_step += 1
            daily_returns.append(ret)
            daily_failures.append(fails)

            log_dict = {
                "repeat": repeat,
                "day": day,
                "global_step": global_step,
                "elapsed_time": time.time() - start,
                "daily_return": ret,
                "daily_failures": fails,
            }
            for cat in active_cats:
                log_dict[f"failures/cat{cat}"] = cat_daily_fails[cat]
            for cat in active_cats:
                log_dict[f"epsilon/cat{cat}"] = agents[cat].epsilon
            wandb.log(log_dict)

wandb.finish()
# =============================================================================
# SAVE RESULTS
# =============================================================================

q_tables_dir = os.path.join(output_dir, "q_tables")
results_dir = os.path.join(output_dir, "results")
os.makedirs(q_tables_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

for cat in active_cats:
    with open(
        os.path.join(
            q_tables_dir, f"q_table_{beta}_{args.categories}_{args.seed}_cat{cat}.pkl"
        ),
        "wb",
    ) as file:
        pickle.dump(agents[cat].q_table, file)

np.save(
    os.path.join(
        results_dir, f"learning_curve_{args.categories}_cat_{beta}_{args.seed}.npy"
    ),
    daily_returns,
)

print(
    f"Finished simulation with seed: {args.seed}, categories: {args.categories} and beta: {beta}"
)

total_bikes = sum(G.nodes[i]["bikes"] for i in range(num_stations))
np.save(
    os.path.join(results_dir, f"bikes_{args.categories}_cat_{beta}_{args.seed}.npy"),
    total_bikes,
)
