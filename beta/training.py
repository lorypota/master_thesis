"""
Beta training script.

Usage:
    uv run beta/training.py --beta 0.5 --categories 5 --seed 100
"""

import argparse
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fmt_token(value):
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s

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
parser.add_argument("--cpu-cores", default=CPU_CORES, type=str, help="CPU core range")
args = parser.parse_args()

# Limit resource usage for app-reken12
# os.system(f"procgov64 --nowait --minws 10M --maxws {MAX_MEMORY_MB}M -p {os.getpid()}")
p = psutil.Process()
p.cpu_affinity(
    list(
        range(int(args.cpu_cores.split("-")[0]), int(args.cpu_cores.split("-")[1]) + 1)
    )
)

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
boundaries = scenario["boundaries"]
b_token = f"b{fmt_token(beta)}"
cat_dirname = f"cat{args.categories}"
seed_dirname = f"seed{args.seed}"
q_tables_dir = os.path.join(output_dir, "q_tables", cat_dirname, seed_dirname)
results_dir = os.path.join(output_dir, "results", cat_dirname, seed_dirname)
os.makedirs(q_tables_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

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
np.random.seed(args.seed)
random.seed(args.seed)
all_days_demand_vectors, transformed_demand_vectors = generate_global_demand(
    node_list, NUM_TRAIN_DAYS, demand_params, TIME_SLOTS
)

num_stations = np.sum(node_list)
daily_returns = []
daily_reb_costs = []
daily_failures = []
daily_total_bikes = []
daily_nonzero_actions = []
daily_mean_abs_action = []
daily_cat_failures = []
daily_cat_period_failures = []
daily_cat_bikes = []
daily_cat_nonzero_actions = []
daily_cat_mean_abs_action = []

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
        reb_ret = 0
        fails = 0
        cat_daily_fails = {cat: 0.0 for cat in active_cats}
        cat_period_fails = {cat: {} for cat in active_cats}
        cat_daily_nonzero_actions = {cat: 0 for cat in active_cats}
        cat_daily_abs_actions = {cat: 0.0 for cat in active_cats}
        nonzero_actions_day = 0
        abs_actions_sum_day = 0.0
        for _times in (0, 1):
            actions = np.zeros(num_stations, dtype=np.int64)
            if not (repeat == 0 and day == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]["station"]
                    actions[i] = agents[cat].decide_action(state[i])

            nonzero_actions_day += int(np.count_nonzero(actions))
            abs_actions_sum_day += float(np.sum(np.abs(actions)))
            for cat_idx, cat in enumerate(active_cats):
                start_idx = boundaries[cat_idx]
                end_idx = boundaries[cat_idx + 1]
                cat_actions = actions[start_idx:end_idx]
                cat_daily_nonzero_actions[cat] += int(np.count_nonzero(cat_actions))
                cat_daily_abs_actions[cat] += float(np.sum(np.abs(cat_actions)))

            next_state, reward, failures, reb_costs = env.step(actions)
            ret += np.sum(reward)
            reb_ret += np.sum(reb_costs)
            fails += np.sum(failures)

            for cat_idx, cat in enumerate(active_cats):
                cat_failures = np.sum(
                    failures[boundaries[cat_idx] : boundaries[cat_idx + 1]]
                )
                cat_daily_fails[cat] += cat_failures
                period = 0 if env.next_rebalancing_hour == 23 else 1
                cat_period_fails[cat][period] = cat_failures / node_list[cat_idx]

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
            daily_reb_costs.append(reb_ret)
            daily_failures.append(fails)
            daily_nonzero_actions.append(nonzero_actions_day)
            daily_mean_abs_action.append(abs_actions_sum_day / (2 * num_stations))
            daily_cat_failures.append([cat_daily_fails[cat] for cat in active_cats])
            daily_cat_period_failures.append(
                [
                    [cat_period_fails[cat].get(0, 0.0), cat_period_fails[cat].get(1, 0.0)]
                    for cat in active_cats
                ]
            )
            daily_cat_nonzero_actions.append(
                [cat_daily_nonzero_actions[cat] for cat in active_cats]
            )
            daily_cat_mean_abs_action.append(
                [
                    cat_daily_abs_actions[cat] / (2 * node_list[cat_idx])
                    for cat_idx, cat in enumerate(active_cats)
                ]
            )
            cat_bikes_end = []
            for cat_idx, _cat in enumerate(active_cats):
                start_idx = boundaries[cat_idx]
                end_idx = boundaries[cat_idx + 1]
                cat_bikes_end.append(
                    int(sum(G.nodes[i]["bikes"] for i in range(start_idx, end_idx)))
                )
            daily_cat_bikes.append(cat_bikes_end)
            daily_total_bikes.append(int(sum(cat_bikes_end)))

            log_dict = {
                "repeat": repeat,
                "day": day,
                "global_step": global_step,
                "elapsed_time": time.time() - start,
                "daily_return": ret,
                "daily_failures": fails,
                "daily_reb_costs": reb_ret,
                "daily_total_bikes": daily_total_bikes[-1],
                "daily_nonzero_actions": nonzero_actions_day,
                "daily_mean_abs_action": daily_mean_abs_action[-1],
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

for cat in active_cats:
    with open(
        os.path.join(q_tables_dir, f"q_table_{b_token}_cat{cat}.pkl"),
        "wb",
    ) as file:
        pickle.dump(agents[cat].q_table, file)

np.save(
    os.path.join(results_dir, f"learning_curve_{b_token}.npy"),
    daily_returns,
)

np.save(
    os.path.join(results_dir, f"reb_costs_{b_token}.npy"),
    daily_reb_costs,
)

print(
    f"Finished simulation with seed: {args.seed}, categories: {args.categories} and beta: {beta}"
)

total_bikes = sum(G.nodes[i]["bikes"] for i in range(num_stations))
np.save(
    os.path.join(results_dir, f"bikes_{b_token}.npy"),
    total_bikes,
)

np.savez_compressed(
    os.path.join(results_dir, f"train_diag_{b_token}.npz"),
    daily_return=np.asarray(daily_returns),
    daily_reb_costs=np.asarray(daily_reb_costs),
    daily_failures=np.asarray(daily_failures),
    daily_total_bikes=np.asarray(daily_total_bikes),
    daily_nonzero_actions=np.asarray(daily_nonzero_actions),
    daily_mean_abs_action=np.asarray(daily_mean_abs_action),
    daily_cat_failures=np.asarray(daily_cat_failures),
    daily_cat_period_failures=np.asarray(daily_cat_period_failures),
    daily_cat_bikes=np.asarray(daily_cat_bikes),
    daily_cat_nonzero_actions=np.asarray(daily_cat_nonzero_actions),
    daily_cat_mean_abs_action=np.asarray(daily_cat_mean_abs_action),
)

with open(os.path.join(results_dir, f"meta_{b_token}.json"), "w") as file:
    json.dump(
        {
            "args": vars(args),
            "tokens": {"b": b_token},
            "scenario": {
                "categories": args.categories,
                "node_list": node_list,
                "active_cats": active_cats,
                "boundaries": boundaries.tolist(),
            },
            "train_until": {str(cat): v for cat, v in TRAIN_UNTIL.items()},
        },
        file,
        indent=2,
    )
