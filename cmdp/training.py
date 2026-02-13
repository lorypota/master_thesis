import argparse
import os
import pickle
import random
import time

import numpy as np
import psutil
import wandb

from cmdp.environment import CMDPEnv
from common.agent import RebalancingAgent
from common.config import (
    CPU_CORES,
    GAMMA,
    MAX_MEMORY_MB,
    NUM_TRAIN_DAYS,
    TIME_SLOTS,
    TRAIN_UNTIL,
    get_scenario,
)
from common.demand import generate_global_demand
from common.network import generate_network

# Limit resource usage app-reken12
os.system(f"procgov64 --nowait --minws 10M --maxws {MAX_MEMORY_MB}M -p {os.getpid()}")
p = psutil.Process()
p.cpu_affinity(
    list(range(int(CPU_CORES.split("-")[0]), int(CPU_CORES.split("-")[1]) + 1))
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--r-max", default=0.15, type=float)
parser.add_argument("--categories", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument(
    "--output-dir", default=SCRIPT_DIR, type=str, help="Output directory for results"
)
parser.add_argument(
    "--constrained-cats",
    nargs="+",
    type=int,
    default=[0],
    help="Category indices to constrain (default: [0] = remote only)",
)
parser.add_argument("--eta", default=0.1, type=float, help="Dual step size")
parser.add_argument(
    "--n-dual", default=100, type=int, help="Days between dual variable updates"
)
parser.add_argument(
    "--run-group", default=None, type=str, help="Wandb group ID for grouping runs"
)
args = parser.parse_args()

r_max = args.r_max
eta = args.eta
n_dual = args.n_dual
constrained_cats = set(args.constrained_cats)
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
# DUAL VARIABLE SETUP
# =============================================================================

# Boundaries for slicing failures by category
boundaries = np.cumsum([0] + node_list)

# Initialize dual variables (lambdas) for constrained categories only
lambdas = {}
failure_thresholds = {}
for cat_idx, cat in enumerate(active_cats):
    if cat in constrained_cats:
        lambdas[cat] = [0.0, 0.0]  # [morning, evening]
        # Threshold = r_max * 12 * lambda_d (per area per 12-hour period)
        failure_thresholds[cat] = [
            r_max * 12 * demand_params[cat_idx][0][1],  # morning
            r_max * 12 * demand_params[cat_idx][1][1],  # evening
        ]

# Failure accumulator and tracking
failure_accumulator = {cat: [0.0, 0.0] for cat in lambdas}
day_counter = 0
lambda_history = []

# =============================================================================
# WANDB
# =============================================================================

wandb.init(
    project="fairmss",
    group=f"cmdp-{args.categories}cat-{args.run_group}",
    name=f"rmax{r_max}_seed{args.seed}",
    config={
        "method": "cmdp",
        "r_max": r_max,
        "categories": args.categories,
        "seed": args.seed,
        "eta": eta,
        "n_dual": n_dual,
        "constrained_cats": list(constrained_cats),
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
daily_returns = []
daily_failures = []
np.random.seed(args.seed)
random.seed(args.seed)

env = CMDPEnv(G, transformed_demand_vectors, lambdas, GAMMA, station_params)
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
        cat_period_fails = {cat: {} for cat in active_cats}
        for _times in (0, 1):
            actions = np.zeros(num_stations, dtype=np.int64)
            if not (repeat == 0 and day == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]["station"]
                    actions[i] = agents[cat].decide_action(state[i])

            next_state, reward, failures = env.step(actions)
            ret += np.sum(reward)
            fails += np.sum(failures)

            # Accumulate per-category per-period failures for dual update
            period = env.current_period
            for cat_idx, cat in enumerate(active_cats):
                cat_failures = np.sum(
                    failures[boundaries[cat_idx] : boundaries[cat_idx + 1]]
                )
                cat_daily_fails[cat] += cat_failures
                cat_period_fails[cat][period] = cat_failures / node_list[cat_idx]
                if cat in failure_accumulator:
                    # Normalize by number of areas in this category
                    failure_accumulator[cat][period] += (
                        cat_failures / node_list[cat_idx]
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

        # Dual variable update every n_dual days
        dual_update_info = {}
        day_counter += 1
        if day_counter >= n_dual:
            for cat in list(failure_accumulator.keys()):
                # Only update if this category is still training
                if repeat < TRAIN_UNTIL[cat]:
                    for p in (0, 1):
                        f_hat = failure_accumulator[cat][p] / n_dual
                        f_bar = failure_thresholds[cat][p]
                        violation = f_hat - f_bar
                        pname = "morning" if p == 0 else "evening"
                        dual_update_info[f"dual/cat{cat}_{pname}_f_hat"] = f_hat
                        dual_update_info[f"dual/cat{cat}_{pname}_f_bar"] = f_bar
                        dual_update_info[f"dual/cat{cat}_{pname}_violation"] = violation
                        lambdas[cat][p] = max(0.0, lambdas[cat][p] + eta * violation)

            # Log snapshot
            lambda_history.append(
                (repeat, day, {c: list(v) for c, v in lambdas.items()})
            )

            # Reset accumulator
            failure_accumulator = {cat: [0.0, 0.0] for cat in lambdas}
            day_counter = 0

        if not (repeat == 0 and day == 0):
            global_step += 1
            daily_returns.append(ret)
            daily_failures.append(fails)

            # wandb logging
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
            for cat_idx, cat in enumerate(active_cats):
                for p in (0, 1):
                    pname = "morning" if p == 0 else "evening"
                    lambda_d = demand_params[cat_idx][p][1]
                    per_area = cat_period_fails[cat].get(p, 0.0)
                    rate = per_area / (12 * lambda_d) if lambda_d > 0 else 0.0
                    log_dict[f"failure_rate/cat{cat}_{pname}"] = rate
            for cat, lam in lambdas.items():
                log_dict[f"lambda/cat{cat}_morning"] = lam[0]
                log_dict[f"lambda/cat{cat}_evening"] = lam[1]
            for cat in active_cats:
                log_dict[f"epsilon/cat{cat}"] = agents[cat].epsilon
            log_dict.update(dual_update_info)
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
            q_tables_dir, f"q_table_{r_max}_{args.categories}_{args.seed}_cat{cat}.pkl"
        ),
        "wb",
    ) as file:
        pickle.dump(agents[cat].q_table, file)

np.save(
    os.path.join(
        results_dir, f"learning_curve_{args.categories}_cat_{r_max}_{args.seed}.npy"
    ),
    daily_returns,
)

total_bikes = sum(G.nodes[i]["bikes"] for i in range(num_stations))
np.save(
    os.path.join(results_dir, f"bikes_{args.categories}_cat_{r_max}_{args.seed}.npy"),
    total_bikes,
)

# Save lambda history and final values
with open(
    os.path.join(
        results_dir, f"lambda_history_{args.categories}_cat_{r_max}_{args.seed}.pkl"
    ),
    "wb",
) as file:
    pickle.dump(lambda_history, file)

with open(
    os.path.join(
        results_dir, f"final_lambdas_{args.categories}_cat_{r_max}_{args.seed}.pkl"
    ),
    "wb",
) as file:
    pickle.dump(dict(lambdas), file)

print(
    f"Finished simulation with seed: {args.seed}, categories: {args.categories} and r_max: {r_max}"
)
print(f"Final lambdas: {dict(lambdas)}")
