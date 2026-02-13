import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = range(100, 110)
categories = [5]
r_max_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

run_group = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Starting CMDP training...")

training_script = os.path.join(SCRIPT_DIR, "training.py")

for s in seeds:
    for c in categories:
        for r in r_max_values:
            cmd = f"uv run {training_script} --r-max {r} --categories {c} --seed {s} --run-group {run_group}"

            print(f"Running: {cmd}")

            # Run the command
            exit_code = os.system(cmd)

            # Check for errors
            if exit_code != 0:
                print(f"!!! Error encountered running: {cmd}")
                sys.exit(1)
