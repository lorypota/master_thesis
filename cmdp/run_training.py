import os
import sys
from datetime import datetime

from common.config import R_MAX_VALUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = range(100, 110)
categories = [5]
r_max_values = R_MAX_VALUES

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
