import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


seeds = range(100, 110)
categories = [5]
betas = [round(b * 0.1, 1) for b in range(11)]

run_group = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Starting training reproduction...")

training_script = os.path.join(SCRIPT_DIR, "training.py")

for s in seeds:
    for c in categories:
        for b in betas:
            cmd = f"uv run {training_script} --beta {b} --categories {c} --seed {s} --run-group {run_group}"

            print(f"Running: {cmd}")

            # Run the command
            exit_code = os.system(cmd)

            # Check for errors
            if exit_code != 0:
                print(f"!!! Error encountered running: {cmd}")
                sys.exit(1)
