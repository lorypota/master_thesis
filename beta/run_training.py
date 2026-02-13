import os
from datetime import datetime

from common.config import CPU_CORES, MAX_MEMORY_MB

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


seeds = range(100, 110)
categories = range(2, 6)
betas = [round(b * 0.1, 1) for b in range(11)]

run_group = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Starting training reproduction...")

training_script = os.path.join(SCRIPT_DIR, "training.py")

for s in seeds:
    for c in categories:
        for b in betas:
            cmd = (
                f"procgov64 --nowait --minws 10M --maxws {MAX_MEMORY_MB}M --cpu {CPU_CORES}"
                f" -- uv run {training_script} --beta {b} --categories {c} --seed {s} --run-group {run_group}"
            )

            print(f"Running: {cmd}")

            # Run the command
            exit_code = os.system(cmd)

            # Check for errors
            if exit_code != 0:
                print(f"!!! Error encountered running: {cmd}")
                # Uncomment next line to stop on error
                # sys.exit(1)
