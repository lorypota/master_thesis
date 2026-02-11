import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = range(100, 110)
categories = range(2, 6)
betas = range(0, 11)

seeds = [100]
categories = [2]
betas = [0]

print("Starting training reproduction...")

training_script = os.path.join(SCRIPT_DIR, "training.py")

for s in seeds:
    for c in categories:
        for b in betas:
            cmd = f"python {training_script} --beta {b} --categories {c} --seed {s}"

            print(f"Running: {cmd}")

            # Run the command
            exit_code = os.system(cmd)

            # Check for errors
            if exit_code != 0:
                print(f"!!! Error encountered running: {cmd}")
                # Uncomment next line to stop on error
                # sys.exit(1)