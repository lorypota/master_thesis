"""Generate all beta plots for a given category scenario."""

import argparse
import subprocess
import sys

scripts = [
    "beta/plots/boxplots.py",
    "beta/plots/paretoplots.py",
    "beta/plots/learning_curves.py",
    "beta/plots/failure_rates_by_beta.py",
]

parser = argparse.ArgumentParser()
parser.add_argument("--categories", required=True, type=int)
args = parser.parse_args()

for script in scripts:
    print(f"\n{'='*60}\nRunning {script}...\n{'='*60}")
    env = {"MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, script, "--categories", str(args.categories), "--save"],
        env={**__import__("os").environ, **env},
    )
    if result.returncode != 0:
        print(f"FAILED: {script}")

print(f"\n{'='*60}\nAll done.\n{'='*60}")
