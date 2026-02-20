"""Generate all CMDP plots for a given category scenario."""

import argparse
import subprocess
import sys

scripts = [
    "cmdp/plots/boxplots.py",
    "cmdp/plots/paretoplots.py",
    "cmdp/plots/lambda_convergence.py",
    "cmdp/plots/failure_rates_by_rmax.py",
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
