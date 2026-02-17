# A Fairness-Oriented Reinforcement Learning Approach for the Operation and Control of Shared Micromobility Services

## Project Structure

```
FairMSS/
├── common/                        # Shared modules
│   ├── config.py                  # Scenario definitions, constants, helpers
│   ├── agent.py                   # Q-learning RebalancingAgent
│   ├── demand.py                  # Demand generation (Skellam distribution)
│   ├── network.py                 # Network graph generation
│   └── av_actions.py              # Available actions
│
├── beta/                          # Beta-weighted MDP formulation
│   ├── environment.py             # FairEnv with beta fairness parameter
│   ├── training.py                # Training script
│   ├── evaluation.py              # Evaluation across beta values
│   ├── run_training.py            # Batch training runner
│   ├── boxplots.py                # Boxplot visualizations
│   ├── paretoplots.py             # Pareto front plots
│   ├── learning_curves.py         # Learning curve plots
│   ├── q_tables/                  # Trained Q-tables
│   ├── results/                   # Evaluation outputs (.npy)
│   └── plots/                     # Generated figures
│
├── cmdp/                          # Lagrangian CMDP formulation
│   ├── environment.py             # CMDPEnv with adaptive dual variables
│   ├── training.py                # Training with Lagrangian dual updates
│   ├── evaluation.py              # Evaluation with constraint checking
│   ├── run_training.py            # Batch training runner
│   ├── q_tables/                  # Trained Q-tables
│   └── results/                   # Evaluation outputs (.npy, .pkl)
│
├── preliminary_studies/           # Earlier experiments and prototypes
│   ├── baseline/
│   └── failure_rate_analysis/
│
├── pyproject.toml
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies with: `uv sync`

## Beta formulation

### Training

```bash
# Train a single configuration
uv run beta/training.py --beta 5 --categories 3 --seed 100

# Train all configurations
uv run beta/run_training.py
```

### Evaluation

```bash
# Evaluate a scenario (uses pre-trained Q-tables from beta/q_tables/)
uv run beta/evaluation.py --categories 3

# With detailed cost breakdowns
uv run beta/evaluation.py --categories 5 --seeds 100 110 --save-detailed
```

### Plotting

```bash
uv run beta/boxplots.py --cat 5 --save
uv run beta/paretoplots.py --cat 5 --save
uv run beta/learning_curves.py --cat 5 --save
```

## CMDP formulation

Replaces the fixed beta fairness weight with adaptive Lagrange multipliers that enforce explicit failure-rate constraints.

### Training

```bash
# Train a single configuration (r_max = max allowed failure rate)
uv run python cmdp/training.py --r-max 0.15 --categories 2 --seed 100

# Train all configurations
uv run python cmdp/run_training.py
```

### Evaluation

```bash
# Evaluate a scenario
uv run python cmdp/evaluation.py --categories 2

# With custom r_max values
uv run python cmdp/evaluation.py --categories 2 --r-max-values 0.05 0.10 0.15 0.20 0.25
```
