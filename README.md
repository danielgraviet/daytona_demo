# Harbor + Daytona · CartPole-v1 Distributed RL Demo

A demo that spins up N isolated Daytona sandboxes in parallel, runs a reinforcement learning training job inside each one, and streams live results into a Rich terminal dashboard.

Each sandbox trains a PPO agent (via Stable-Baselines3) on the classic `CartPole-v1` environment with a randomly assigned learning rate. The orchestrator collects JSON progress lines from each sandbox and renders a live table as training progresses.

## How it works

1. `run_demo.py` creates N Daytona sandboxes concurrently using a thread pool.
2. It uploads `cartpole_task.py` to each sandbox and executes it.
3. Each sandbox trains a PPO model for 50,000 timesteps, then evaluates it over the requested number of episodes, printing JSON progress every 50 episodes.
4. The orchestrator parses those JSON lines and updates the live Rich dashboard.
5. Sandboxes are automatically cleaned up after the run (auto-stop: 10 min, auto-delete: 30 min).

## Requirements

- Python 3.13+
- A [Daytona](https://www.daytona.io/) account and API key
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd daytona_demo

# Install dependencies (with uv)
uv sync

# Or with pip
pip install daytona-sdk rich gymnasium
```

Set your Daytona API key:

```bash
export DAYTONA_API_KEY=your_key_here
```

## Usage

```bash
# Run with default settings (25 sandboxes, 300 episodes each)
python run_demo.py

# Custom sandbox count and episode count
python run_demo.py --sandboxes 25 --episodes 300
python run_demo.py --sandboxes 100 --episodes 300

# Large-scale demo
python run_demo.py --sandboxes 25000 --episodes 300

# Limit parallel threads (default: min(sandboxes, 200))
python run_demo.py --sandboxes 50 --workers 20
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--sandboxes` | `25` | Number of Daytona sandboxes to spin up |
| `--episodes` | `300` | Evaluation episodes per sandbox after training |
| `--workers` | `min(sandboxes, 200)` | Maximum parallel threads |

## Project structure

```
daytona_demo/
├── run_demo.py        # Orchestrator: creates sandboxes and drives the dashboard
├── cartpole_task.py   # RL training script uploaded to each sandbox
├── main.py            # Minimal entrypoint placeholder
└── pyproject.toml     # Project metadata and dependencies
```

## Dashboard

The live terminal dashboard shows (up to 20 sandboxes at once):

- Elapsed time, running count, completion count
- Per-sandbox status (`pending` / `running` / `done` / `error`)
- Current episode, rolling 100-episode average score, best score
- Whether CartPole was "solved" (avg ≥ 195) and at which episode
- Learning rate used for that sandbox

After all sandboxes finish, a summary is printed and any errors are written to `errors.log`.

## CartPole solve criteria

The environment is considered **solved** when the rolling 100-episode average reward reaches **195.0**, matching the original OpenAI Gym benchmark.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DAYTONA_API_KEY` | Yes | Your Daytona API key |
