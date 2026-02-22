# Harbor + Daytona · CartPole-v1 Distributed RL Demo

A demo that spins up N isolated Daytona sandboxes in parallel, runs a CartPole-v1 evaluation job inside each one, and streams live results into a Rich terminal dashboard.

Each sandbox runs an analytical controller against the `CartPole-v1` environment. The orchestrator collects JSON progress lines from each sandbox and renders a live table as evaluation progresses.

## How it works

1. `main.py` creates N Daytona sandboxes concurrently using a thread pool.
2. It uploads `cartpole_task.py` to each sandbox and executes it.
3. Each sandbox runs the analytical controller for the requested number of episodes, printing JSON progress every 50 episodes.
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
python main.py

# Or with uv
uv run python main.py --sandboxes 10 --episodes 300

# Custom sandbox count and episode count
python main.py --sandboxes 25 --episodes 300
python main.py --sandboxes 100 --episodes 300

# Large-scale demo
python main.py --sandboxes 25000 --episodes 300

# Limit parallel threads (default: min(sandboxes, 200))
python main.py --sandboxes 50 --workers 20
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--sandboxes` | `25` | Number of Daytona sandboxes to spin up |
| `--episodes` | `300` | Evaluation episodes per sandbox |
| `--workers` | `min(sandboxes, 200)` | Maximum parallel threads |

## Project structure

```
daytona_demo/
├── main.py                  # Entrypoint: parses args, creates sandboxes, drives the dashboard
├── cartpole_task.py         # Controller script uploaded and run inside each sandbox
├── models/
│   ├── sandbox_result.py    # SandboxResult dataclass
│   └── demo_state.py        # DemoState class (thread-safe result aggregation)
├── dashboard/
│   └── builder.py           # build_dashboard() — renders the live Rich panel
├── sandbox/
│   └── runner.py            # run_sandbox() — per-sandbox worker function
└── pyproject.toml           # Project metadata and dependencies
```

## Dashboard

The live terminal dashboard shows (up to 20 sandboxes at once):

- Elapsed time, running count, completion count
- Per-sandbox status (`pending` / `running` / `done` / `error`)
- Current episode, rolling 100-episode average score, best score
- Whether CartPole was "solved" (avg ≥ 195) and at which episode
- Learning rate column (passed for CLI compatibility)

After all sandboxes finish, a summary is printed and any errors are written to `errors.log`.

## CartPole controller

`cartpole_task.py` uses a simple analytical controller: it pushes the cart in the direction the pole is falling based on the pole angle and angular velocity (`obs[2] + obs[3]`). No training or ML libraries are required — only `gymnasium`.

## CartPole solve criteria

The environment is considered **solved** when the rolling 100-episode average reward reaches **195.0**, matching the original OpenAI Gym benchmark.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DAYTONA_API_KEY` | Yes | Your Daytona API key |
