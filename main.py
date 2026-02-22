"""
main.py
-------
Orchestration layer for the Harbor + Daytona CartPole demo.

Spins up N sandboxes in parallel, uploads cartpole_task.py to each,
runs it, and streams results into a live Rich terminal dashboard.

Usage:
    uv run python main.py --sandboxes 10 --episodes 300
    python main.py --sandboxes 100 --episodes 300
    python main.py --sandboxes 25000 --episodes 300  # the big demo
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.live import Live
from daytona_sdk import Daytona, DaytonaConfig

from models import DemoState
from dashboard import build_dashboard
from sandbox import run_sandbox

API_KEY     = os.environ.get("DAYTONA_API_KEY", "")
TASK_SCRIPT = Path(__file__).parent / "cartpole_task.py"

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Harbor + Daytona CartPole Demo")
    parser.add_argument("--sandboxes", type=int, default=25,
                        help="Number of sandboxes to spin up (default: 25)")
    parser.add_argument("--episodes",  type=int, default=300,
                        help="Training episodes per sandbox (default: 300)")
    parser.add_argument("--workers",   type=int, default=None,
                        help="Max parallel threads (default: min(sandboxes, 200))")
    args = parser.parse_args()

    if not API_KEY:
        console.print("[bold red]Error:[/] Set DAYTONA_API_KEY environment variable.")
        console.print("  export DAYTONA_API_KEY=your_key_here")
        sys.exit(1)

    if not TASK_SCRIPT.exists():
        console.print(f"[bold red]Error:[/] cartpole_task.py not found at {TASK_SCRIPT}")
        sys.exit(1)

    task_code = TASK_SCRIPT.read_text()
    n         = args.sandboxes
    episodes  = args.episodes
    workers   = args.workers or min(n, 200)

    # Vary learning rates across sandboxes for interesting diversity
    lr_choices = [0.001, 0.005, 0.01, 0.02, 0.05]
    lrs = [random.choice(lr_choices) for _ in range(n)]

    daytona_client = Daytona(DaytonaConfig(api_key=API_KEY))
    state = DemoState(total=n)

    console.print("\n[bold cyan]Harbor + Daytona  ·  CartPole-v1 Distributed RL[/]")
    console.print(f"Spinning up [bold]{n}[/] sandboxes with [bold]{episodes}[/] episodes each...\n")

    with Live(build_dashboard(state, episodes), refresh_per_second=4, console=console) as live:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_sandbox, i, state, daytona_client, episodes, lrs[i], task_code): i
                for i in range(n)
            }
            while any(f.running() for f in futures) or not all(f.done() for f in futures):
                live.update(build_dashboard(state, episodes))
                time.sleep(0.25)
            # Final update after all done
            live.update(build_dashboard(state, episodes))

    # ── Final summary ─────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Final Results[/]")
    console.print(f"  Total sandboxes : [bold]{n}[/]")
    console.print(f"  Completed       : [bold green]{state.n_complete}[/]")
    console.print(f"  Solved CartPole : [bold yellow]{state.n_solved}[/] ({round(state.n_solved/max(state.n_complete,1)*100,1)}%)")
    console.print(f"  Avg final score : [bold magenta]{state.avg_final}[/]")
    console.print(f"  Wall time       : [bold cyan]{state.elapsed}s[/]")
    console.print()

    # ── Write errors to log file ───────────────────────────────────────────
    errors = [r for r in state.results.values() if r.status == "error"]
    if errors:
        log_path = Path(__file__).parent / "errors.log"
        with log_path.open("w") as f:
            for r in sorted(errors, key=lambda r: r.index):
                f.write(f"[sandbox {r.index + 1}] id={r.sandbox_id}\n{r.error}\n\n")
        console.print(f"[bold red]{len(errors)} errors[/] written to [cyan]{log_path}[/]\n")


if __name__ == "__main__":
    main()
