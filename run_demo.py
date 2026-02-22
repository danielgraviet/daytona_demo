"""
run_demo.py
-----------
Demo orchestrator for the Harbor + Daytona presentation.

Spins up N sandboxes in parallel, uploads cartpole_task.py to each,
runs it, and streams results into a live Rich terminal dashboard.

Usage:
    python run_demo.py --sandboxes 25 --episodes 300
    python run_demo.py --sandboxes 100 --episodes 300
    python run_demo.py --sandboxes 25000 --episodes 300  # the big demo

Requirements:
    pip install daytona rich
"""

import argparse
import json
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

# ── Rich for the live dashboard ─────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
except ImportError:
    print("Installing rich...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "--quiet"], check=True)
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich import box

# ── Daytona SDK ──────────────────────────────────────────────────────────────
from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxBaseParams as CreateSandboxParams

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("DAYTONA_API_KEY", "")
TASK_SCRIPT  = Path(__file__).parent / "cartpole_task.py"

console = Console()

# ── State tracking ───────────────────────────────────────────────────────────
@dataclass
class SandboxResult:
    sandbox_id:  str
    index:       int
    status:      str = "pending"   # pending | running | complete | error
    episode:     int = 0
    avg_100:     float = 0.0
    best:        float = 0.0
    solved:      bool = False
    solved_at:   Optional[int] = None
    elapsed_s:   float = 0.0
    lr:          float = 0.01
    error:       str = ""


class DemoState:
    def __init__(self, total: int):
        self.total     = total
        self.results   = {}
        self.lock      = Lock()
        self.start_time = time.time()

    def update(self, index: int, sandbox_id: str, data: dict):
        with self.lock:
            if index not in self.results:
                self.results[index] = SandboxResult(sandbox_id=sandbox_id, index=index)
            r = self.results[index]
            r.sandbox_id = sandbox_id
            if data.get("status") == "complete":
                r.status    = "complete"
                r.avg_100   = data.get("final_avg", 0)
                r.best      = data.get("best", 0)
                r.solved    = data.get("solved", False)
                r.solved_at = data.get("solved_at")
                r.elapsed_s = data.get("elapsed_s", 0)
                r.lr        = data.get("lr", 0.01)
            else:
                r.status  = "running"
                r.episode = data.get("episode", 0)
                r.avg_100 = data.get("avg_100", 0)
                r.solved  = data.get("solved", False)

    def mark_error(self, index: int, sandbox_id: str, msg: str):
        with self.lock:
            if index not in self.results:
                self.results[index] = SandboxResult(sandbox_id=sandbox_id, index=index)
            self.results[index].status = "error"
            self.results[index].error  = msg

    def mark_running(self, index: int, sandbox_id: str):
        with self.lock:
            if index not in self.results:
                self.results[index] = SandboxResult(sandbox_id=sandbox_id, index=index)
            self.results[index].status = "running"

    # ── Computed stats ────────────────────────────────────────────────────
    @property
    def n_complete(self):
        return sum(1 for r in self.results.values() if r.status == "complete")

    @property
    def n_running(self):
        return sum(1 for r in self.results.values() if r.status == "running")

    @property
    def n_solved(self):
        return sum(1 for r in self.results.values() if r.solved)

    @property
    def avg_final(self):
        vals = [r.avg_100 for r in self.results.values() if r.status == "complete"]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    @property
    def elapsed(self):
        return round(time.time() - self.start_time, 1)


# ── Dashboard rendering ───────────────────────────────────────────────────────
def build_dashboard(state: DemoState, episodes: int) -> Panel:
    elapsed = state.elapsed
    n_done  = state.n_complete
    n_run   = state.n_running
    n_pend  = state.total - n_done - n_run - sum(
        1 for r in state.results.values() if r.status == "error"
    )

    # ── Header stats ─────────────────────────────────────────────────────
    solved_pct = round(state.n_solved / max(n_done, 1) * 100, 1)
    header = Text()
    header.append("⏱  ", style="dim")
    header.append(f"{elapsed}s elapsed   ", style="bold cyan")
    header.append("🟢 ", style="dim")
    header.append(f"{n_run} running   ", style="bold green")
    header.append("✅ ", style="dim")
    header.append(f"{n_done}/{state.total} complete   ", style="bold white")
    header.append("🎯 ", style="dim")
    header.append(f"{state.n_solved} solved ({solved_pct}%)   ", style="bold yellow")
    header.append("📈 ", style="dim")
    header.append(f"avg score {state.avg_final}", style="bold magenta")

    # ── Results table (show up to 20 rows) ────────────────────────────────
    with state.lock:
        sample = sorted(state.results.values(), key=lambda r: r.index)[:20]

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#",         style="dim",          width=5)
    table.add_column("Sandbox",   style="dim cyan",     width=14)
    table.add_column("Status",    width=10)
    table.add_column("Episode",   justify="right",      width=9)
    table.add_column("Avg(100)",  justify="right",      width=10)
    table.add_column("Best",      justify="right",      width=8)
    table.add_column("Solved",    justify="center",     width=8)
    table.add_column("LR",        justify="right",      width=8)

    status_styles = {
        "pending":  ("dim", "…"),
        "running":  ("green", "⚙ running"),
        "complete": ("bold white", "✓ done"),
        "error":    ("red", "✗ error"),
    }

    for r in sample:
        style, label = status_styles.get(r.status, ("dim", r.status))
        solved_icon   = "[bold green]✓[/]" if r.solved else "[dim]–[/]"
        solved_ep     = f" ep{r.solved_at}" if r.solved_at else ""
        table.add_row(
            str(r.index + 1),
            r.sandbox_id[:12] + "…" if len(r.sandbox_id) > 12 else r.sandbox_id,
            f"[{style}]{label}[/]",
            str(r.episode) if r.episode else "–",
            str(r.avg_100) if r.avg_100 else "–",
            str(r.best)    if r.best    else "–",
            f"{solved_icon}{solved_ep}",
            str(r.lr)      if r.lr      else "–",
        )

    if state.total > 20:
        table.add_row(
            "…", f"(+{state.total - 20} more)", "", "", "", "", "", "",
            style="dim"
        )

    layout = Layout(name="root")
    layout.split_column(
        Layout(header, size=1),
        Layout(table),
    )
    return Panel(
        layout,
        title="[bold cyan]🚀 Harbor + Daytona  ·  CartPole-v1 Distributed RL Demo[/]",
        subtitle=f"[dim]{state.total} sandboxes  ·  {episodes} episodes each[/]",
        border_style="cyan",
        padding=(1, 2),
    )


# ── Per-sandbox worker ────────────────────────────────────────────────────────
def run_sandbox(
    index: int,
    state: DemoState,
    daytona: Daytona,
    episodes: int,
    lr: float,
    task_code: str,
):
    sandbox = None
    sandbox_id = f"sb-{index:05d}"
    try:
        # Create sandbox
        sandbox = daytona.create(CreateSandboxParams(
            language="python",
            auto_stop_interval=10,       # auto-stop 10 min after inactivity
            auto_delete_interval=30,     # auto-delete 30 min after stopped
        ))
        sandbox_id = sandbox.id
        state.mark_running(index, sandbox_id)

        # Upload task script
        sandbox.fs.upload_file(
            task_code.encode(),
            "/home/daytona/cartpole_task.py",
        )

        # Install gymnasium
        sandbox.process.exec("pip install gymnasium --quiet")

        # Run training — capture stdout line by line
        response = sandbox.process.exec(
            f"python /home/daytona/cartpole_task.py {sandbox_id} {episodes} {lr}"
        )

        # Parse JSON lines from stdout
        for line in (response.result or "").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    state.update(index, sandbox_id, data)
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        state.mark_error(index, sandbox_id, str(e))
    finally:
        if sandbox:
            try:
                daytona.delete(sandbox)
            except Exception:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────
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
