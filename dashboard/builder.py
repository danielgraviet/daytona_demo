from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from models.demo_state import DemoState


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
    header.append("time: ", style="dim")
    header.append(f"{elapsed}s elapsed   ", style="bold cyan")
    header.append("run: ", style="dim")
    header.append(f"{n_run} running   ", style="bold green")
    header.append("done: ", style="dim")
    header.append(f"{n_done}/{state.total} complete   ", style="bold white")
    if state.n_error:
        header.append("err: ", style="bold red")
        header.append(f"{state.n_error} failed   ", style="bold red")
    header.append("solved: ", style="dim")
    header.append(f"{state.n_solved} ({solved_pct}%)   ", style="bold yellow")
    header.append("^ score: ", style="dim")
    header.append(f"avg {state.avg_final}", style="bold magenta")

    # ── Results table ─────────────────────────────────────────────────────
    with state.lock:
        sample = sorted(state.results.values(), key=lambda r: r.index)

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#",        style="dim",      width=5)
    table.add_column("Sandbox",  style="dim cyan", width=14)
    table.add_column("Status",   width=10)
    table.add_column("Episode",  justify="right",  width=9)
    table.add_column("Avg(100)", justify="right",  width=10)
    table.add_column("Best",     justify="right",  width=8)
    table.add_column("Solved",   justify="center", width=8)
    table.add_column("LR",       justify="right",  width=8)

    status_styles = {
        "pending":  ("dim", "..."),
        "running":  ("green", "> running"),
        "complete": ("bold white", "+ done"),
        "error":    ("red", "x error"),
    }

    for r in sample:
        style, label = status_styles.get(r.status, ("dim", r.status))
        solved_icon  = "[bold green]+[/]" if r.solved else "[dim]-[/]"
        solved_ep    = f" ep{r.solved_at}" if r.solved_at else ""
        table.add_row(
            str(r.index + 1),
            r.sandbox_id[:12] + "..." if len(r.sandbox_id) > 12 else r.sandbox_id,
            f"[{style}]{label}[/]",
            str(r.episode) if r.episode else "–",
            str(r.avg_100) if r.avg_100 else "–",
            str(r.best)    if r.best    else "–",
            f"{solved_icon}{solved_ep}",
            str(r.lr)      if r.lr      else "–",
        )

    layout = Layout(name="root")
    layout.split_column(
        Layout(header, size=1),
        Layout(table),
    )
    return Panel(
        layout,
        title="[bold #00FF41]Daytona Infra  ·  CartPole-v1 Distributed RL Demo[/]",
        subtitle=f"[dim]{state.total} sandboxes  ·  {episodes} episodes each[/]",
        border_style="#00FF41",
        padding=(1, 2),
    )
