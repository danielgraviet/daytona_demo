import time
from threading import Lock

from .sandbox_result import SandboxResult


class DemoState:
    def __init__(self, total: int):
        self.total      = total
        self.results    = {}
        self.lock       = Lock()
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
    def n_error(self):
        return sum(1 for r in self.results.values() if r.status == "error")

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
