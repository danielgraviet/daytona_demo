from dataclasses import dataclass
from typing import Optional


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
