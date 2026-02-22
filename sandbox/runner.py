import json

from daytona_sdk import Daytona, CreateSandboxBaseParams as CreateSandboxParams

from models.demo_state import DemoState


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
            auto_stop_interval=10,    # auto-stop 10 min after inactivity
            auto_delete_interval=30,  # auto-delete 30 min after stopped
        ))
        sandbox_id = sandbox.id
        state.mark_running(index, sandbox_id)

        # Upload task script
        sandbox.fs.upload_file(
            task_code.encode(),
            "/home/daytona/cartpole_task.py",
        )

        # Install gymnasium
        sandbox.process.exec("pip install gymnasium --quiet", timeout=120)

        # Run training — capture stdout line by line
        response = sandbox.process.exec(
            f"python /home/daytona/cartpole_task.py {sandbox_id} {episodes} {lr}",
            timeout=30,
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
