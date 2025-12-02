import os
import signal
import subprocess
from typing import Tuple

def cmd_exec(cmd: list[str], *, capture_output: bool = False) -> int | Tuple[int, str] | None:
    try:
        kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        kwargs["start_new_session"] = True
        collected: list[str] = []

        with subprocess.Popen(cmd, **kwargs) as p:
            try:
                for line in p.stdout:
                    if capture_output:
                        collected.append(line)
                    print(line, end="")
                rc = p.wait()
                print(f"Exit code: {rc}")
                if capture_output:
                    return rc, "".join(collected)
                return rc

            except KeyboardInterrupt:
                try:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGINT)
                    p.wait(timeout=5)
                except Exception:
                    pass

                try:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass

                raise
    except KeyboardInterrupt:
        print("\ninterrupted")
        if capture_output:
            return None, ""
        return None
