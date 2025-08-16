import os
import signal
import subprocess

def cmd_exec(cmd: list[str]):
    try:
        kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        kwargs["start_new_session"] = True

        with subprocess.Popen(cmd, **kwargs) as p:
            try:
                for line in p.stdout:
                    print(line, end="")
                rc = p.wait()
                print(f"Exit code: {rc}")
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