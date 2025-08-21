import pynvml
import time
import sys

def is_gpu_available() -> bool:
    try:
        pynvml.nvmlInit()
    except Exception:
        return False

    try:
        count = pynvml.nvmlDeviceGetCount()
        return count > 0
    except Exception:
        return False
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def exit_with_delay(delay: int = 10):
    print(f"Exiting in {delay} seconds...")
    time.sleep(delay)
    sys.exit(1)