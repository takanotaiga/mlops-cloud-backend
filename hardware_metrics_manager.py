import time
from typing import Any, Dict, List, Optional

import psutil
import pynvml

from backend_module.database import DataBaseManager
from backend_module.config import load_surreal_config
from backend_module import gpu_check

def _gather_system_metrics() -> Dict[str, Any]:
    """Collect system-level CPU and memory metrics.

    Returns a dict with totals and percents for memory, and CPU utilization percent.
    """
    vm = psutil.virtual_memory()
    # psutil.cpu_percent() is non-blocking if called periodically; first call is priming.
    cpu_percent = psutil.cpu_percent(interval=0.0)
    return {
        "cpu_percent": cpu_percent,
        "memory": {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "free": vm.free,
            "percent": vm.percent,
        },
        # Optional: load average where available
        "load_average": list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None,
    }


def _safe_nvml_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _gather_gpu_metrics() -> List[Dict[str, Any]]:
    """Collect per-GPU metrics via NVML (if available).

    Includes utilization, memory, power, temperature, fan, PCIe throughput, and clocks.
    Returns empty list if NVML is unavailable or no GPUs present.
    """
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    gpus: List[Dict[str, Any]] = []
    try:
        count = _safe_nvml_call(pynvml.nvmlDeviceGetCount, 0) or 0
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except Exception:
                continue

            def _name() -> Optional[str]:
                try:
                    n = pynvml.nvmlDeviceGetName(handle)
                    return n.decode() if isinstance(n, bytes) else str(n)
                except Exception:
                    return None

            util = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle), None)
            memi = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle), None)
            temp = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU), None)
            fan = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetFanSpeed(handle), None)
            pow_mw = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), None)
            pow_w = (pow_mw / 1000.0) if isinstance(pow_mw, (int, float)) else None
            pcie_tx_kbs = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES), None)
            pcie_rx_kbs = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES), None)
            sm_clock = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM), None)
            mem_clock = _safe_nvml_call(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM), None)

            gpus.append({
                "index": i,
                "name": _name(),
                "utilization": {
                    "gpu_percent": getattr(util, "gpu", None) if util is not None else None,
                    "memory_percent": getattr(util, "memory", None) if util is not None else None,
                },
                "memory": {
                    "total": getattr(memi, "total", None) if memi is not None else None,
                    "used": getattr(memi, "used", None) if memi is not None else None,
                    "free": getattr(memi, "free", None) if memi is not None else None,
                },
                "power_watts": pow_w,
                "temperature_c": temp,
                "fan_speed_percent": fan,
                "pcie": {
                    "tx_kb_s": pcie_tx_kbs,
                    "rx_kb_s": pcie_rx_kbs,
                },
                "clocks_mhz": {
                    "sm": sm_clock,
                    "mem": mem_clock,
                },
            })
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


class HardwareMetricsManager:
    """Polls hardware metrics each second and persists to SurrealDB.

    - Records: system CPU/memory and per-GPU metrics (if NVML available).
    - Retention: deletes rows older than 10 minutes.
    - Table: `hardware_metric` with fields { timestamp, system, gpus }.
    """

    def __init__(self, db: DataBaseManager, *, cleanup_interval_sec: int = 30):
        self.db = db
        self.cleanup_interval_sec = cleanup_interval_sec
        self._last_cleanup_ts = 0.0

        # Prime CPU percent measurement
        try:
            psutil.cpu_percent(interval=0.0)
        except Exception:
            pass

    def _insert_metrics(self, system: Dict[str, Any], gpus: List[Dict[str, Any]]):
        # Use CREATE ... SET for SurrealDB and avoid reserved name 'timestamp'.
        self.db.query(
            "CREATE hardware_metric SET ts = time::now(), system = $SYS, gpus = $GPUS",
            {"SYS": system, "GPUS": gpus},
        )

    def _cleanup_old(self):
        """Delete rows older than 10 minutes."""
        self.db.query(
            "DELETE FROM hardware_metric WHERE ts < time::now() - 10m;"
        )

    def run_forever(self, *, interval_sec: float = 1.0):
        while True:
            started = time.time()
            system = _gather_system_metrics()
            gpus = _gather_gpu_metrics()
            self._insert_metrics(system, gpus)

            now = time.time()
            if (now - self._last_cleanup_ts) >= self.cleanup_interval_sec:
                self._cleanup_old()
                self._last_cleanup_ts = now

            # Maintain roughly 1 second sampling
            elapsed = time.time() - started
            sleep_for = max(0.0, interval_sec - elapsed)
            time.sleep(sleep_for)


def main():
    if gpu_check.is_gpu_available():
        print("GPU detected ✅")
    else:
        print("No GPU detected ❌")
        gpu_check.exit_with_delay()

    cfg = load_surreal_config()
    db = DataBaseManager(
        endpoint_url=cfg["endpoint_url"],
        username=cfg["username"],
        password=cfg["password"],
        namespace=cfg["namespace"],
        database=cfg["database"],
    )
    mgr = HardwareMetricsManager(db)
    try:
        mgr.run_forever()
    except KeyboardInterrupt:
        # Graceful exit
        pass


if __name__ == "__main__":
    main()
