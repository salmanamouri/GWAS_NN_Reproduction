from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import psutil
import torch


@dataclass
class ResourceMetrics:
    wall_time_seconds: float
    start_ram_mb: float
    end_ram_mb: float
    peak_ram_mb: float
    peak_gpu_allocated_mb: float | None
    peak_gpu_reserved_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryMonitor:
    """
    Periodically measures the RAM used by the current Python process.

    RSS memory is used because NumPy and PyTorch often allocate memory
    outside Python's standard object allocator.
    """

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            rss = self.process.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
            time.sleep(self.interval_seconds)

    def start(self) -> None:
        self.peak_bytes = self.process.memory_info().rss
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024**2)


def profile_callable(
    function: Callable[..., Any],
    *args: Any,
    profiling_device: str = "cpu",
    **kwargs: Any,
) -> tuple[Any, ResourceMetrics]:
    """
    Execute a function and measure:
    - wall-clock time
    - start/end RAM
    - peak RAM
    - peak GPU allocated and reserved memory, when CUDA is used
    """

    gc.collect()

    process = psutil.Process(os.getpid())
    start_ram_mb = process.memory_info().rss / (1024**2)

    use_cuda = (
        profiling_device.startswith("cuda")
        and torch.cuda.is_available()
    )

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    monitor = MemoryMonitor()
    monitor.start()

    start_time = time.perf_counter()

    try:
        result = function(*args, **kwargs)
    finally:
        if use_cuda:
            torch.cuda.synchronize()

        elapsed_seconds = time.perf_counter() - start_time
        monitor.stop()

    end_ram_mb = process.memory_info().rss / (1024**2)

    if use_cuda:
        peak_gpu_allocated_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
        )
        peak_gpu_reserved_mb = (
            torch.cuda.max_memory_reserved() / (1024**2)
        )
    else:
        peak_gpu_allocated_mb = None
        peak_gpu_reserved_mb = None

    metrics = ResourceMetrics(
        wall_time_seconds=float(elapsed_seconds),
        start_ram_mb=float(start_ram_mb),
        end_ram_mb=float(end_ram_mb),
        peak_ram_mb=float(monitor.peak_mb),
        peak_gpu_allocated_mb=(
            float(peak_gpu_allocated_mb)
            if peak_gpu_allocated_mb is not None
            else None
        ),
        peak_gpu_reserved_mb=(
            float(peak_gpu_reserved_mb)
            if peak_gpu_reserved_mb is not None
            else None
        ),
    )

    return result, metrics