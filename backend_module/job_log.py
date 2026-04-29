from __future__ import annotations

import contextlib
import sys
import threading
from typing import Iterator, Optional, TextIO

from backend_module.database import DataBaseManager
from backend_module.object_storage import MinioS3Uploader
from query.inference_job_log_query import append_job_log


class _JobLogCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def next(self) -> int:
        with self.lock:
            self.value += 1
            return self.value


class _JobLogStream:
    def __init__(
        self,
        *,
        db_manager: DataBaseManager,
        job_id: str,
        source: str,
        stream: str,
        wrapped: TextIO,
        counter: _JobLogCounter,
        archive_uploader: Optional[MinioS3Uploader] = None,
    ):
        self.db_manager = db_manager
        self.job_id = job_id
        self.source = source
        self.stream = stream
        self.wrapped = wrapped
        self.counter = counter
        self.archive_uploader = archive_uploader
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        try:
            self.wrapped.write(data)
            self.wrapped.flush()
        except Exception:
            pass
        with self._lock:
            self._buffer += data.replace("\r", "\n")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line)
        return len(data)

    def flush(self) -> None:
        try:
            self.wrapped.flush()
        except Exception:
            pass
        with self._lock:
            if self._buffer:
                line = self._buffer
                self._buffer = ""
                self._emit(line)

    def _emit(self, line: str) -> None:
        message = line.rstrip()
        if not message:
            return
        try:
            append_job_log(
                self.db_manager,
                job_id=self.job_id,
                source=self.source,
                stream=self.stream,
                message=message[-4000:],
                seq=self.counter.next(),
                archive_uploader=self.archive_uploader,
            )
        except Exception:
            # Logging must never break the actual inference pipeline.
            pass

    def isatty(self) -> bool:
        return bool(getattr(self.wrapped, "isatty", lambda: False)())


@contextlib.contextmanager
def capture_job_logs(
    db_manager: DataBaseManager,
    *,
    job_id: str,
    source: str,
    archive_uploader: Optional[MinioS3Uploader] = None,
) -> Iterator[None]:
    counter = _JobLogCounter()
    stdout = _JobLogStream(
        db_manager=db_manager,
        job_id=job_id,
        source=source,
        stream="stdout",
        wrapped=sys.stdout,
        counter=counter,
        archive_uploader=archive_uploader,
    )
    stderr = _JobLogStream(
        db_manager=db_manager,
        job_id=job_id,
        source=source,
        stream="stderr",
        wrapped=sys.stderr,
        counter=counter,
        archive_uploader=archive_uploader,
    )
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()
