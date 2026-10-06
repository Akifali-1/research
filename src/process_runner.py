"""Stream subprocess output to the notebook and a durable log with a deadline."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _stop(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def run_logged(command, *, cwd, log_path, timeout, output=None):
    """Tee merged stdout/stderr without blocking deadline checks on readline.

    Tests use tiny plain Python subprocesses only. This helper neither launches
    experiments nor bypasses registry/push/Drive authorization checks.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output = sys.stdout if output is None else output
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    messages = queue.Queue()
    with path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   start_new_session=os.name == "posix")

        def read_output():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        def write(line):
            log.write(line)
            log.flush()
            output.write(line)
            output.flush()

        try:
            closed = False
            while not closed or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    line = messages.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if line is None:
                    closed = True
                else:
                    write(line)
            return subprocess.CompletedProcess(command, process.wait())
        except BaseException:
            _stop(process)
            reader.join(timeout=5)
            while True:
                try:
                    line = messages.get_nowait()
                except queue.Empty:
                    break
                if line is not None:
                    write(line)
            raise
        finally:
            reader.join(timeout=1)
            if not reader.is_alive():
                process.stdout.close()
