"""Tiny subprocess tests for live logging; no dataset/model/experiment runs."""
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.process_runner import run_logged


class ProcessLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_stdout_stderr_are_visible_and_saved(self):
        output = io.StringIO()
        result = run_logged([sys.executable, "-u", "-c", "import sys; print('epoch 1'); print('warning', file=sys.stderr)"],
                            cwd=self.root, log_path=self.root / "training.log", timeout=5, output=output)
        self.assertEqual(result.returncode, 0)
        self.assertIn("epoch 1", output.getvalue())
        self.assertIn("warning", output.getvalue())
        self.assertEqual(output.getvalue(), (self.root / "training.log").read_text())

    def test_silent_process_still_times_out_and_preserves_prior_output(self):
        start = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_logged([sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(20)"],
                       cwd=self.root, log_path=self.root / "timeout.log", timeout=1, output=io.StringIO())
        self.assertLess(time.monotonic() - start, 8)
        self.assertIn("started", (self.root / "timeout.log").read_text())

    def test_nonzero_exit_is_returned_for_registry_failure_handling(self):
        result = run_logged([sys.executable, "-c", "raise SystemExit(3)"],
                            cwd=self.root, log_path=self.root / "failure.log", timeout=5, output=io.StringIO())
        self.assertEqual(result.returncode, 3)

    @unittest.skipUnless(os.name == "posix", "Colab/Linux process-group behavior")
    def test_timeout_stops_descendant_after_parent_has_exited(self):
        code = "import subprocess, sys; subprocess.Popen([sys.executable, '-u', '-c', \"import time; print('child', flush=True); time.sleep(20)\"]); print('parent exits', flush=True)"
        start = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_logged([sys.executable, "-u", "-c", code], cwd=self.root,
                       log_path=self.root / "descendant.log", timeout=1, output=io.StringIO())
        self.assertLess(time.monotonic() - start, 4)
        self.assertIn("child", (self.root / "descendant.log").read_text())


if __name__ == "__main__":
    unittest.main()
