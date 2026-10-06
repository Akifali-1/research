"""WSL-compatible synthetic budget tests; standard library only."""
import tempfile
import ast
import json
import unittest
from pathlib import Path

from src.experiment_registry import ActiveExperiment, BudgetExceeded, RecoveryRequired, Registry


class BudgetSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = Registry(self.root / "registry.json")

    def reserve(self, signature="one", retry=False):
        return self.registry.reserve(model="stgt", signature=signature, configuration={"seed": 42},
                                     git_commit="synthetic", dataset_manifest={}, artifact_dir=self.root / "attempts",
                                     force_retry=retry)

    def test_ten_run_limit_survives_restart(self):
        for i in range(10):
            record = self.reserve(str(i))["record"]
            self.registry.update(record["experiment_id"], "failed")
        self.registry = Registry(self.root / "registry.json")
        with self.assertRaises(BudgetExceeded):
            self.reserve("eleven")

    def test_completed_skip_even_at_limit(self):
        self.registry = Registry(self.root / "registry.json", max_runs=1)
        record = self.reserve()["record"]
        self.registry.update(record["experiment_id"], "completed")
        self.assertEqual(self.reserve()["action"], "skip_completed")
        self.assertEqual(self.registry.status()["submitted_runs"], 1)

    def test_two_retries_maximum_and_unique_directories(self):
        paths = set()
        for i in range(3):
            record = self.reserve(retry=i > 0)["record"]
            paths.add(record["artifact_dir"])
            self.registry.update(record["experiment_id"], "failed")
        self.assertEqual(len(paths), 3)
        with self.assertRaises(BudgetExceeded):
            self.reserve(retry=True)

    def test_interruption_requires_confirmation(self):
        record = self.reserve()["record"]
        with self.assertRaises(ActiveExperiment):
            self.reserve("two")
        self.registry.mark_interrupted_active("synthetic interruption")
        with self.assertRaises(RecoveryRequired):
            self.reserve()
        retried = self.reserve(retry=True)["record"]
        self.assertEqual(retried["attempt"], 2)
        self.assertNotEqual(record["artifact_dir"], retried["artifact_dir"])

    def test_no_unapproved_budget_increase(self):
        with self.assertRaises(ValueError):
            Registry(self.root / "registry.json", max_runs=11)

    def test_deadline_does_not_reset(self):
        deadline = self.registry.establish_deadline(60)
        self.assertEqual(Registry(self.root / "registry.json").establish_deadline(3600), deadline)


class NotebookSafetyTests(unittest.TestCase):
    def test_cell_syntax_and_order_without_execution(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/colab_experiment.ipynb").read_text())
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertFalse(cell.get("outputs"))
            source = "".join(cell["source"])
            ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))))
            code.append(source)
        joined = "\n".join(code)
        self.assertLess(joined.index("drive.mount("), joined.index("prepare_data(PROJECT"))
        self.assertLess(joined.index("COLAB_CONFIG.write_text("), joined.index("--models"))
        self.assertLess(joined.index("--models"), joined.index("--finalize-test"))


if __name__ == "__main__":
    unittest.main()
