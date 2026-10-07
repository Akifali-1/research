"""Synthetic stage recovery; no inference, model training, or real data."""
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation_recovery import EvaluationWorkspace, evaluation_lock
from src.integrity import atomic_json


class EvaluationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = ["stgt-test-attempt1", "gat_lstm-test-attempt1"]
        self.selection = {"experiment_ids": self.ids, "git_commit": "training-version"}
        atomic_json(self.root / "selection.json", self.selection)
        self.workspace = EvaluationWorkspace(self.root, self.ids)

    def produce(self, folder):
        (folder / "result.json").write_text('{"synthetic": true}')
        (folder / "predictions.npz").write_text("synthetic artifact, not actual predictions")

    def test_completed_stage_is_verified_without_repeating_producer(self):
        stage = self.workspace.stage("stgt", {"checkpoint_sha256": "a"}, self.produce, ["result.json", "predictions.npz"])
        before = (stage / "result.json").stat().st_mtime_ns
        def must_not_run(folder):
            self.fail("Completed inference stage was repeated")
        reused = EvaluationWorkspace(self.root, self.ids).stage("stgt", {"checkpoint_sha256": "a"}, must_not_run,
                                                               ["result.json", "predictions.npz"])
        self.assertEqual(reused, stage)
        self.assertEqual(before, (stage / "result.json").stat().st_mtime_ns)

    def test_failed_second_model_can_resume_without_repeating_first(self):
        first = self.workspace.stage("stgt", {}, self.produce, ["result.json", "predictions.npz"])
        def interrupted(folder):
            (folder / "result.json").write_text("partial")
            raise OSError("synthetic session interruption")
        with self.assertRaises(OSError):
            self.workspace.stage("gat_lstm", {}, interrupted, ["result.json", "predictions.npz"])
        self.assertTrue(first.exists())
        self.assertFalse((self.root / "stages/gat_lstm").exists())
        self.assertEqual(list((self.root / ".work").iterdir()), [])
        recovered = EvaluationWorkspace(self.root, self.ids)
        recovered.stage("stgt", {}, lambda folder: self.fail("STGT repeated"), ["result.json", "predictions.npz"])
        self.assertTrue(recovered.stage("gat_lstm", {}, self.produce, ["result.json", "predictions.npz"]).exists())
        self.assertEqual(json.loads((self.root / "selection.json").read_text()), self.selection)

    def test_changed_selection_or_checkpoint_is_rejected(self):
        self.workspace.stage("stgt", {"checkpoint_sha256": "a"}, self.produce, ["result.json", "predictions.npz"])
        with self.assertRaises(ValueError):
            EvaluationWorkspace(self.root, ["different", self.ids[1]])
        with self.assertRaises(ValueError):
            self.workspace.stage("stgt", {"checkpoint_sha256": "b"}, self.produce, ["result.json", "predictions.npz"])

    def test_corrupt_completed_stage_is_not_overwritten_or_recomputed(self):
        stage = self.workspace.stage("stgt", {}, self.produce, ["result.json", "predictions.npz"])
        path = stage / "predictions.npz"
        path.write_text("corrupt artifact")
        with self.assertRaises(ValueError):
            self.workspace.stage("stgt", {}, lambda folder: self.fail("Corrupt completed stage must not trigger inference"),
                                 ["result.json", "predictions.npz"])
        self.assertEqual(path.read_text(), "corrupt artifact")

    def test_legacy_partial_results_are_preserved(self):
        (self.root / "metrics").mkdir()
        original = self.root / "metrics/model_comparison.csv"
        original.write_text("legacy partial result")
        def report(folder):
            (folder / "model_comparison.csv").write_text("new synthetic report")
        new = self.workspace.stage("report", {}, report, ["model_comparison.csv"])
        self.assertEqual(original.read_text(), "legacy partial result")
        self.assertEqual((new / "model_comparison.csv").read_text(), "new synthetic report")

    def test_simultaneous_evaluators_are_blocked(self):
        with evaluation_lock(self.root):
            with self.assertRaises(FileExistsError):
                with evaluation_lock(self.root):
                    self.fail("Concurrent evaluator admitted")
        self.assertFalse((self.root / ".evaluation.lock").exists())


if __name__ == "__main__":
    unittest.main()
