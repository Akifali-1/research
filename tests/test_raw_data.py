"""Tiny source-layout fixtures; never access or preprocess the real dataset."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.raw_data import SOURCE_STEMS, extract_outer_zip, prepare_raw_sources, source_paths


class RawLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_zip(self, missing=None):
        zip_path = self.root / "favorita.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for stem in SOURCE_STEMS:
                if stem != missing:
                    archive.writestr(f"dataset/{stem}.csv.7z", f"tiny synthetic {stem}")
        return zip_path

    def test_outer_zip_preserved_and_sources_flattened(self):
        zip_path = self.make_zip()
        raw = prepare_raw_sources(self.root / "raw", zip_path)
        self.assertTrue(zip_path.exists())
        self.assertEqual(len(source_paths(raw)[0]), 8)
        self.assertEqual(source_paths(raw)[1], [])
        self.assertFalse((raw / "dataset").exists())
        # Rerun never rewrites completed source files.
        before = (raw / "train.csv.7z").stat().st_mtime_ns
        extract_outer_zip(zip_path, raw)
        self.assertEqual(before, (raw / "train.csv.7z").stat().st_mtime_ns)

    def test_csv_layout_and_single_nested_folder_are_accepted(self):
        nested = self.root / "raw" / "favorita"
        nested.mkdir(parents=True)
        for stem in SOURCE_STEMS:
            (nested / f"{stem}.csv").write_text("fixture\n")
        self.assertEqual(prepare_raw_sources(self.root / "raw"), nested)

    def test_missing_error_lists_present_files_and_all_missing_sources(self):
        (self.root / "train.csv").write_text("fixture\n")
        with self.assertRaises(FileNotFoundError) as caught:
            prepare_raw_sources(self.root)
        message = str(caught.exception)
        self.assertIn("Present: train.csv", message)
        self.assertIn("holidays_events", message)
        self.assertIn("SOURCE_ZIP", message)

    def test_incomplete_zip_is_rejected_before_source_extraction(self):
        raw = self.root / "raw"
        with self.assertRaises(ValueError):
            extract_outer_zip(self.make_zip(missing="items"), raw)
        self.assertFalse(raw.exists())

    def test_conflicting_existing_file_is_not_overwritten(self):
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "train.csv.7z").write_text("user source")
        with self.assertRaises(FileExistsError):
            extract_outer_zip(self.make_zip(), raw)
        self.assertEqual((raw / "train.csv.7z").read_text(), "user source")
        self.assertEqual(len(list(raw.iterdir())), 1)

    def test_zip_traversal_is_rejected(self):
        zip_path = self.make_zip()
        with zipfile.ZipFile(zip_path, "a") as archive:
            archive.writestr("../train.csv.7z", "unsafe")
        raw = self.root / "raw"
        with self.assertRaises(ValueError):
            extract_outer_zip(zip_path, raw)
        self.assertFalse(raw.exists())


if __name__ == "__main__":
    unittest.main()
