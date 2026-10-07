"""Small CSV/7z fixtures test extraction recovery and schema validation."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.source_schema import SCHEMAS, validate_source_schema

HAS_7Z = importlib.util.find_spec("py7zr") is not None


class SchemaTests(unittest.TestCase):
    def test_all_eight_direct_csv_schemas(self):
        with tempfile.TemporaryDirectory() as folder:
            for stem, columns in SCHEMAS.items():
                with self.subTest(stem=stem):
                    path = Path(folder) / f"{stem}.csv"
                    path.write_text(",".join(sorted(columns)) + "\n")
                    self.assertEqual(set(validate_source_schema(path, stem)), columns)
                    path.write_text("wrong,headers\n")
                    with self.assertRaises(ValueError):
                        validate_source_schema(path, stem)

    @unittest.skipUnless(HAS_7Z, "py7zr unavailable")
    def test_all_eight_compressed_headers_and_bad_schema(self):
        import py7zr
        with tempfile.TemporaryDirectory() as folder:
            for stem, columns in SCHEMAS.items():
                with self.subTest(stem=stem):
                    csv = Path(folder) / f"{stem}.csv"
                    csv.write_text(",".join(sorted(columns)) + "\n" + "synthetic body\n" * 100)
                    archive = csv.with_suffix(".csv.7z")
                    with py7zr.SevenZipFile(archive, "w") as writer:
                        writer.write(csv, arcname=csv.name)
                    self.assertEqual(set(validate_source_schema(archive, stem)), columns)

    @unittest.skipUnless(HAS_7Z, "py7zr unavailable")
    def test_compressed_schema_mismatch_is_rejected(self):
        import py7zr
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "train.csv"
            csv.write_text("wrong,headers\n")
            archive = csv.with_suffix(".csv.7z")
            with py7zr.SevenZipFile(archive, "w") as writer:
                writer.write(csv, arcname=csv.name)
            with self.assertRaises(ValueError):
                validate_source_schema(archive, "train")


@unittest.skipUnless(HAS_7Z and importlib.util.find_spec("pandas"), "py7zr/pandas unavailable")
class CsvCacheTests(unittest.TestCase):
    def setUp(self):
        import py7zr
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.cache = self.root / "raw", self.root / "cache"
        self.raw.mkdir()
        self.cache.mkdir()
        self.body = "id,date,store_nbr,item_nbr,unit_sales,onpromotion\n1,2020-01-01,1,1,10,False\n"
        source = self.root / "train.csv"
        source.write_text(self.body)
        with py7zr.SevenZipFile(self.raw / "train.csv.7z", "w") as writer:
            writer.write(source, arcname="train.csv")

    def materialize(self):
        from src.preprocessing import materialize_csv
        return materialize_csv(self.raw, "train.csv", self.cache)

    def test_stale_legacy_staging_does_not_block_retry(self):
        legacy = self.cache / ".train.csv.extracting"
        legacy.mkdir()
        (legacy / "partial").write_text("preserve existing work")
        result = self.materialize()
        self.assertEqual(result.read_text(), self.body)
        self.assertTrue((legacy / "partial").exists())
        self.assertEqual(self.materialize(), result)

    def test_same_size_corruption_is_recovered_without_deleting_old_cache(self):
        original = self.materialize()
        original.write_text(self.body.replace(",10,", ",90,"))
        recovered = self.materialize()
        self.assertNotEqual(original, recovered)
        self.assertEqual(recovered.read_text(), self.body)
        self.assertIn(",90,", original.read_text())
        self.assertEqual(self.materialize(), recovered)

    def test_failed_extraction_cleans_only_its_unique_temporary_directory(self):
        import py7zr
        def interrupted(archive, path=None, **kwargs):
            (Path(path) / "train.csv").write_text("partial")
            raise OSError("synthetic interruption")
        with patch.object(py7zr.SevenZipFile, "extractall", autospec=True, side_effect=interrupted):
            with self.assertRaises(OSError):
                self.materialize()
        self.assertEqual(list(self.cache.rglob(".extract-*")), [])
        self.assertTrue((self.raw / "train.csv.7z").exists())
        self.assertEqual(self.materialize().read_text(), self.body)

    def test_corrupt_checksum_record_is_preserved_and_recovered(self):
        original = self.materialize()
        record = original.with_name("train.csv.integrity.json")
        record.write_text("invalid checksum metadata")
        recovered = self.materialize()
        self.assertNotEqual(original, recovered)
        self.assertEqual(recovered.read_text(), self.body)
        self.assertEqual(record.read_text(), "invalid checksum metadata")
        self.assertEqual(self.materialize(), recovered)


if __name__ == "__main__":
    unittest.main()
