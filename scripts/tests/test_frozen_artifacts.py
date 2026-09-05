"""Reproduction helper tests are separate from the frozen modeling test inventory."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_artifacts", ROOT / "frozen_artifacts.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ArtifactTransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "config").mkdir()
        shutil.copyfile(helper.ROOT / "config/model_config.json", self.root / "config/model_config.json")
        self.archive = self.root / "models.zip"
        helper.export_models(helper.ROOT, self.archive)

    def test_exact_roundtrip_and_idempotent_restore(self):
        self.assertEqual(helper.restore_models(self.root, self.archive), 9)
        self.assertEqual(helper.restore_models(self.root, self.archive), 0)
        for name in helper.expected_artifacts(self.root):
            self.assertEqual((self.root / name).read_bytes(), (helper.ROOT / name).read_bytes())

    def test_existing_different_model_is_preserved(self):
        (self.root / "models").mkdir()
        path = self.root / "models/model_b_xgb.json"
        path.write_bytes(b"different model")
        with self.assertRaisesRegex(helper.FreezeError, "checksum mismatch"):
            helper.restore_models(self.root, self.archive)
        self.assertEqual(path.read_bytes(), b"different model")
        self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_unexpected_path_missing_member_and_corruption_are_rejected(self):
        with zipfile.ZipFile(self.archive) as source:
            content = {name: source.read(name) for name in source.namelist()}
        for kind in ("extra", "missing", "corrupt", "duplicate"):
            candidate = self.root / f"{kind}.zip"
            with zipfile.ZipFile(candidate, "w") as bundle:
                for index, (name, data) in enumerate(content.items()):
                    if kind == "missing" and index == 0:
                        continue
                    bundle.writestr(name, b"corrupt" if kind == "corrupt" and index == 0 else data)
                if kind == "extra":
                    bundle.writestr("../outside.txt", b"invalid")
                if kind == "duplicate":
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bundle.writestr(next(iter(content)), b"duplicate")
            with self.subTest(kind=kind), self.assertRaises(helper.FreezeError):
                helper.restore_models(self.root, candidate)
            self.assertFalse((self.root / "models").exists())

    def test_export_does_not_replace_existing_archive(self):
        before = self.archive.read_bytes()
        with self.assertRaisesRegex(helper.FreezeError, "already exists"):
            helper.export_models(helper.ROOT, self.archive)
        self.assertEqual(before, self.archive.read_bytes())


if __name__ == "__main__":
    unittest.main()
