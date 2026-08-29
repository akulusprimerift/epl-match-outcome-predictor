"""Phase 0 checks for required imports and repository structure."""

from importlib import import_module
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    """Verify the Phase 0 project skeleton and runtime dependencies."""

    def test_project_package_imports(self) -> None:
        """The repository's source package is importable."""
        self.assertIsNotNone(import_module("src"))

    def test_required_dependencies_import(self) -> None:
        """The four required third-party libraries are importable."""
        for module_name in ("pandas", "sklearn", "xgboost", "matplotlib"):
            with self.subTest(module=module_name):
                self.assertIsNotNone(import_module(module_name))

    def test_required_directories_exist(self) -> None:
        """The repository contains every directory from the project tree."""
        required_directories = (
            "config",
            "data/raw/football_data",
            "data/raw/api_football",
            "data/processed",
            "models",
            "reports",
            "src",
            "tests/fixtures",
        )

        for relative_path in required_directories:
            with self.subTest(path=relative_path):
                self.assertTrue(
                    (PROJECT_ROOT / relative_path).is_dir(),
                    f"Missing required directory: {relative_path}",
                )

    def test_required_phase_zero_files_exist(self) -> None:
        """All Phase 0 files named by the specification are present."""
        required_files = (
            ".gitignore",
            ".env.example",
            "AGENTS.md",
            "EPL_MATCH_OUTCOME_PREDICTOR_SPEC.md",
            "README.md",
            "requirements.txt",
            "requirements.lock.txt",
            "src/__init__.py",
            "tests/__init__.py",
            "tests/test_project_structure.py",
        )

        for relative_path in required_files:
            with self.subTest(path=relative_path):
                self.assertTrue(
                    (PROJECT_ROOT / relative_path).is_file(),
                    f"Missing required file: {relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
