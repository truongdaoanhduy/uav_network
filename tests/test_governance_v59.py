"""Unit probes for changed-scope guard, including adversarial new files."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source_governance", ROOT / "scripts/check_project_policy.py")
assert spec and spec.loader
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class ProjectPolicyTests(unittest.TestCase):
    def test_new_501_line_source_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            target = root / "scripts/new_source.py"
            target.parent.mkdir()
            target.write_text("pass\n" * 501)
            with patch.object(policy, "ROOT", root):
                errors = policy.violations("scripts/new_source.py", full=False)
            self.assertTrue(any("500" in e for e in errors))

    def test_new_500_line_source_allowed(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            target = root / "scripts/new_source.py"
            target.parent.mkdir()
            target.write_text("pass\n" * 500)
            with patch.object(policy, "ROOT", root):
                errors = policy.violations("scripts/new_source.py", full=False)
            self.assertFalse(errors)

    def test_disallowed_flat_new_research_code(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            name = "research/zhang_xue/src/monster.py"
            file = root / name
            file.parent.mkdir(parents=True)
            file.write_text("pass\n")
            with patch.object(policy, "ROOT", root):
                errors = policy.violations(name, full=False)
            self.assertTrue(any("package folder" in e for e in errors))

    def test_tracked_secret_name_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            secret = root / "key.txt"
            secret.write_text("fake-value\n")
            with patch.object(policy, "ROOT", root):
                errors = policy.violations("key.txt", full=False)
            self.assertTrue(any("credential" in e for e in errors))

    def test_large_notebook_rejected_for_new_commit(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            file = root / "notebooks/new.ipynb"
            file.parent.mkdir()
            file.write_bytes(b"0" * 250_001)
            with patch.object(policy, "ROOT", root):
                errors = policy.violations("notebooks/new.ipynb", full=False)
            self.assertTrue(any("heavy artifact" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
