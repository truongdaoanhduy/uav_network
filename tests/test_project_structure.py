"""Keep the user's explicit source-size contract visible in the check suite."""
from pathlib import Path
import unittest


class StructureTests(unittest.TestCase):
    def test_every_owned_text_source_stays_below_500_lines(self):
        root = Path(__file__).resolve().parents[1]
        files = [root / name for name in ["README.md", "setup.sh", "pyproject.toml"]]
        for folder in ["uav_network", "tests", "configs", "scripts", "docs"]:
            files.extend(p for p in (root / folder).rglob("*") if p.is_file()
                         and p.suffix in {".py", ".yaml", ".txt", ".md", ".sh", ".toml"})
        for path in files:
            with self.subTest(file=str(path.relative_to(root))):
                self.assertLessEqual(len(path.read_text().splitlines()), 500)


if __name__ == "__main__":
    unittest.main()


def test_scenario_v2_keeps_standard_algorithm_sources_unchanged():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    expected = {
        "uav_network/algorithms/masac.py": "e6f38a562c7ae4a1bccaa5324689876450f4a75b4f4eb5f63e36c1b869660419",
        "uav_network/algorithms/matd3.py": "fee6b3692d0ac6115c340ab397bb8098c811a97331e0bca2cb0f8aab729ecf4b",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
