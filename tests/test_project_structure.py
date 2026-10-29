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


def test_scientific_algorithm_source_contract_with_chunked_execution():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    expected = {
        "uav_network/algorithms/masac.py": "f7f1bad7fd6c7a9775e537e58d77883c527f507518ca7ea5bd4aa750e8432f66",
        "uav_network/algorithms/matd3.py": "1020e672b46fe21231bb53bec03ee21f12d6c4fa2a455d49260fdb8e234a26a0",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
