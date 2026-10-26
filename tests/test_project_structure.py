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
        "uav_network/algorithms/masac.py": "5bd140d93ee8269c395c0dfe929afbc6a5fa1a7e979d303b45056c130c00864a",
        "uav_network/algorithms/matd3.py": "0a25fc4966596242f474f68a9289b367672781d08e2a8a9beee937fd1bce962c",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
