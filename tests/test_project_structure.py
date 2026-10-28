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
        "uav_network/algorithms/masac.py": "7fe0dc5d417d26f83c0679841fe3841564d45fd799b13cd898f54f9a7f21ae4b",
        "uav_network/algorithms/matd3.py": "91461b6b30324a6c8f5fcb4082301f8f40a0c959c15d8c0fd9f2bc93a5ee90f1",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
