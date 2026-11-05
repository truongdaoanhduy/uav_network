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


def test_reference_algorithm_source_contract():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    # Repinned for the user-approved 2026-11-05 MASAC/MATD3 hidden-LayerNorm architecture.
    expected = {
        "uav_network/algorithms/masac.py": "04cfbe661a7e6d840967727b2de3af03c9343c081fbb7d9a8f346f793a20d55a",
        "uav_network/algorithms/matd3.py": "702c85b4ddd07ca554be3f27d01f2d7e3bd4277c9eb405363ffe504d985bf9ea",
        "uav_network/algorithms/models.py": "4a653aca0bb14bc10d2b695d98eb41b7e601b18ab5bd30795aa3e7eecf1f6d40",
        "uav_network/algorithms/optimization.py": "fae2798711325662ef5604e6271802e9bde631aa0c936e96e9d42a520eb351b8",
        "uav_network/algorithms/action_utils.py": "1e9aee9168b0f87c9809bfaaf4016277dd31ce38df0b232ee10fb42779fe2463",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
