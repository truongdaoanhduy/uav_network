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
    # Repinned for reviewed, user-approved alpha-dual and opt-in initialization changes.
    expected = {
        "uav_network/algorithms/masac.py": "097bd0b69b90d3323fd48733d2db0bc1a9dcc487fbf55d9e4edf4e447ad63b33",
        "uav_network/algorithms/matd3.py": "6555d6eaff2127c181a819492b2249b02e6119814561b0ad981e7ad24ba3db27",
        "uav_network/algorithms/models.py": "474a209f5172ad39dbb853108c94bea6c70e5cc89f5aa5b31dee89cfbbbead41",
        "uav_network/algorithms/optimization.py": "fae2798711325662ef5604e6271802e9bde631aa0c936e96e9d42a520eb351b8",
        "uav_network/algorithms/action_utils.py": "1e9aee9168b0f87c9809bfaaf4016277dd31ce38df0b232ee10fb42779fe2463",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
