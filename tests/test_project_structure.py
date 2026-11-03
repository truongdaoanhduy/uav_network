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
    expected = {
        "uav_network/algorithms/masac.py": "f40586dbf19840e5a907e651b24ebbd7a4f8d493a5625f53e5d41b4e85d9bd1f",
        "uav_network/algorithms/matd3.py": "d2de70b5b842443994919eb418302fd2fa599dbda1f7f9ab75455ed35a658b93",
        "uav_network/algorithms/models.py": "264e1f716d72f3e6ab2c2e92e4dd9180e81bfd76bdf5a0f794f4fd3485f2cbcd",
        "uav_network/algorithms/optimization.py": "d5b4c72b3bf44cb91d86424aa5f6f1dd1afd73ae97d5594ff80e32623742bc8f",
        "uav_network/algorithms/action_utils.py": "1e9aee9168b0f87c9809bfaaf4016277dd31ce38df0b232ee10fb42779fe2463",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
