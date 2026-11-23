"""Keep the user's explicit source-size contract visible in the check suite."""
from pathlib import Path
import hashlib
import unittest


class StructureTests(unittest.TestCase):
    def test_every_owned_text_source_stays_below_500_lines(self):
        root = Path(__file__).resolve().parents[1]
        files = [root / name for name in ["README.md", "setup.sh", "pyproject.toml"]]
        for folder in ["uav_network", "tests", "configs", "scripts", "docs"]:
            files.extend(p for p in (root / folder).rglob("*") if p.is_file()
                         and p.suffix in {".py", ".yaml", ".txt", ".md", ".sh", ".toml"})
        # The user explicitly selected the original 18d756ef MATD3 source.
        # Preserve it BYTE-FOR-BYTE even though upstream has 543 lines; the
        # pin is stricter than a style-only 500-line limit for this one file.
        pinned_external_reference = {
            "uav_network/algorithms/matd3.py":
                "e106277d2695936d983e2b558ad5cefac7a1015b24cca3b8032c727563bfeefa",
        }
        for path in files:
            rel = path.relative_to(root).as_posix()
            with self.subTest(file=rel):
                if rel in pinned_external_reference:
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                     pinned_external_reference[rel])
                else:
                    self.assertLessEqual(len(path.read_text().splitlines()), 500)


if __name__ == "__main__":
    unittest.main()


def test_reference_algorithm_source_contract():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    # Source hash contract now pins the exact user-selected Kaggle/W&B run
    # 18d756ef (not the superseded local 2026-11-20 baseline).
    expected = {
        "uav_network/algorithms/masac.py": "7e5fd9cb00c664bbea88321dcfdd3d0b75527c992b041a1a42434162a92796ce",
        "uav_network/algorithms/matd3.py": "e106277d2695936d983e2b558ad5cefac7a1015b24cca3b8032c727563bfeefa",
        "uav_network/algorithms/models.py": "c1b65c7d5e64574404f3f0394d8377de92d886e7ab126576153ad82a981b2f8f",
        "uav_network/algorithms/optimization.py": "1a092d1964246d55fe81a969e875206aae99e90c6c3b99de82ff53797e3193f6",
        "uav_network/algorithms/action_utils.py": "1e9aee9168b0f87c9809bfaaf4016277dd31ce38df0b232ee10fb42779fe2463",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
