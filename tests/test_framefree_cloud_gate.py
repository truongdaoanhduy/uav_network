"""Public, credential-free cloud gate for Framefree CUDA exactness."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class FramefreeCloudGateTests(unittest.TestCase):
    def module(self):
        from uav_network.experiments import framefree_cuda_gate
        return framefree_cuda_gate

    def test_metadata_is_public_gpu_and_has_no_private_sources(self):
        m = self.module()
        metadata = m.kernel_metadata("owner/framefree-exact")
        self.assertFalse(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertTrue(metadata["enable_internet"])
        for key in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
            self.assertEqual(metadata[key], [])

    def test_kernel_script_pins_supplied_commit_and_contains_no_credentials(self):
        m = self.module()
        commit = "a" * 40
        script = m.kernel_script(commit)
        self.assertIn(commit, script)
        self.assertIn("tests/test_framefree_cuda.py", script)
        for forbidden in ("WANDB_API_KEY", "HF_TOKEN", "KAGGLE_API_TOKEN", "key.txt"):
            self.assertNotIn(forbidden, script)
        with self.assertRaises(ValueError):
            m.kernel_script("deadbeef")

    def test_prepare_writes_public_metadata_and_exact_commit(self):
        m = self.module()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "gate"
            m.prepare_gate(output, "owner/framefree-exact", "b" * 40)
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            script = (output / "kernel.py").read_text()
            self.assertFalse(metadata["is_private"])
            self.assertEqual(metadata["id"], "owner/framefree-exact")
            self.assertIn("b" * 40, script)

    def test_verify_remote_downloads_public_result_and_validates_exact_commit(self):
        m = self.module()
        commit = "d" * 40
        calls = []
        class Result:
            def __init__(self, stdout=""):
                self.stdout = stdout; self.stderr = ""; self.returncode = 0
        with TemporaryDirectory() as directory:
            output = Path(directory) / "download"
            def run(args, **kwargs):
                calls.append(args)
                if args[1:3] == ["kernels", "status"]:
                    return Result("Kernel has status 'KernelWorkerStatus.COMPLETE'")
                destination = Path(args[args.index("-p") + 1]); destination.mkdir(parents=True, exist_ok=True)
                (destination / "framefree_cuda_result.json").write_text(json.dumps({
                    "commit": commit, "torch_cuda": True, "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"], "pytest_returncode": 0,
                    "pytest_stdout": "all passed", "pytest_stderr": ""}))
                return Result()
            result = m.verify_remote("owner/framefree-exact", commit, output, runner=run)
            self.assertEqual(result["commit"], commit)
            self.assertEqual(calls[0][:3], ["kaggle", "kernels", "status"] )
            self.assertEqual(calls[1][:3], ["kaggle", "kernels", "output"] )

    def test_cli_prepare_writes_public_bundle(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "gate"
            proc = subprocess.run([sys.executable, "scripts/framefree_cuda_gate.py", "prepare",
                                   "--kernel-ref", "owner/framefree-exact",
                                   "--commit", "e" * 40, "--output", str(output)],
                                  cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            self.assertFalse(metadata["is_private"])
            self.assertIn("e" * 40, (output / "kernel.py").read_text())

    def test_validate_result_requires_exact_commit_cuda_and_green_pytest(self):
        m = self.module()
        commit = "c" * 40
        valid = {"commit": commit, "torch_cuda": True, "gpu_count": 2,
                 "gpu_names": ["Tesla T4", "Tesla T4"], "pytest_returncode": 0,
                 "pytest_stdout": "25 passed", "pytest_stderr": ""}
        self.assertEqual(m.validate_result(valid, commit)["pytest_returncode"], 0)
        for change in ({"commit": "d" * 40}, {"torch_cuda": False},
                       {"gpu_count": 0}, {"pytest_returncode": 1}):
            payload = dict(valid); payload.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_result(payload, commit)


if __name__ == "__main__":
    unittest.main()
