"""Bundle only reviewed local sources into one Kaggle API code payload.

The installed Kaggle CLI transmits only kernel-metadata.json's code_file;
sibling files in --path are not uploaded. Therefore the kernel.py bootstrap
embeds this complete, compressed source archive directly.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import tarfile


# No credentials, workspaces, .git directories, models or outputs are included.
RUNTIME_SCRIPTS = {
    "scripts/requirements-network.txt",
    "scripts/upstream-revision.txt",
    "scripts/benchmark_pql.py",
    "scripts/compare_training_gpu.py",
}
UPSTREAM_CODE_DIRS = {
    "allocation", "energy", "entities", "mac", "mobility", "path_planning",
    "phy", "routing", "scene", "simulator", "telemetry", "topology", "utils",
}


def _revision(root):
    """Read the local detached HEAD without any Git subprocess."""
    head = (Path(root) / ".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        head = (Path(root) / ".git" / head[5:]).read_text().strip()
    return head


def build_source_archive(project_root):
    """Return a safe local runtime bundle. No remote commands or secrets."""
    root = Path(project_root).resolve()
    upstream = root / "third_party/UavNetSim"
    if not (upstream / "simulator/simulator.py").is_file():
        raise FileNotFoundError("Missing local pinned third_party/UavNetSim source; never fetch inside Kaggle")
    required = (root / "scripts/upstream-revision.txt").read_text().strip()
    if _revision(upstream) != required:
        raise ValueError("Local UavNetSim version is not the pinned upstream revision")

    included = {"pyproject.toml": root / "pyproject.toml"}
    for name in RUNTIME_SCRIPTS:
        included[name] = root / name
    for directory, suffixes in (("uav_network", {".py"}),
                                ("configs", {".yaml", ".yml", ".json"}),
                                ("tests", {".py"})):
        for source in (root / directory).rglob("*"):
            if (source.is_file() and source.suffix in suffixes and
                    not any(part.startswith(".") or part == "__pycache__"
                            for part in source.relative_to(root).parts)):
                included[source.relative_to(root).as_posix()] = source
    for source in upstream.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(upstream)
        if (relative.parts[0] not in UPSTREAM_CODE_DIRS or
                source.suffix not in {".py", ".json", ".txt", ".yaml", ".yml", ".xml"}):
            continue
        included[(Path("third_party/UavNetSim") / relative).as_posix()] = source
    for needed in ("pyproject.toml", "scripts/requirements-network.txt",
                   "scripts/upstream-revision.txt", "third_party/UavNetSim/simulator/simulator.py"):
        if needed not in included:
            raise FileNotFoundError(f"Missing required source file: {needed}")
    if not included or len(included) > 800:
        raise ValueError("Unexpected bundled source file count")
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz", compresslevel=9) as tar:
        for name, path in sorted(included.items()):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
                raise ValueError("Unsafe or oversized source entry: " + name)
            payload = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, BytesIO(payload))
    data = archive.getvalue()
    if len(data) > 2_000_000:
        raise ValueError(f"Bundle is too large for Kaggle kernel code API: {len(data)} bytes")
    return data


def source_sha256(archive):
    return hashlib.sha256(archive).hexdigest()
