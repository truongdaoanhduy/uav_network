"""Kaggle account2 private GPU smoke must be reproducible and credential-free."""
import base64
import io
import json
from pathlib import Path
import zipfile

from scripts.build_zhang_xue_v43_kaggle_cuda_smoke import (
    build_kaggle_cuda_smoke,
    PINNED_SOURCE_COMMIT,
    SOURCE_PATHS,
)


def test_builder_embeds_only_pinned_source_no_credentials(tmp_path):
    root=Path(__file__).resolve().parents[1]
    files=build_kaggle_cuda_smoke(
        root,owner="testaccount",slug="zx-v43-smoke-test",dest=tmp_path,
    )
    metadata=json.loads(files["metadata"].read_text())
    assert metadata["id"]=="testaccount/zx-v43-smoke-test"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"]=="script"
    content=files["script"].read_text()
    assert PINNED_SOURCE_COMMIT in content
    assert "KAGGLE_API_TOKEN" not in content
    assert "key.txt" not in content
    assert "cuda:0" in content
    assert "cuda:1" in content
    assert "training" not in content.lower()
    import ast
    ast.parse(content)
    import re
    match=re.search(r'BASE64_SOURCE = "([A-Za-z0-9+/=]+)"',content)
    assert match is not None
    archive=zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1))))
    assert set(archive.namelist())==set(SOURCE_PATHS)
    assert len(archive.namelist())<20
    assert not any(name.endswith((".pem","key.txt","kaggle.json")) for name in archive.namelist())
