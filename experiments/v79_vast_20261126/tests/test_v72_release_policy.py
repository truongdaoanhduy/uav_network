"""Release scanning must recognize Kaggle tokens without using real credentials."""
from scripts.maintenance import check_project_policy as policy


def test_source_policy_detects_kaggle_token_in_arbitrary_source_file(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    # Deliberately synthetic token assembled at runtime, never a user credential.
    token = 'KG' + 'AT_' + 'v' * 32
    (tmp_path / 'accidental.py').write_text('value = ' + repr(token))
    assert 'possible access token in source' in policy.violations('accidental.py', full=True)
