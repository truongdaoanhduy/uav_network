"""Repository-wide maintenance entry points live under explicit namespace folders."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_governance_tool_belongs_to_maintenance_folder():
    assert (ROOT/'scripts/maintenance/check_project_policy.py').is_file()
    assert not (ROOT/'scripts/check_project_policy.py').exists()
