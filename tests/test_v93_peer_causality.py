"""The peer utility claim must follow from actual delivered origin evidence."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/gpu"))
from v92_network_and_peer_probe import run_peer_relay_partition_probe


def test_peer_probe_demonstrates_an_actual_causal_delivery_difference():
    result = run_peer_relay_partition_probe(
        ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml", "cpu", 44)
    assert result["peer_relay_unlocks_gcs_confirmation"], result

