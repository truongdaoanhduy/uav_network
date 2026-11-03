"""Matched experimental conditions, especially the PQL replica and vmap toggle."""
from scripts.prepare_matched_architecture_study import prepare, scientific_settings
from uav_network.configuration import load_config


def test_ten_architecture_profiles_differ_only_in_intended_axes(tmp_path):
    manifest = prepare(tmp_path / "study")
    rows = manifest["cases"]
    assert len(rows) == 10 and not manifest["launched"]
    configs = [load_config(item["config"]) for item in rows]
    assert all(c.runtime.seed == 44 and c.runtime.deterministic
               and c.runtime.collector_overlap is False for c in configs)
    assert all(c.runtime.critic_chunk_size == 0 and c.runtime.amp_mode == "off"
               and c.runtime.compile_models is False for c in configs)
    assert scientific_settings(configs[3]) == scientific_settings(configs[9])
    vmap = configs[8]
    vmap.runtime.vectorize_critics = False
    assert scientific_settings(configs[1]) == scientific_settings(vmap)
    for c in configs:
        assert c.world.motion_dimensions == 3 and c.sensor.model == "bayesian"
        assert c.matd3.output_normalization is False
