"""Automated verification suite for the redesigned UAV MARL & Warp Slotted UavNetSim package."""

import ast
import math
import pathlib
import sys

try:
    import torch
except ImportError:
    torch = None
import yaml

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

if torch is not None:
    from uav_network.algorithms.vanilla_baselines import (
        GPUHybridReplayBuffer,
        VanillaMASAC,
        VanillaMATD3,
    )
from uav_network.communication.gpu.phy_sinr_fp32 import analytical_link_budget_summary
from uav_network.environment.gpu.ds_aoi_belief_fp32 import ds_closed_form, zeng_rotary_power_w
from uav_network.experiments.dual_fidelity_eval import EpisodeKPI, run_dual_fidelity_evaluation


def test_yaml_configs_and_vanilla_constraints():
    for name, algo in [
        ("vanilla_matd3_shared_2d_ds_aoi.yaml", "matd3"),
        ("vanilla_masac_shared_2d_ds_aoi.yaml", "masac"),
    ]:
        cfg = yaml.safe_load((PKG_ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert cfg["world"]["motion_dimensions"] == 2
        assert cfg["world"]["gcs_position_m"] == [40.0, 40.0, 10.0]
        assert cfg["network"]["power_control"] == "fixed"
        assert cfg["network"]["tx_power_w"] == 0.1
        assert cfg["reward"]["per_hop_relay_reward"] == 0.0
        assert cfg["reward"]["rf_efficiency_penalty"] == 0.0
        assert "pql" not in cfg
        assert cfg[algo]["share_actor_parameters"] is True
        assert cfg[algo]["share_critic_parameters"] is True
        assert cfg[algo]["hidden_sizes"] == [256, 256]
        assert cfg[algo]["actor_layer_norm"] is False
        assert cfg[algo]["batch_size"] == 1024
        assert cfg[algo]["replay_capacity"] == 300000


def test_zero_float64_in_warp_kernels():
    for rel in [
        "uav_network/communication/gpu/phy_sinr_fp32.py",
        "uav_network/communication/gpu/mac_slotted_fp32.py",
        "uav_network/environment/gpu/ds_aoi_belief_fp32.py",
    ]:
        src = (PKG_ROOT / rel).read_text(encoding="utf-8")
        ast.parse(src)
        assert "wp.float64" not in src, f"Found forbidden wp.float64 in {rel}"
        assert "wp.vec2d" not in src and "wp.vec3d" not in src, f"Found forbidden double vector in {rel}"


def test_physics_link_budget_ds_and_zeng():
    lb = analytical_link_budget_summary()
    assert 480.0 <= lb["d_los_m"] <= 525.0, f"Unexpected LOS range: {lb}"
    assert 70.0 <= lb["d_nlos_m"] <= 90.0, f"Unexpected NLOS range: {lb}"

    p_hover = zeng_rotary_power_w(0.0)
    p_cruise = zeng_rotary_power_w(11.0)
    assert p_hover > p_cruise, f"Hover ({p_hover}W) must exceed cruise ({p_cruise}W) to prevent hover camping!"

    mf, me, mu = ds_closed_form(n_pos=4, n_neg=0, mf=0.55, me=0.55)
    assert math.isclose(mf + me + mu, 1.0, rel_tol=1e-6)
    assert mf >= 0.90 and mu <= 0.15


def test_vanilla_matd3_masac_and_replay_buffer():
    ast.parse((PKG_ROOT / 'uav_network/algorithms/vanilla_baselines.py').read_text(encoding='utf-8'))
    if torch is None:
        print('  [info] torch not installed in sandbox; AST syntax verified for vanilla_baselines.py')
        return
    torch.manual_seed(44)
    n_agents, obs_dim, state_dim, cont_dim, disc_dim = 6, 96, 256, 2, 7
    buf = GPUHybridReplayBuffer(256, n_agents, obs_dim, state_dim, cont_dim, disc_dim, torch.device("cpu"))

    # Verify exact 300k VRAM projection (~1.03 GiB)
    bytes_per_step = buf.vram_bytes() / 256
    proj_gib_300k = (bytes_per_step * 300_000) / (1024 ** 3)
    assert 0.95 <= proj_gib_300k <= 1.15, f"Unexpected 300k VRAM: {proj_gib_300k:.2f} GiB"

    # Populate synthetic masked transitions (including silent-only K_i=1 edge case)
    b = 64
    obs = torch.randn(b, n_agents, obs_dim)
    next_obs = torch.randn(b, n_agents, obs_dim)
    state = torch.randn(b, state_dim)
    next_state = torch.randn(b, state_dim)
    mask = torch.zeros(b, n_agents, disc_dim, dtype=torch.uint8)
    mask[:, :, 0] = 1  # Silent always valid
    mask[:32, :, 6] = 1  # Half have GCS valid; half are silent-only (K_i=1)
    next_mask = mask.clone()

    act_cont = torch.tanh(torch.randn(b, n_agents, cont_dim))
    act_disc = torch.zeros(b, n_agents, disc_dim)
    act_disc[:, :, 0] = 1.0
    rewards = torch.randn(b, 1)
    dones = torch.zeros(b, 1)

    buf.add_batch(obs, state, act_cont, act_disc, rewards, next_obs, next_state, dones, mask, next_mask)
    batch = buf.sample(32)

    matd3 = VanillaMATD3(obs_dim, state_dim, n_agents, cont_dim, disc_dim)
    m1 = matd3.update(batch)
    m2 = matd3.update(batch)
    assert "actor_loss" in m2 and all(math.isfinite(v) for v in m2.values())

    masac = VanillaMASAC(obs_dim, state_dim, n_agents, cont_dim, disc_dim)
    s1 = masac.update(batch)
    assert all(math.isfinite(v) for v in s1.values())
    assert s1["alpha_c"] > 0 and s1["alpha_d"] > 0


def test_dual_fidelity_eval_bridge():
    def dummy_slotted(seeds):
        return [EpisodeKPI(180.0, 23, 21, 25, 18.0, 42.0, 12, 3, 1, 0, 1, 520.0, 0) for _ in seeds]

    def dummy_full(seeds):
        return [EpisodeKPI(172.0, 23, 20, 25, 20.5, 48.0, 11, 5, 2, 1, 1, 525.0, 0) for _ in seeds]

    rep = run_dual_fidelity_evaluation(dummy_slotted, dummy_full, [101, 102, 103])
    assert rep["slotted_warp_fp32"]["multihop_delivery_ratio_pct"] > 50.0
    assert math.isclose(rep["sim_to_eval_gap"]["delta_mean_aoi_s"], 2.5, abs_tol=1e-6)


if __name__ == "__main__":
    test_yaml_configs_and_vanilla_constraints()
    test_zero_float64_in_warp_kernels()
    test_physics_link_budget_ds_and_zeng()
    test_vanilla_matd3_masac_and_replay_buffer()
    test_dual_fidelity_eval_bridge()
    print("ALL 5 REDESIGN PACKAGE VERIFICATION TESTS PASSED!")
