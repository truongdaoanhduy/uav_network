#!/usr/bin/env python3
"""Export effective scenarios and deterministic audit figures, never training claims."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/zhang_xue/src')]
from zhang_xue.belief.dempster_shafer import (
    counts_to_canonical_ds_effective, sensor_likelihood_to_bba,
)
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.mobility.flight_safety_energy import zeng_propulsion_power_w

PROFILES = [('v67', '1km'), ('v67', '2km'), ('v70', '3km')]


def inventory(output, seed=44, plot=False):
    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / 'effective_configs'
    config_dir.mkdir(exist_ok=True)
    rows, environments = [], []
    for version, size in PROFILES:
        for peer in ('datamule', 'no_peer'):
            name = f'scenario_{version}_{peer}_{size}'
            cfg = load_scenario(ROOT / 'research/zhang_xue/configs' / (name + '.yaml'))
            import yaml
            (config_dir / (name + '.yaml')).write_text(yaml.safe_dump(asdict(cfg), sort_keys=True))
            if peer == 'no_peer':
                continue
            env = ZhangXue7NodePacketEnv(cfg, seed=seed)
            obs, _, _ = env.reset(seed=seed)
            environments.append(env)
            distances = [float(np.linalg.norm(env.positions[i, :2] - env.positions[j, :2]))
                         for i in range(cfg.num_uavs) for j in range(i)]
            rows.append({'profile':name, 'map_size_m':cfg.map_size_m, 'grid_w':cfg.grid_w,
                'targets':cfg.num_targets, 'target_density_per_km2':cfg.num_targets / (cfg.map_size_m / 1000)**2,
                'uavs_per_km2':cfg.num_uavs / (cfg.map_size_m / 1000)**2,
                'obs_shape':list(obs.shape), 'buildings':len(env.buildings),
                'minimum_spawn_spacing_m':min(distances),
                'hover_power_with_avionics_w':zeng_propulsion_power_w(0, env.batt_cfg) + env.batt_cfg.avionics_w,
                'ideal_hover_endurance_no_rf_s':cfg.e_max_j / (zeng_propulsion_power_w(0, env.batt_cfg) + env.batt_cfg.avionics_w),
                'maximum_speed_power_with_avionics_w':zeng_propulsion_power_w(cfg.v_max_mps, env.batt_cfg) + env.batt_cfg.avionics_w,
                'unoccluded_fov_area_fraction_per_uav':math.pi * cfg.fov_radius_m**2 / cfg.map_size_m**2,
                'note':'Geometry/reset audit only; neither rollout throughput nor mission performance'})
    mf, me = sensor_likelihood_to_bba(.85, .03)
    report = {'seed':seed, 'profiles':rows,
        'calibration_example':{'Pd':.85, 'Pf':.03, 'likelihood_ratio_mf':mf,
            'likelihood_ratio_me':me, 'single_positive_exceeds_0_90':mf >= .90,
            'single_positive_exceeds_0_99':mf >= .99,
            'note':'BBA support is not calibrated posterior probability'},
        'aging_counterexample':{'mf':.8, 'weight':.5,
            'count_power_discount_target_mass':counts_to_canonical_ds_effective(.5, 0)[0],
            'mass_discount_target_mass':.5 * .8,
            'equivalent':False},
        'cuda_executed':False, 'full_rl_training':False}
    (output / 'scientific_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    if plot:
        render_maps(environments, output / 'figures/scenario_maps.png')
    return report


def render_maps(environments, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from zhang_xue.communication.common import is_in_blackout

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ax, env in zip(axes, environments):
        cfg = env.cfg
        blackout = np.zeros((env.grid_w, env.grid_w))
        for y in range(env.grid_w):
            for x in range(env.grid_w):
                blackout[y, x] = is_in_blackout(((x+.5)*50, (y+.5)*50),
                    env.blackout_centers, env.blackout_r, env.grid_w, 50)
        ax.imshow(blackout, origin='lower', extent=(0,cfg.map_size_m,0,cfg.map_size_m),
                  cmap='Greens', vmin=0, vmax=3, alpha=.55, interpolation='nearest')
        for bx, by, radius, height in env.buildings:
            ax.add_patch(Circle((bx,by), radius, color='#525c6a', alpha=.85))
        ids = np.flatnonzero(env.true_targets)
        ax.scatter((ids % env.grid_w + .5)*50, (ids // env.grid_w + .5)*50,
                   s=12, marker='x', color='#c74938', label='Target truth (audit only)')
        ax.scatter(*env.positions[:env.actor_n,:2].T, s=20, color='#2066a8', label='UAV')
        for x, y in env.positions[:env.actor_n,:2]:
            ax.add_patch(Circle((x,y), cfg.fov_radius_m, fill=False, color='#2066a8', alpha=.4))
        ax.scatter(*env.positions[env.gcs_id,:2], marker='*', s=75, color='#d19b00', label='GCS')
        ax.set(xlim=(0,cfg.map_size_m), ylim=(0,cfg.map_size_m), aspect='equal',
               xlabel='x (m)', ylabel='y (m)', title=f'{cfg.map_size_m/1000:g} km | {cfg.num_targets} targets | 6 UAV')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Same target density; different travel, FOV and radio workload. No trained policy shown.')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/qa_evidence_v72'))
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to((ROOT / 'outputs').resolve()):
        parser.error('Generated audit files must be under outputs/')
    print(json.dumps(inventory(output, args.seed, args.plot), indent=2))


if __name__ == '__main__':
    main()
