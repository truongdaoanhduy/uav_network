"""Build pinned 3-cell MAP1000×1000, TX<=.6 W FASTGRID Kaggle fullflow.

Inherit audited reference builder and exact source archive workflow. Scientific
changes are strictly world.map_size_m, sensor.grid_cell_m (keep 40×40 belief),
and network.tx_power_max_w. Keep same deep forensics, 100 seed/checkpoint,
Test1000, strict CUDA T4x2, and all speed flags. No existing config overwritten.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import hashlib
import nbformat

import scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 as base
import scripts.build_nooracle_map800_speed_fullflow_ui_20261123 as fast

ROOT=Path(__file__).resolve().parents[1]
CONFIG='configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml'
NAME='MATD3-NOORACLE-POSTERIORONLY-MAP1000x1000-TX06-FASTGRID-T4X2-E4096-U4-B8192-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME


def replace_once(source,old,new):
    count=source.count(old)
    if count!=1:
        raise ValueError(f'Unexpected notebook template: {old!r} count={count}; refuse stale science')
    return source.replace(old,new,1)


@contextmanager
def map1000_tx06_builder_context():
    """Restore imported baseline globals even on a failed build/test."""
    old_config=base.CONFIG
    old_spec=base.make_spec
    try:
        base.CONFIG=CONFIG
        def new_spec(commit):
            spec=old_spec(commit)
            spec['config']=CONFIG
            spec['kernel_ref']='kaggle-ui/imported-posterior-only-map1000-tx06-fastgrid'
            for before,after in (
                ('world.map_size_m=800','world.map_size_m=1000'),
                ('sensor.grid_cell_m=20','sensor.grid_cell_m=25'),
                ('network.tx_power_max_w=0.8','network.tx_power_max_w=0.6')):
                assert spec['overrides'].count(before)==1
                spec['overrides'][spec['overrides'].index(before)]=after
            spec['runs']['matd3']['name']=spec['runs']['matd3']['name'].replace(
                'MAP800-TX08','MAP1000-TX06')
            assert 'MAP1000-TX06' in spec['runs']['matd3']['name']
            return spec
        base.make_spec=new_spec
        yield
    finally:
        base.CONFIG=old_config
        base.make_spec=old_spec


def build_notebook():
    from uav_network.configuration import load_config
    c=load_config(CONFIG)
    c.validate()
    assert c.world.map_size_m==1000 and c.sensor.grid_cell_m==25
    assert c.network.tx_power_max_w==.6 and c.network.tx_power_min_w==.001
    assert c.sensor.confirmation_mode=='posterior_only' and c.matd3.batch_size==8192
    with map1000_tx06_builder_context():
        notebook=fast.build_speed_notebook()
    intro=notebook.cells[0].source
    for old,new in (
        ('MAP800','MAP1000'),
        ('TX0,8 W','TX0,6 W'),
        ('800×800 m','1.000×1.000 m'),
        ('20m/ô','25m/ô'),
        ('0,001–0,8 W','0,001–0,6 W'),
        ('TX0.8W','TX0.6W'),
    ):
        assert old in intro,(old,'intro missing expected science label')
        intro=intro.replace(old,new)
    notebook.cells[0].source=intro
    code=notebook.cells[1].source
    code=replace_once(code,'TX_POWER_MAX_W = .8','TX_POWER_MAX_W = .6')
    code=replace_once(code,'POSTERIOR_THRESHOLD = .99',
         'POSTERIOR_THRESHOLD = .99\nMAP_SIZE_M = 1000\nGRID_CELL_M = 25')
    code=replace_once(code,'TX<=0.8W','TX<=0.6W')
    code=replace_once(code,'assert VALIDATION_EPISODES==100 and FINAL_TEST_EPISODES==1000',
        'assert VALIDATION_EPISODES==100 and FINAL_TEST_EPISODES==1000\n'
        'assert MAP_SIZE_M==1000 and GRID_CELL_M==25 and TX_POWER_MAX_W==.6')
    notebook.cells[1].source=code
    src=notebook.cells[2].source
    # These are executable host script checks; edits do NOT touch encoded source archive.
    for old,new in (
        ("'tx_power_cap_w':.8","'tx_power_cap_w':.6"),
        ('assert c.sensor.confirmation_threshold==.99 and c.network.tx_power_max_w==.8;',
         'assert c.sensor.confirmation_threshold==.99 and c.network.tx_power_max_w==.6;'),
        ('assert c.world.map_size_m==800 and c.sensor.grid_cell_m==20;',
         'assert c.world.map_size_m==1000 and c.sensor.grid_cell_m==25;'),
        ('MATD3-POSTERIORONLY-MAP800-TX08','MATD3-POSTERIORONLY-MAP1000-TX06'),
        ('POSTERIORONLY_TX08_FULLFLOW_COMPLETE','POSTERIORONLY_TX06_FULLFLOW_COMPLETE'),
    ):
        src=replace_once(src,old,new)
    notebook.cells[2].source=src
    notebook.metadata.update({
        'science_config':CONFIG,
        'tx_power_max_w':.6,
        'map_size_m':1000,
        'map_dimensions_m':[1000,1000],
        'grid_cell_m':25,
        'belief_grid_cells_per_side':40,
        'speed_profile':'fastgrid_periodic_only_batch_scalars_fast_grad_clip_map1000_tx06',
    })
    # The archive digest is sealed by the shared source builder; changing it
    # here would invalidate source integrity. The preflight tests unpack it.
    nbformat.validate(notebook)
    if len(notebook.cells)!=3 or len(nbformat.writes(notebook).encode())>=850_000:
        raise ValueError('Invalid oversized 3-cell Kaggle notebook')
    return notebook


def main():
    n=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    print('MAP1000_TX06_FASTGRID_NOTEBOOK',OUT,'BYTES',OUT.stat().st_size,
          'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),
          'EMBEDDED_SOURCE_SHA256',n.metadata['source_sha256'],
          'GIT_COMMIT',n.metadata['reference_commit'])


if __name__=='__main__':main()
