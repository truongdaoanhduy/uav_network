"""W&B display names must stay compact while retaining run identity and study dimensions."""
from itertools import product
import re

from uav_network.configuration import load_config
from uav_network.cloud.identities import run_name, authoritative_tags, replay_layout


def test_compact_names_unique_for_cohort_and_contain_seed_layout_mode():
    c = load_config('configs/comparison-reference-masac-matd3-xyz-sensing.yaml')
    seen = set()
    for algorithm, mode, shared, vectorized in product(('masac', 'matd3'), ('base', 'pql'), (True, False), (False, True)):
        cfg = getattr(c, algorithm)
        cfg.share_actor_parameters = shared
        cfg.share_critic_parameters = shared
        c.runtime.vectorize_critics = vectorized
        suffix = f'{len(seen):08x}'
        name = run_name(c, algorithm, mode, 'train', suffix)
        assert name.startswith(f'{algorithm.upper()}-{mode.upper()}-')
        assert ('-SH-' if shared else '-IND-') in name
        assert ('-VEC-' in name) == vectorized
        assert name.endswith('-' + suffix)
        assert len(name) <= 38
        assert re.fullmatch('[A-Z0-9a-f-]+', name)
        assert name not in seen
        seen.add(name)
    assert len(seen) == 16


def test_actor_only_and_critic_only_names_are_unambiguous():
    c = load_config('configs/comparison-20k-hybrid75-1kb.yaml')
    c.masac.share_actor_parameters = True
    c.masac.share_critic_parameters = False
    assert '-AS-' in run_name(c, 'masac', 'base', 'train', 'a1b2c3d4')
    c.masac.share_actor_parameters = False
    c.masac.share_critic_parameters = True
    assert '-CS-' in run_name(c, 'masac', 'pql', 'train', 'a1b2c3d4')


def test_nontrain_names_short_distinct_and_timestamp_elided():
    c = load_config('configs/comparison-20k-hybrid75-1kb.yaml')
    train = run_name(c, 'masac', 'base', 'train', '20261106-100000-aabbcc')
    validate = run_name(c, 'masac', 'base', 'validate', 'aabbcc')
    benchmark = run_name(c, 'masac', 'base', 'benchmark', 'aabbcc')
    assert train.endswith('-aabbcc')
    assert '20261106' not in train
    assert validate.startswith('VAL-MASAC-BASE-')
    assert benchmark.startswith('BENCH-MASAC-BASE-')
    assert len(benchmark) <= 38


def test_technical_details_stay_in_tags_and_config_not_title():
    c = load_config('configs/comparison-reference-masac-matd3-xyz-sensing.yaml')
    n = run_name(c, 'matd3', 'base', 'train', '11112222')
    assert all(item not in n for item in ('XYZ', 'BAYES', 'HYBRID', 'E4096', 'B8192', 'R1M'))
    tags = authoritative_tags(c, 'matd3')
    assert 'replay-' + replay_layout(c).lower() in tags
    assert 'actor-shared' in tags and 'critic-shared' in tags


def test_cpu_and_controller_names_do_not_duplicate_parent_id():
    from uav_network.cloud.identities import related_run_name
    root = 'MATD3-BASE-SH-VEC-S44-a1b2c3d4'
    cpu = related_run_name(root, 'EVALCPU', 'e5f6a7b8')
    ctrl = related_run_name(root, 'CTRL', 'C1')
    assert cpu == 'CPU-MATD3-BASE-SH-VEC-S44-e5f6a7b8'
    assert ctrl == 'CTRL-MATD3-BASE-SH-VEC-S44-C1'
    assert 'a1b2c3d4' not in cpu and 'a1b2c3d4' not in ctrl
    assert related_run_name('MASAC-base-seed44-root', 'EVALCPU', 'abc') == 'EVALCPU-MASAC-base-seed44-root-abc'
