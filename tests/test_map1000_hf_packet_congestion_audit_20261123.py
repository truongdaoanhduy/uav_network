"""HF congestion audit must not mislabel absent pre-NoOracle metrics as zeros."""
import pytest
import scripts.audit_hf_packet_congestion_20261123 as mod


def test_stats_sample_and_missing_without_fake_zero():
    rows=[{'network':{'generated_packets':2,'collisions':0},'targets_delivered':1},
          {'network':{'generated_packets':6,'collisions':3},'targets_delivered':2},
          {'network':{'generated_packets':10,'collisions':9},'targets_delivered':3}]
    assert mod.stats(rows,'generated_packets','network')['mean']==6
    assert mod.stats(rows,'collisions','network')['mean']==4
    assert mod.stats(rows,'targets_delivered')['median']==2
    assert mod.stats(rows,'false_confirmed_cells') is None


def test_audit_is_reproducible_and_retains_missing_fields(monkeypatch):
    evidence={'old':[{'network':{'generated_packets':3,'collisions':0}}]*3,
              'new':[{'network':{'generated_packets':15,'collisions':5},
                      'false_confirmed_cells':8}]*3}
    monkeypatch.setattr(mod,'load_episode_part',lambda run,part:evidence[run])
    rows=mod.audit(runs=('old','new'),parts=(25,),min_count=2)
    assert len(rows)==2
    assert rows[0]['generated_packets_mean']==3
    assert 'false_confirmed_cells_mean' not in rows[0]
    assert rows[1]['generated_packets_mean']==15
    assert rows[1]['false_confirmed_cells_mean']==8
    assert rows[0]['source'].endswith('/diagnostics-000025.tar.gz')
