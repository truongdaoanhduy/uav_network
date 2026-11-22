"""The first Warp/JIT job must never masquerade as a 47% speedup."""
from scripts.analyze_matd3_ab_cold_start_20261123 import analyze


def test_actual_two_pair_shape_does_not_promote_startup_cost_to_speedup():
    template=dict(science_sha256='same',model_sha256='same',episode_sha256='empty',
                  completed_episodes=0,actual_learner_updates=132)
    runs=[dict(template,variant=name,train_joint_transitions_per_second=tps)
          for name,tps in [('baseline',2140.3227),('candidate',6029.2511),
                           ('candidate',6062.4428),('baseline',6054.5347)]]
    result=analyze(runs)
    assert result['raw_all_runs_speedup_percent']>40
    assert abs(result['steady_state_candidate_speedup_percent'])<1
    assert result['first_run_startup_excluded']
    assert result['episode_records_informative'] is False
    assert result['candidate_verified_beneficial'] is False
    assert result['warm_run_counts']=={'baseline':1,'candidate':2}


def test_future_kaggle_training_notebook_stops_promoting_cold_start_as_speedup():
    from scripts.build_nooracle_gpu_train_speed_ab_20261123 import build_notebook
    code=build_notebook().cells[3].source
    assert 'warm_runs=runs[1:]' in code
    assert 'episode_records_informative' in code
    assert 'first_run_startup_excluded' in code
    assert 'raw_all_runs_speedup_percent' in code
    assert 'min(warm_counts.values())>=2' in code
