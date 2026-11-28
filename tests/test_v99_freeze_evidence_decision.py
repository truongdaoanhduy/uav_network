from scripts.maintenance.v99_freeze_evidence_decision import evaluate


def test_empty_reports_fail_closed():
    d=evaluate({}, {}, {})
    assert not d['scenario_frozen']
    assert not d['phase2_matd3_masac_training_permitted']
    assert all(not x['pass'] for x in d['gates'].values())


def test_three_self_reported_passes_still_fail():
    d=evaluate({'status':'PASS'}, {'status':'PASS','envs':4096,'steps':900,'device':'cuda:0','max_reward_sum_err':0}, {'status':'PASS','seeds':[44,2027,10044]})
    assert not d['scenario_frozen']
    assert 'upstream' in ' '.join(d['gates']['S12']['blocking_reasons'])


def test_real_v98_endurance_lacks_all_step_accountability():
    d=evaluate({'status':'FAIL'}, {'status':'PASS','envs':4096,'steps':900,'device':'cuda:0','max_reward_sum_err':0,'torch_peak_vram_mib':0,'safety_failed_worlds':0}, {'status':'PASS'})
    assert not d['gates']['ENDURANCE']['pass']
    assert any('all-step' in s for s in d['gates']['ENDURANCE']['blocking_reasons'])


def test_exact_zero_reward_ledger_not_flagged_as_failure():
    d = evaluate({}, {"status":"PASS","device":"cuda:0","envs":4096,"steps":900,"max_reward_sum_err":0.0}, {})
    assert not any('reward ledger failed' in s for s in d['gates']['ENDURANCE']['blocking_reasons'])


def test_missing_ablation_branch_fail_closed():
    d=evaluate({}, {}, {'status':'PASS', 'seeds':[1,2,3,4,5], 'paired_world_analysis': True, 'peer_action_necessity_confirmed': True, 'coverage_term_matched_only_acknowledged': True, 'battery_energy_accounting_verified': True, 'results': {}})
    assert any('missing required ablation case' in s for s in d['gates']['ABLATION']['blocking_reasons'])
