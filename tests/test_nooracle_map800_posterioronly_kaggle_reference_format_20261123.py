"""User requested original therwed notebook5051433b29 format: MD+CONFIG+SOURCE FULL FLOW."""
import ast,base64,hashlib,io,json,re,tarfile
from pathlib import Path
import nbformat
from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import (
    build_notebook,OUT,CONFIG,SOURCE_REF)


def test_three_cell_reference_format_real_run_not_bounded_pilot():
    n=build_notebook()
    nbformat.validate(n)
    assert len(n.cells)==3
    assert [cell.cell_type for cell in n.cells]==['markdown','code','code']
    assert 'Posterior' in n.cells[0].source and 'LR' in n.cells[0].source
    assert 'Posterior ≥ 0,99' in n.cells[0].source
    assert '0,8 W' in n.cells[0].source
    assert 'TRAIN_EPISODES = 20000' in n.cells[1].source
    assert 'CHECKPOINTS_PER_GPU_BATCH = 81' in n.cells[1].source
    assert 'MAX_WARP_EVAL_WORLDS_PER_GPU = 8192' in n.cells[1].source
    assert 'FINAL_TEST_EPISODES = 1000' in n.cells[1].source
    assert 'VALIDATION_EPISODES = 100' in n.cells[1].source
    assert 'UserSecretsClient' in n.cells[1].source
    assert 'NOTEBOOK_REF' in n.cells[1].source
    assert 'torch.cuda.device_count()' in n.cells[1].source
    assert n.metadata['full_20000_training'] is True
    assert n.metadata['validated_checkpoint_bank']=='all'
    assert n.metadata['num_envs']==4096
    assert n.metadata['actor_critic_hidden_sizes']==[256,256]
    assert n.metadata['confirmation_mode']=='posterior_only'
    assert n.metadata['likelihood_ratio_gate']==False
    assert n.metadata['tx_power_max_w']==.8
    assert n.metadata['ref_notebook']=='therwed/notebook5051433b29'
    assert OUT.name.endswith('.ipynb')
    # Kaggle refuses kernel commits at >=1 MB of notebook source. Leave margin.
    assert len(nbformat.writes(n).encode('utf-8'))<850_000
    assert len(n.cells[2].source.encode('utf-8'))<825_000
    ast.parse(n.cells[1].source)
    ast.parse(n.cells[2].source)


def test_bootstrap_full_flow_real_batch_and_hf_deep_profile():
    code=build_notebook().cells[2].source
    assert 'one_session_train_eval_test' in code
    assert 'validate_all_snapshots' in code
    assert 'contiguous_adaptive' in code
    assert 'str(CHECKPOINTS_PER_GPU_BATCH)' in code
    assert 'str(MAX_WARP_EVAL_WORLDS_PER_GPU)' in code
    assert 'str(FINAL_TEST_EPISODES // 2)' in code
    assert 'UAV_ROOT_CAUSE_AUDIT' in code
    assert 'UAV_FORENSIC_ENABLED' in code
    assert 'UAV_FORENSIC_WORLD_IDS' in code
    assert 'UAV_DIAGNOSTIC_STORE' in code
    assert 'UAV_DIAGNOSTIC_HF_REPO' in code
    assert 'UAV_WANDB_PLOTS_ONLY' in code
    assert 'UAV_HF_VERIFY_PUBLIC_READBACK' in code
    assert 'UAV_DEFERRED_EXPORT_HF' in code
    assert 'WANDB_API_KEY' in code and 'HF_TOKEN' in code
    assert 'one_session' in code
    assert 'NO_CBF' in code
    assert "'gpu_eval_candidate_limit': 0" in code
    assert 'MATD3_POSTERIORONLY_TRAIN_VAL100_TEST1000_COMPLETE' in code
    assert 'training_only' in code
    assert 'require_full_episode_budget' in code


def test_embedded_source_is_sha_verified_no_password_and_correct_new_science():
    n=build_notebook()
    text=n.cells[2].source
    m=re.search(r'payload = base64\.b85decode\(("[^"]+")\)',text)
    assert m, 'Missing self-contained source compressed archive'
    compressed=base64.b85decode(ast.literal_eval(m.group(1)))
    assert hashlib.sha256(compressed).hexdigest()==n.metadata['source_sha256']
    with tarfile.open(fileobj=io.BytesIO(compressed),mode='r:gz') as tar:
        files={f.name:tar.extractfile(f).read() for f in tar if f.isfile()}
    assert 280<=len(files)<=320, 'Bundle only runtime sources, not test suites or old experiments'
    assert not any(n.startswith('tests/') for n in files)
    assert [n for n in files if n.startswith('configs/')]==[CONFIG]
    assert 'scripts/requirements-network.txt' in files
    assert 'scripts/upstream-revision.txt' in files
    assert 'third_party/UavNetSim/simulator/simulator.py' in files
    assert 'uav_network/experiments/one_session.py' in files
    assert 'uav_network/experiments/deferred_gpu_evaluation.py' in files
    assert 'uav_network/experiments/training.py' in files
    assert 'key.txt' not in files and '.env' not in files
    assert not any(name.endswith(('kaggle.json','credentials.json','.pem','.key')) for name in files)
    new=files[CONFIG].decode()
    assert 'confirmation_mode: posterior_only' in new
    assert 'confirmation_min_likelihood_ratio: 0.0' in new
    assert 'tx_power_max_w: 0.8' in new
    assert 'compile_models: true' in new
    assert 'hidden_sizes:\n  - 256\n  - 256' in new
    assert 'return 11  # No LR/50m gate' in files['uav_network/environment/observations.py'].decode()
    assert 'if p.confirmation_mode==2:' in files['uav_network/environment/gpu/bayesian.py'].decode()
