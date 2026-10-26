"""Republishing saved media keeps its policy protocol, including unknown history."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_tracking import OnlineRun
from uav_network.cloud import visualize


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only host")
@pytest.mark.parametrize("version", [None,2,3])
@pytest.mark.parametrize("generation", [None,1,2])
def test_publication_recovery_uses_saved_observation_provenance(tmp_path,version,generation):
    run = OnlineRun()
    initialized = []
    def initialize(**kwargs):
        initialized.append(kwargs)
        return run
    sdk = SimpleNamespace(init=initialize,Settings=lambda **kwargs:kwargs)
    spec = {'entity':'owner','project':'project','account':'selected',
            'config':'configs/kaggle-smoke.yaml','overrides':[],
            'kernel_ref':'owner/cpu-recovery','training_mode':'base',
            'publication_source_ref':'owner/old-cpu',
            'runs':{'matd3':{'id':'cpu1','name':'CPU recovery'}},
            'artifacts':{'matd3':'owner/project/model:v1'},
            'training_status':{'matd3':{'episodes':20000,'global_step':12345}},
            'cpu_eval':{'episodes':3,'seed':10044,'frame_stride':2,'fps':5}}
    def load_assets(child,algorithm,output):
        folder = Path(output)/'visualize'/algorithm
        folder.mkdir(parents=True)
        scenario = {} if version is None else {'observation_version':version}
        if generation is not None:
            scenario['world'] = {'generation_version':generation}
        result = {'algorithm':algorithm,'episodes':3,'device':'cpu','records':[],
                  'render_path':str(folder/'final_episode.png'),
                  'animation_path':str(folder/'final_episode.mp4'),
                  'mean_return':0.,'success_rate':0.,'mean_search_rate_percent':0.,
                  'mean_delivery_rate_percent':0.,
                  'protocol':{'scenario':scenario,'training_global_step':12345}}
        (folder/'final_episode.png').write_bytes(b'image')
        (folder/'final_episode.mp4').write_bytes(b'video')
        (folder/'metrics.json').write_text(json.dumps(result))
        return result
    with patch.dict(sys.modules,{'wandb':sdk}), \
         patch('uav_network.experiments.tracking._authentication',return_value=('owner','project')), \
         patch('uav_network.experiments.dashboard.ensure_project_workspace',return_value='https://wandb.ai/owner/project/workspace'), \
         patch.object(visualize,'load_publication_assets',side_effect=load_assets), \
         patch.object(visualize,'evaluate',side_effect=AssertionError('historical media must not be re-evaluated')), \
         patch.object(visualize,'publish_visualization_media',return_value={'wandb':{'status':'ok'}}):
        result = visualize.run_visualization(spec,tmp_path/'cpu')
    assert initialized[0]['config']['observation_version'] is None
    assert initialized[0]['config']['world']['generation_version'] is None
    assert run.summary.get('evaluation/generation_version') == generation
    assert run.summary.get('evaluation/observation_version') == version
    assert run.summary.get('evaluation/checkpoint_global_step') == 12345
    assert result['matd3']['protocol']['scenario'].get('observation_version') == version
