"""Val100/Test4000 raw records must be published to HF, not just W&B plots."""
import json,tarfile
from pathlib import Path
from uav_network.experiments.post_eval_hf_forensics import PostEvalHFForensics


class FakeHF:
    def __init__(self):self.uploaded=[]
    def save(self,path,*,base_path,policy):
        assert policy=='now'
        self.uploaded.append(Path(path))
        return 'runs/abcd1234/post-eval-forensics/'+Path(path).name


def test_post_eval_incremental_archive_contains_raw_validation_test_and_status(tmp_path):
    base=tmp_path/'gpu_batch_eval'
    base.mkdir()
    hf=FakeHF()
    audit=PostEvalHFForensics(base,hf,interval_seconds=0)
    (base/'eval_0000.json').write_text(json.dumps({'seed_first':2000000044,'records':[{'seed':2000000044,'search_rate_percent':40.}]}))
    (base/'eval_0001.json').write_text('{"incomplete":')
    audit.tick()
    assert len(hf.uploaded)==1
    with tarfile.open(hf.uploaded[0],'r:gz') as t:
        names=t.getnames()
        assert 'eval_0000.json' in names
        assert 'eval_0001.json' not in names
        assert 'manifest.json' in names
    (base/'eval_0001.json').write_text(json.dumps({'seed_first':2000000044,'records':[{'seed':2000000044}]}))
    (base/'final_test.json').write_text(json.dumps({'episodes':4000,'delivery_rate_percent':31.}))
    (base/'report.json').write_text(json.dumps({'validation_completed':2,'best_global_step':4096}))
    audit.tick(final=True)
    assert len(hf.uploaded)==2
    with tarfile.open(hf.uploaded[-1],'r:gz') as t:
        names=t.getnames()
        assert 'eval_0001.json' in names
        assert 'final_test.json' in names
        assert 'report.json' in names
    assert audit.published_count==4


def test_failed_hf_upload_preserves_pending_archive_for_retry(tmp_path):
    class Failing:
        attempts=0
        def save(self,*args,**kwargs):
            self.attempts+=1
            if self.attempts==1:raise OSError('test offline')
            return 'runs/abcd/part.tar.gz'
    base=tmp_path/'gpu_batch_eval';base.mkdir()
    (base/'eval_0000.json').write_text(json.dumps({'episodes':100}))
    hf=Failing();audit=PostEvalHFForensics(base,hf,interval_seconds=0)
    assert audit.tick()==False
    assert audit.published_count==0
    assert audit.failed_uploads==1
    assert audit.tick(final=True)==True
    assert audit.published_count==1
    assert hf.attempts==2
