"""GPU Val100×N and Test4000 evidence may exceed the raw tar part cap."""
import hashlib
import io
import json
import tarfile

from uav_network.experiments.post_eval_hf_forensics import PostEvalHFForensics


class UploadCollector:
    def __init__(self):self.parts=[]
    def save(self,path,*,base_path,policy):
        assert policy=='now'
        with tarfile.open(path,'r:gz') as t:
            manifest=json.load(t.extractfile('manifest.json'))
            content={name:t.extractfile(name).read() for name in t.getnames()
                     if name!='manifest.json'}
        self.parts.append((manifest,content))


def test_overlarge_final_test_and_validation_results_are_bounded_lossless(tmp_path):
    root=tmp_path/'gpu_batch_eval';root.mkdir()
    big=json.dumps({'episodes':4000,'records':[
        {'seed':i,'value':'x'*300} for i in range(80)]},sort_keys=True).encode()
    (root/'final_test.json').write_bytes(big)
    other=json.dumps({'validation_completed':134,'records':[{'policy':i,'scores':'y'*100}
                          for i in range(60)]},sort_keys=True).encode()
    (root/'validation_results.json').write_bytes(other)
    uploaded=UploadCollector()
    tracker=PostEvalHFForensics(root,uploaded,interval_seconds=0,max_raw_bytes=1024)
    assert tracker.tick(final=True)
    assert tracker.published_count==2
    assert len(uploaded.parts)>2
    reconstructed={}
    for manifest,members in uploaded.parts:
        assert sum(x['bytes'] for x in manifest['files'].values())<=1024
        for name,spec in manifest['files'].items():
            raw=members[name]
            assert len(raw)==spec['bytes']
            assert hashlib.sha256(raw).hexdigest()==spec['sha256']
            source=spec.get('source_file',name)
            reconstructed.setdefault(source,{})[spec.get('offset_start',0)]=raw
    for name,expected in [('final_test.json',big),('validation_results.json',other)]:
        result=b''.join(data for _,data in sorted(reconstructed[name].items()))
        assert result==expected


def test_retry_after_midway_large_file_does_not_skip_or_duplicate_bytes(tmp_path):
    root=tmp_path/'gpu_batch_eval';root.mkdir()
    expected=json.dumps({'episodes':4000,'records':['u'*500 for _ in range(12)]}).encode()
    (root/'final_test.json').write_bytes(expected)
    class FailingCollector(UploadCollector):
        calls=0
        def save(self,*args,**kwargs):
            self.calls+=1
            if self.calls==2:raise ConnectionError('HF 429 preflight')
            return super().save(*args,**kwargs)
    hf=FailingCollector()
    archive=PostEvalHFForensics(root,hf,interval_seconds=0,max_raw_bytes=1024)
    assert archive.tick(final=True)
    assert archive.published_count==0
    assert archive.pending is not None
    assert archive.failed_uploads==1
    assert archive.tick(final=True)
    assert archive.published_count==1
    pieces={}
    for manifest,members in hf.parts:
        for name,spec in manifest['files'].items():
            pieces[spec.get('offset_start',0)]=members[name]
    assert b''.join(x for _,x in sorted(pieces.items()))==expected


def test_mutated_large_source_after_first_range_is_rejected(tmp_path):
    """Never mix historical and replaced JSON bytes in a single HF reconstruction."""
    import pytest
    folder=tmp_path/'gpu_batch_eval'
    folder.mkdir()
    source=folder/'validation_results.json'
    source.write_text(json.dumps({'results':[{'return':0,'x':'z'*800} for _ in range(10)]}))
    hf=UploadCollector()
    tracker=PostEvalHFForensics(folder,hf,interval_seconds=0,max_raw_bytes=1024)
    assert tracker.tick()
    assert tracker.published_count==0
    # Append harmless whitespace: still valid JSON, but the bytes changed.
    with source.open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='mutated after first range'):
        tracker.tick(final=True)
    assert tracker.published_count==0
