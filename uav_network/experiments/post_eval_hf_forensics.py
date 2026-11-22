"""Immutable public HF evidence for GPU Val100/Final Test4000, beyond W&B plots.

No oracle or policy execution depends on this module. Completed atomic files
are archived independently of the worker process; a partial JSON is skipped.
Publication acknowledges a file only after verified public HF upload.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


class PostEvalHFForensics:
    def __init__(self,output,uploader,*,interval_seconds=180,max_raw_bytes=8*1024*1024,now=None):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.folder=self.output/'post-eval-forensics'
        self.folder.mkdir(exist_ok=True)
        self.uploader=uploader
        self.interval=float(interval_seconds)
        self.max_raw_bytes=int(max_raw_bytes)
        self.now=now or time.monotonic
        self.last_tick=self.now()
        self.published={}
        self.pending=None
        self.parts_created=0
        self.failed_uploads=0
        self.last_error=None
        if self.interval<0 or self.max_raw_bytes<1024:
            raise ValueError('Invalid post-evaluation archive threshold')

    @property
    def published_count(self):
        return len(self.published)

    @staticmethod
    def _supported(name):
        return (bool(re.fullmatch(r'eval_\d+\.json',name))
                or name in {'final_test.json','report.json','validation_results.json',
                            'validation_results.csv','best.json','one-session-status-start.json',
                            'one-session-status-final.json','one-session-status-error.json'}
                or bool(re.fullmatch(r'[\w.-]+\.jsonl',name)))

    def _ready(self):
        files=[]
        for p in sorted(self.output.iterdir()):
            if (not p.is_file() or p.is_symlink() or not self._supported(p.name)
                    or p.name in self.published):
                continue
            if p.suffix in {'.json','.jsonl'}:
                try:
                    with p.open('r') as stream:
                        if p.suffix=='.json':json.load(stream)
                        else:
                            for line in stream:
                                if line.strip():json.loads(line)
                except (OSError,ValueError,UnicodeDecodeError):
                    # Worker is still writing, or its output is not valid.
                    continue
            files.append(p)
        return files

    def _create(self,files):
        chosen=[]
        size=0
        for path in files:
            n=path.stat().st_size
            if chosen and size+n>self.max_raw_bytes:break
            chosen.append(path)
            size+=n
        if not chosen:return None
        index=self.parts_created+1
        part=self.folder/f'part-{index:06d}.tar.gz'
        tmp=self.folder/f'part-{index:06d}.tmp'
        manifest={'schema':1,'scope':'raw_gpu_validation_final_test_evidence',
                  'files':{p.name:{'bytes':p.stat().st_size,'sha256':_digest(p)} for p in chosen}}
        serialized=json.dumps(manifest,sort_keys=True).encode()
        try:
            with tarfile.open(tmp,'w:gz',compresslevel=6) as tar:
                for p in chosen:
                    info=tarfile.TarInfo(p.name)
                    info.size=p.stat().st_size;info.mtime=0
                    with p.open('rb') as stream:tar.addfile(info,stream)
                info=tarfile.TarInfo('manifest.json')
                info.size=len(serialized);info.mtime=0
                tar.addfile(info,io.BytesIO(serialized))
            with tarfile.open(tmp,'r:gz') as verified:
                if set(verified.getnames())!={*manifest['files'],'manifest.json'}:
                    raise IOError('Incomplete GPU evidence archive')
            tmp.replace(part)
        finally:
            tmp.unlink(missing_ok=True)
        self.parts_created+=1
        self.pending=(part,manifest)
        return part

    def tick(self,*,final=False):
        now=self.now()
        if self.pending is None:
            # A cheap stat-only check avoids parsing hundreds of 100-seed JSON
            # files every three seconds while two GPUs evaluate checkpoints.
            if not final and self.interval>0 and now-self.last_tick<self.interval:
                total=sum(p.stat().st_size for p in self.output.iterdir()
                          if p.is_file() and self._supported(p.name)
                          and p.name not in self.published)
                if total<self.max_raw_bytes:return False
            ready=self._ready()
            if not ready:return False
            new_bytes=sum(p.stat().st_size for p in ready)
            if not final and self.interval>0 and now-self.last_tick<self.interval and new_bytes<self.max_raw_bytes:
                return False
            self._create(ready)
        part,manifest=self.pending
        if not final and self.interval>0 and self.failed_uploads:
            retry_delay=min(120.,2.**min(self.failed_uploads,7))
            if now-self.last_tick<retry_delay:return False
        try:
            self.uploader.save(str(part),base_path=str(self.output),policy='now')
        except Exception as error:
            self.failed_uploads+=1
            self.last_error=type(error).__name__
            self.last_tick=now
            return False
        self.published.update({k:v['sha256'] for k,v in manifest['files'].items()})
        self.pending=None
        self.last_error=None
        self.last_tick=now
        if final and self._ready():self.tick(final=True)
        return True

    def stats(self):
        return {'storage/eval_hf_published_files':self.published_count,
                'storage/eval_hf_parts':self.parts_created,
                'storage/eval_hf_failed_upload_attempts':self.failed_uploads,
                'storage/eval_hf_last_error_type':self.last_error,
                'storage/eval_hf_pending':int(self.pending is not None)}
