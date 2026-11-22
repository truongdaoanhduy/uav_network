"""Bounded, immutable public HF evidence for GPU Val100/Final Test4000.

Large (e.g. 134-checkpoint validation_results.json) scientific outputs are
archived as recoverable byte ranges, NOT as unbounded individual tar parts.
No oracle or Actor policy executes in this diagnostic-only module. A source
file is acknowledged only after *all* of its ranges have uploaded successfully.
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


def _hash_range(path, start, size):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(start)
        remaining=size
        while remaining:
            chunk=stream.read(min(1<<20,remaining))
            if not chunk:
                raise IOError('Incomplete GPU forensic source while reading byte range')
            h.update(chunk)
            remaining-=len(chunk)
    return h.hexdigest()


class _SliceReader(io.RawIOBase):
    def __init__(self,source,length):
        self.source=source
        self.remaining=length
    def readable(self):return True
    def read(self,n=-1):
        if self.remaining<=0:return b''
        n=min(self.remaining, 1<<20 if n is None or n<0 else n)
        buf=self.source.read(n)
        self.remaining-=len(buf)
        return buf


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
        self.offsets={}
        self.source_hashes={}
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
                or bool(re.fullmatch(r'final_test_gpu\d+_\d{6}_\d{6}\.json',name))
                or name in {'final_test.json','report.json','validation_results.json',
                            'validation_results.csv','best.json','one-session-status-start.json',
                            'one-session-status-final.json','one-session-status-error.json'}
                or bool(re.fullmatch(r'[\w.-]+\.jsonl',name)))

    def _ready(self):
        files=[]
        for p in sorted(self.output.iterdir()):
            if (not p.is_file() or p.is_symlink() or not self._supported(p.name)
                    or p.name in self.published or p.stat().st_size==0):
                continue
            if p.suffix in {'.json','.jsonl'}:
                try:
                    with p.open('r') as stream:
                        if p.suffix=='.json':json.load(stream)
                        else:
                            for line in stream:
                                if line.strip():json.loads(line)
                except (OSError,ValueError,UnicodeDecodeError):
                    # The worker may still be writing a JSON file.
                    continue
            if p.name in self.source_hashes and _digest(p)!=self.source_hashes[p.name]:
                # Never splice two different versions of a scientific output.
                raise ValueError('GPU forensic source mutated after first range: '+p.name)
            files.append(p)
        return files

    def _create(self,files):
        entries=[]
        remaining=self.max_raw_bytes
        for path in files:
            total=path.stat().st_size
            start=self.offsets.get(path.name,0)
            if start>=total:continue
            count=min(total-start,remaining)
            if count<=0:break
            name=path.name if start==0 and count==total else f'{path.name}.chunk-{start:012d}.bin'
            if path.name not in self.source_hashes:
                self.source_hashes[path.name]=_digest(path)
            record={'bytes':count,'sha256':_hash_range(path,start,count),
                    'source_file':path.name,'source_total_bytes':total,
                    'source_sha256':self.source_hashes[path.name],
                    'offset_start':start,'offset_end':start+count}
            entries.append((path,name,record))
            remaining-=count
            if remaining==0:break
        if not entries:return None
        index=self.parts_created+1
        part=self.folder/f'part-{index:06d}.tar.gz'
        tmp=self.folder/f'part-{index:06d}.tmp'
        manifest={'schema':2,'scope':'raw_gpu_validation_final_test_evidence',
                  'files':{name:record for _,name,record in entries}}
        serialized=json.dumps(manifest,sort_keys=True).encode()
        try:
            with tarfile.open(tmp,'w:gz',compresslevel=6) as tar:
                for path,name,record in entries:
                    info=tarfile.TarInfo(name)
                    info.size=record['bytes'];info.mtime=0;info.mode=0o600
                    with path.open('rb') as stream:
                        stream.seek(record['offset_start'])
                        tar.addfile(info,_SliceReader(stream,info.size))
                info=tarfile.TarInfo('manifest.json')
                info.size=len(serialized);info.mtime=0;info.mode=0o600
                tar.addfile(info,io.BytesIO(serialized))
            with tarfile.open(tmp,'r:gz') as verified:
                if set(verified.getnames())!={*manifest['files'],'manifest.json'}:
                    raise IOError('Incomplete GPU evidence archive members')
                for name,record in manifest['files'].items():
                    h=hashlib.sha256()
                    with verified.extractfile(name) as stream:
                        for chunk in iter(lambda:stream.read(1<<20),b''):
                            h.update(chunk)
                    if h.hexdigest()!=record['sha256']:
                        raise IOError('GPU evidence range SHA256 mismatch')
            tmp.replace(part)
        finally:
            tmp.unlink(missing_ok=True)
        self.parts_created+=1
        self.pending=(part,manifest)
        return part

    def tick(self,*,final=False):
        uploaded_any=False
        while True:
            now=self.now()
            if self.pending is None:
                # Cheap stat-only fast path; no repeated multi-MB JSON parsing
                # at each 3-second W&B supervisor poll.
                if not final and self.interval>0 and now-self.last_tick<self.interval:
                    total=sum(max(0,p.stat().st_size-self.offsets.get(p.name,0))
                              for p in self.output.iterdir()
                              if p.is_file() and self._supported(p.name)
                              and p.name not in self.published)
                    if total<self.max_raw_bytes:return uploaded_any
                ready=self._ready()
                if not ready:return uploaded_any
                new_bytes=sum(p.stat().st_size-self.offsets.get(p.name,0) for p in ready)
                if not final and self.interval>0 and now-self.last_tick<self.interval and new_bytes<self.max_raw_bytes:
                    return uploaded_any
                self._create(ready)
            part,manifest=self.pending
            if not final and self.interval>0 and self.failed_uploads:
                delay=min(120.,2.**min(self.failed_uploads,7))
                if now-self.last_tick<delay:return uploaded_any
            try:
                self.uploader.save(str(part),base_path=str(self.output),policy='now')
            except Exception as error:
                self.failed_uploads+=1
                self.last_error=type(error).__name__
                self.last_tick=now
                return uploaded_any
            # Offsets advance only after the HF uploader reports success;
            # it verifies remote public readback before acknowledging a part.
            for name,record in manifest['files'].items():
                source=record['source_file'];start=record['offset_start']
                if self.offsets.get(source,0)!=start:
                    raise RuntimeError('Noncontiguous post-evaluation forensic byte offsets')
                self.offsets[source]=record['offset_end']
                if record['offset_end']==record['source_total_bytes']:
                    self.published[source]=record['source_sha256']
            self.pending=None
            self.last_error=None
            self.last_tick=now
            uploaded_any=True
            if not final:return True

    def stats(self):
        return {'storage/eval_hf_published_files':self.published_count,
                'storage/eval_hf_parts':self.parts_created,
                'storage/eval_hf_failed_upload_attempts':self.failed_uploads,
                'storage/eval_hf_last_error_type':self.last_error,
                'storage/eval_hf_pending':int(self.pending is not None)}
