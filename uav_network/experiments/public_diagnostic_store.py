"""Public Hugging Face storage for compressed scientific evidence, not W&B files.

Never fall back to W&B or a private Hub repo. Each immutable archive uploads to
runs/<wandb_run_id>/...; a read-back SHA256 verification is required before it
is marked complete. The writer leaves Kaggle-local archives intact on failure.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

from .checkpoint_store import storage_credentials
from .evidence import secret_values


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_repo_name(config):
    explicit=os.environ.get("UAV_DIAGNOSTIC_HF_REPO","").strip()
    if explicit:
        repo=explicit
    else:
        owner=getattr(config.tracking,"checkpoint_repo","auto")
        if owner=="auto":
            owner=os.environ.get("HF_REPO_ID","")
        if not owner:
            path=Path(config.tracking.credentials_file)
            if not path.is_absolute():
                path=Path(__file__).resolve().parents[2]/path
            if path.is_file():
                owner=json.loads(path.read_text()).get("huggingface",{}).get("repo_id","")
        repo=owner.split("/",1)[0]+"/uav-search-diagnostics" if "/" in owner else ""
    if not isinstance(repo,str) or not re.fullmatch(r"[\w.-]+/[\w.-]+",repo):
        raise ValueError("A public HF diagnostic repo must be 'owner/name'")
    return repo


class PublicHubDiagnosticUploader:
    def __init__(self,config,output,run_id,*,api=None,verify_download=True):
        self.output=Path(output).resolve()
        self.run_id=str(run_id)
        if not re.fullmatch(r"[\w.-]{5,64}",self.run_id):
            raise ValueError("Invalid public diagnostics run ID")
        self.repo_id=public_repo_name(config)
        self.api=api
        self.config=config
        self.verify_download=bool(verify_download)
        self.ready=False
        self.uploaded={}
        self.remote_verification_failures=0
        self.blocked_sensitive_parts=0
        self.last_error_type=None
        self.last_error_message=None

    @property
    def url(self):
        return "https://huggingface.co/datasets/"+self.repo_id+"/tree/main/runs/"+self.run_id

    def _ensure_ready(self):
        if self.ready:return
        if self.api is None:
            from huggingface_hub import HfApi
            token,_=storage_credentials(self.config,require_repo=False)
            self.api=HfApi(token=token)
        # Reusing an existing public repo must not create it on every run.
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            info=self.api.repo_info(repo_id=self.repo_id,repo_type="dataset")
        except RepositoryNotFoundError:
            self.api.create_repo(repo_id=self.repo_id,repo_type="dataset",
                                 private=False,exist_ok=True)
            info=self.api.repo_info(repo_id=self.repo_id,repo_type="dataset")
        if info.private is not False:
            raise RuntimeError("Refusing to upload diagnostics to a private HF repo; must be public")
        self.ready=True

    def _reject_secrets(self,path):
        # Prevent accidental public credential exposure. A failed scan blocks
        # the upload and leaves the original local artifact intact for review.
        candidates=[s.encode("utf-8") for s in secret_values(self.config)
                    if isinstance(s,str) and len(s)>=12]
        needles=list(dict.fromkeys(candidates))
        if not needles:return
        def check(stream):
            tail=b""
            for block in iter(lambda:stream.read(1<<20),b""):
                data=tail+block
                if any(needle in data for needle in needles):
                    raise ValueError("Public forensic archive contains a credential value; upload blocked")
                tail=data[-max(map(len,needles)):]
        path=Path(path)
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path,"r:gz") as archive:
                for member in archive:
                    if member.isfile():
                        with archive.extractfile(member) as stream:
                            check(stream)
        elif path.suffix==".gz":
            with gzip.open(path,"rb") as stream:check(stream)
        else:
            with path.open("rb") as stream:check(stream)

    def save(self,filename,*,base_path,policy):
        if policy!="now":raise ValueError("HF diagnostics must be immutable and uploaded once")
        file=Path(filename).resolve()
        root=Path(base_path).resolve()
        if root!=self.output or not file.is_relative_to(self.output) or not file.is_file():
            raise ValueError("Diagnostic file must be under this run output directory")
        rel=file.relative_to(self.output)
        if file.suffix!=".gz" or any(part in {"..","."} for part in rel.parts):
            raise ValueError("Only compressed immutable diagnostics may be published")
        path_in_repo=str(PurePosixPath("runs")/self.run_id/rel.as_posix())
        digest=_sha256(file)
        if self.uploaded.get(path_in_repo)==digest:return path_in_repo
        try:
            self._reject_secrets(file)
            self._ensure_ready()
            commit=self.api.upload_file(
                path_or_fileobj=str(file),path_in_repo=path_in_repo,
                repo_id=self.repo_id,repo_type="dataset",
                commit_message="Archive UAV forensics "+self.run_id+" "+file.name)
            if self.verify_download:
                from huggingface_hub import hf_hub_download
                with tempfile.TemporaryDirectory(prefix="uav-hf-audit-") as folder:
                    downloaded=hf_hub_download(
                        repo_id=self.repo_id,repo_type="dataset",filename=path_in_repo,
                        revision=commit.oid,token=False,local_dir=folder,force_download=True)
                    if _sha256(downloaded)!=digest:
                        self.remote_verification_failures+=1
                        raise IOError("Public Hugging Face read-back SHA-256 differs")
            self.uploaded[path_in_repo]=digest
            self.last_error_type=self.last_error_message=None
            return path_in_repo
        except Exception as exc:
            self.last_error_type=type(exc).__name__
            self.last_error_message=str(exc)[:150]
            if "credential value" in str(exc):
                self.blocked_sensitive_parts+=1
            raise

    def stats(self):
        return {"storage/hf_public_repo":self.repo_id,
                "storage/hf_public_url":self.url,
                "storage/hf_verified_uploaded_files":len(self.uploaded),
                "storage/hf_readback_sha256_enabled":int(self.verify_download),
                "storage/hf_readback_failures":self.remote_verification_failures,
                "storage/hf_blocked_sensitive_parts":self.blocked_sensitive_parts,
                "storage/hf_last_error_type":self.last_error_type}
