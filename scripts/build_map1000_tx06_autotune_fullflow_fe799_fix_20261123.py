"""Independent Kaggle notebook addressing confirmed old kernel-ref default.

The Kaggle 372917422 UI failed to serve its log publicly at inspection time.
No claim that a fixed wrong reference explains that unobserved traceback.
The science bundle, trainer/val/test settings and optimizer remain unchanged.
"""
from pathlib import Path
import hashlib
import nbformat
from scripts.build_map1000_tx06_autotune_fullflow_ui_20261123 import (
    build_notebook as build_prior_notebook, ROOT,
)

TARGET_KERNEL_REF='longlonthiu/notebookfe799af211'
NAME='MATD3-MAP1000-TX06-T4X2-AUTOTUNE-FIXFE799-V372917422-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME


def build_notebook():
    n=build_prior_notebook()
    setup=n.cells[1].source
    old="os.environ.get('KAGGLE_KERNEL_REF','longlonthiu/notebookbbd0aec150')"
    new=f"os.environ.get('KAGGLE_KERNEL_REF',{TARGET_KERNEL_REF!r})"
    if setup.count(old)!=1:
        raise RuntimeError('Unexpected Kaggle reference template, refusing unsafe replacement')
    setup=setup.replace(old,new,1)
    setup=setup.replace("if not re.fullmatch(r'[\\w-]+/[\\w-]+',NOTEBOOK_REF):",
        "print('KAGGLE_TARGET_KERNEL_REF',NOTEBOOK_REF,flush=True)\nif not re.fullmatch(r'[\\w-]+/[\\w-]+',NOTEBOOK_REF):",1)
    assert 'notebookbbd0aec150' not in setup
    n.cells[1].source=setup
    n.cells[0].source+='''

### V372917422 kernel-reference fix
This is an independent version of the same full-flow MAP1000/TX0.6 notebook;
its default `KAGGLE_KERNEL_REF` is now `longlonthiu/notebookfe799af211`,
not the obsolete bbd0aec150 notebook. Set `KAGGLE_KERNEL_REF` explicitly in
Kaggle only if importing under another username/slug.

**Verification scope:** local embedded-source preflight, simulated T4 autotune,
CPU MATD3 smoke, forensic configuration, Val100/Test1000 protocol. The original
Kaggle 372917422 traceback was not accessible through Kaggle's public UI,
so this fix is not yet claimed to resolve that remote error.
'''
    n.metadata['target_kernel_ref']=TARGET_KERNEL_REF
    n.metadata['origin_failed_version_id']='372917422'
    n.metadata['fix_scope']='stale_kernel_ref_plus_full_kaggle_bootstrap_regression'
    n.metadata['remote_cuda_run_verified']=False
    nbformat.validate(n)
    if len(nbformat.writes(n).encode('utf-8'))>=850_000:
        raise RuntimeError('Notebook unexpectedly exceeds Kaggle 1MB source budget')
    return n


def main():
    n=build_notebook();OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    print('FIXED_V372917422_NOTEBOOK',str(OUT),'BYTES',OUT.stat().st_size,
          'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),
          'SOURCE_SHA256',n.metadata['source_sha256'])

if __name__=='__main__':main()
