"""Provider-neutral source bootstrap and artifact verification for this experiment."""

import hashlib
import json
from pathlib import Path


def bootstrap(revision: str, root: str) -> str:
    return f"""import json,os,subprocess,sys,zipfile,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path({root!r});root.mkdir(parents=True,exist_ok=False)
repo=root/'ontogeny'
def run(args,cwd=None,timeout=900):subprocess.run(args,cwd=cwd,check=True,timeout=timeout)
try:
 run(['git','clone','https://github.com/hieulc0/ontogeny.git',str(repo)])
 run(['git','checkout','--detach',{revision!r}],repo)
 run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
 run([sys.executable,'-m','pip','install','numpy==2.1.3'])
 os.environ['PYTHONPATH']=str(repo/'src')
 run([sys.executable,str(repo/'experiments/neural/initial_student/pilot.py'),'--output',str(root/'result')],repo,timeout=300)
finally:
 with zipfile.ZipFile(root/'result.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  if (root/'result').exists():
   for p in (root/'result').rglob('*'):
    if p.is_file():z.write(p,p.relative_to(root/'result'))
 if (root/'result/result.json').exists():
  shutil.copyfile(root/'result/result.json',root/'summary.json')
"""


def verify(directory: Path) -> dict:
    result = json.loads((directory / "result.json").read_text())
    if (
        result["execution"] != "SUCCEEDED"
        or result["science"] != "NOT_RUN"
        or not result["continuation_exact"]
        or not all(result["state_groups"].values())
    ):
        raise RuntimeError("pilot result did not pass its engineering contract")
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "result.json"
    }
    if actual != set(result["files"]):
        raise RuntimeError("missing or unindexed pilot files")
    for name, entry in result["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError("pilot artifact size or link mismatch")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha != entry["sha256"]:
            raise RuntimeError("pilot artifact hash mismatch")
    return result
