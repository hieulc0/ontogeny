"""Bounded Colab allocation, positive readiness proof and export-before-release."""

import json
import secrets
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from ontogeny.execution.artifacts import confined, verify_bundle, verify_inputs, write_json
from ontogeny.execution.contracts import ExecutionContext, Request, relative_path


class ColabError(RuntimeError):
    def __init__(self, code: str, *, fallback_safe: bool = False):
        super().__init__(code)
        self.fallback_safe = fallback_safe


class Session:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.directory = directory.resolve()
        self.config = self.directory / "sessions.json"
        self.name = "ontogeny-" + secrets.token_hex(8)
        self.nonce = secrets.token_hex(16)
        self.commands = 0

    def invoke(self, args: list[str], timeout: int) -> subprocess.CompletedProcess:
        self.commands += 1
        result = subprocess.run(
            ["colab", "--config", str(self.config), *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        # CLI traces/state can contain credentials. They are private and never bundled.
        path = self.directory / f"command-{self.commands}.log"
        path.write_text(result.stdout + result.stderr)
        path.chmod(0o600)
        if self.config.exists():
            self.config.chmod(0o600)
        return result

    def assigned(self) -> bool:
        if not self.config.exists():
            return False
        data = json.loads(self.config.read_text()).get(self.name)
        return bool(data and data.get("endpoint"))

    def release(self) -> None:
        if self.assigned():
            self.invoke(["stop", "-s", self.name], 45)
            # status synchronizes server assignments; exit zero alone is insufficient.
            self.invoke(["status", "-s", self.name], 30)
            if self.assigned():
                raise ColabError("cleanup_unconfirmed")

    def allocate(self) -> dict:
        write_json(self.directory / "ownership.json", {"session": self.name, "nonce": self.nonce})
        try:
            result = self.invoke(["new", "-s", self.name, "--gpu", "T4"], 90)
        except subprocess.TimeoutExpired as error:
            # The server may have assigned a VM before the client timed out.
            # No second allocation and no concurrent GPU fallback without reconciliation.
            raise ColabError("allocation_unknown") from error
        if not self.assigned():
            text = (result.stdout + result.stderr).lower()
            explicit_refusal = any(
                s in text
                for s in (
                    "allocation refused",
                    "backend rejected",
                    "service unavailable",
                    "503",
                    "quota",
                )
            )
            if explicit_refusal or result.returncode is not None:
                raise ColabError("allocation_unavailable", fallback_safe=True)
            raise ColabError("allocation_or_auth_unconfirmed")
        # Successful-looking output is not enough; warm up this one assignment only.
        remote = "/content/ontogeny-ready-" + self.nonce + ".json"
        probe = self.directory / "probe.py"
        probe.write_text(
            "import json,os,torch\nfrom pathlib import Path\n"
            "ok=torch.cuda.is_available()\n"
            'if ok:\n torch.empty(1,device="cuda");torch.cuda.synchronize()\n'
            f"Path({remote!r}).write_text(json.dumps(dict(nonce={self.nonce!r},cuda=ok,"
            "gpu=torch.cuda.get_device_name() if ok else None,"
            'colab=bool(os.environ.get("COLAB_RELEASE_TAG")))))\n'
        )
        deadline = time.monotonic() + 120
        for attempt in range(2):
            local = self.directory / f"proof-{attempt}.json"
            try:
                self.invoke(["exec", "-s", self.name, "-f", str(probe), "--timeout", "40"], 50)
                self.invoke(["download", "-s", self.name, remote, str(local)], 20)
                proof = json.loads(local.read_text())
                if (
                    proof.get("nonce") == self.nonce
                    and proof.get("cuda") is True
                    and proof.get("colab") is True
                    and "T4" in str(proof.get("gpu"))
                ):
                    return proof
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            if time.monotonic() + 70 > deadline:
                break
        self.release()
        raise ColabError("readiness_failed", fallback_safe=True)


def bootstrap(request: Request, context: ExecutionContext, remote_root: str) -> str:
    return f"""import os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_VISIBLE_DEVICES']='0'
root=Path({remote_root!r})
root.mkdir(exist_ok=True)
request=root/'request.json'
request.write_text({json.dumps(request.to_dict())!r})
inputs=root/'inputs'
inputs.mkdir(exist_ok=True)
def run(args,cwd=None):
    subprocess.run(args,cwd=cwd,check=True,timeout=900)
try:
    for name,revision in [('ontogeny',{request.source["ontogeny_sha"]!r}),
                          ('adaptive-mini-gpt',{request.source["upstream_sha"]!r})]:
        repo=root/name
        run(['git','clone','https://github.com/hieulc0/'+name+'.git',str(repo)])
        run(['git','checkout','--detach',revision],repo)
    run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
    run([sys.executable,'-m','pip','install','numpy==2.1.3','pyyaml==6.0.2'])
    os.environ['PYTHONPATH']=str(root/'ontogeny/src')+':'+str(root/'adaptive-mini-gpt/src')
    run([sys.executable,'-m','ontogeny.execution.worker','--request',str(request),
         '--context',{json.dumps(vars(context))!r},'--upstream',str(root/'adaptive-mini-gpt'),
         '--inputs',str(inputs),'--output',str(root/'result')],root/'ontogeny')
finally:
    with zipfile.ZipFile(root/'result.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        if (root/'result').exists():
            for path in (root/'result').rglob('*'):
                if path.is_file():archive.write(path,str(path.relative_to(root/'result')))
"""


def extract_bundle(archive_path: Path, output: Path, limit: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        if sum(i.file_size for i in archive.infolist()) > limit:
            raise ColabError("archive_budget_exceeded")
        names = set()
        for item in archive.infolist():
            name = relative_path(item.filename)
            if name in names or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ColabError("unsafe_archive")
            names.add(name)
            target = confined(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)


def execute(session: Session, request: Request, inputs: Path, directory: Path) -> dict:
    if request.runtime["visible_cuda_devices"] != 1:
        raise ColabError("single_visible_device_required")
    verify_inputs(request, inputs)
    remote_root = "/content/ontogeny-job-" + session.nonce
    script = session.directory / "worker_bootstrap.py"
    context = ExecutionContext("colab", session.name, "attempt-1")
    script.write_text(bootstrap(request, context, remote_root))
    # Fresh qualification is implemented first; private checkpoint transport is explicit.
    if request.inputs:
        raise ColabError("checkpoint_upload_not_qualified")
    try:
        session.invoke(["exec", "-s", session.name, "-f", str(script), "--timeout", "1000"], 1050)
    except subprocess.TimeoutExpired:
        # Always try recovering the bundle before stopping; never reroute a started trial.
        pass
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "result.zip"
    session.invoke(["download", "-s", session.name, remote_root + "/result.zip", str(archive)], 60)
    extract_bundle(archive, directory / "result", request.limits.output_bytes)
    receipt = verify_bundle(directory / "result", request)
    write_json(directory / "receipt.json", receipt)
    return receipt
