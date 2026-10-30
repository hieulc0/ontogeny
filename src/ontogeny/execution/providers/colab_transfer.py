"""Bounded Colab file transport with full-file integrity verification."""

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Protocol


class ColabCommands(Protocol):
    name: str
    directory: Path

    def invoke(self, args: list[str], timeout: int) -> subprocess.CompletedProcess[str]: ...


CHUNK_BYTES = 8 * 1024 * 1024


def _command(session: ColabCommands, args: list[str], timeout: int = 90) -> None:
    if session.invoke(args, timeout).returncode:
        raise RuntimeError("Colab file transport failed; inspect private command log")


def _script(session: ColabCommands, name: str, code: str) -> None:
    path = session.directory / name
    path.write_text(code)
    _command(session, ["exec", "-s", session.name, "-f", str(path), "--timeout", "90"], 100)


def upload_file(
    session: ColabCommands,
    source: Path,
    remote: str,
    expected_sha256: str,
    *,
    chunk_bytes: int = CHUNK_BYTES,
) -> None:
    """Stage one file in small requests and verify its full remote contents."""
    if chunk_bytes <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid transfer parameters")
    parts = remote + ".upload-parts"
    _script(session, "upload-setup.py", f"from pathlib import Path\nPath({parts!r}).mkdir()\n")
    count = 0
    digest = hashlib.sha256()
    total = 0
    local_part = session.directory / "upload-part.bin"
    with source.open("rb") as stream:
        while data := stream.read(chunk_bytes):
            digest.update(data)
            total += len(data)
            local_part.write_bytes(data)
            _command(
                session, ["upload", "-s", session.name, str(local_part), f"{parts}/{count:06d}"]
            )
            count += 1
    if count == 0 or digest.hexdigest() != expected_sha256:
        raise ValueError("local upload does not match declared checkpoint")
    local_part.unlink()
    receipt = remote + ".receipt.json"
    code = f"""import hashlib,json,os
from pathlib import Path
parts=Path({parts!r}); dest=Path({remote!r}); digest=hashlib.sha256(); size=0
with dest.open('xb') as out:
 for i in range({count}):
  data=(parts / f"{{i:06d}}").read_bytes(); out.write(data); digest.update(data); size+=len(data)
 out.flush(); os.fsync(out.fileno())
if digest.hexdigest()!={expected_sha256!r} or size!={total}:
 raise ValueError('uploaded file mismatch')
Path({receipt!r}).write_text(json.dumps(dict(sha256=digest.hexdigest(),size=size,count={count})))
"""
    _script(session, "upload-assemble.py", code)
    local_receipt = session.directory / "upload-receipt.json"
    _command(session, ["download", "-s", session.name, receipt, str(local_receipt)])
    if json.loads(local_receipt.read_text()) != {
        "sha256": expected_sha256,
        "size": total,
        "count": count,
    }:
        raise ValueError("remote upload receipt mismatch")


def download_file(
    session: ColabCommands,
    remote: str,
    destination: Path,
    *,
    max_bytes: int,
    chunk_bytes: int = CHUNK_BYTES,
) -> None:
    """Recover bounded pieces, verify every piece and the final remote digest."""
    if chunk_bytes <= 0 or max_bytes <= 0:
        raise ValueError("invalid transfer bounds")
    parts = remote + ".download-parts"
    manifest = remote + ".manifest.json"
    code = f"""import hashlib,json
from pathlib import Path
source=Path({remote!r}); parts=Path({parts!r}); parts.mkdir()
entries=[]; total=0; full=hashlib.sha256()
with source.open('rb') as stream:
 while data:=stream.read({chunk_bytes}):
  total+=len(data)
  if total>{max_bytes}:raise ValueError('download exceeds budget')
  full.update(data); name=f"{{len(entries):06d}}"; (parts/name).write_bytes(data)
  entries.append(dict(name=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
Path({manifest!r}).write_text(json.dumps(dict(size=total,sha256=full.hexdigest(),parts=entries)))
"""
    _script(session, "download-split.py", code)
    local_manifest = session.directory / "download-manifest.json"
    _command(session, ["download", "-s", session.name, manifest, str(local_manifest)])
    info = json.loads(local_manifest.read_text())
    if not 0 < info["size"] <= max_bytes or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"]):
        raise ValueError("invalid remote file manifest")
    full = hashlib.sha256()
    size = 0
    part = session.directory / "download-part.bin"
    with destination.open("xb") as out:
        for index, item in enumerate(info["parts"]):
            name = f"{index:06d}"
            if item["name"] != name or not 0 < item["size"] <= chunk_bytes:
                raise ValueError("invalid remote chunk manifest")
            _command(session, ["download", "-s", session.name, f"{parts}/{name}", str(part)])
            data = part.read_bytes()
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("downloaded chunk mismatch")
            out.write(data)
            full.update(data)
            size += len(data)
            if size > max_bytes:
                raise ValueError("download exceeds budget")
    part.unlink(missing_ok=True)
    if size != info["size"] or full.hexdigest() != info["sha256"]:
        raise ValueError("downloaded file mismatch")
