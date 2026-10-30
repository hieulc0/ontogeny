"""File transport tests use only local bytes and a fake Colab command surface."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ontogeny.execution.providers.colab_transfer import download_file, upload_file


class FakeSession:
    name = "test"

    def __init__(self, directory: Path, *, corrupt: bool = False):
        self.directory = directory
        self.directory.mkdir()
        self.corrupt = corrupt
        self.uploads = 0
        self.downloads = 0

    def invoke(self, args: list[str], timeout: int) -> subprocess.CompletedProcess:
        if args[0] == "exec":
            return subprocess.run([sys.executable, args[args.index("-f") + 1]], check=False)
        if args[0] == "upload":
            self.uploads += 1
            shutil.copyfile(args[-2], args[-1])
        elif args[0] == "download":
            self.downloads += 1
            shutil.copyfile(args[-2], args[-1])
            if self.corrupt and args[-2].endswith("/000001"):
                Path(args[-1]).write_bytes(b"bad")
        return subprocess.CompletedProcess(args, 0)


def test_chunked_file_round_trip(tmp_path):
    session = FakeSession(tmp_path / "private")
    source = tmp_path / "original"
    data = bytes(range(250)) * 9
    source.write_bytes(data)
    remote = tmp_path / "remote"
    output = tmp_path / "recovered"
    upload_file(session, source, str(remote), hashlib.sha256(data).hexdigest(), chunk_bytes=512)
    assert session.uploads == 5
    assert remote.read_bytes() == data
    download_file(session, str(remote), output, max_bytes=4096, chunk_bytes=512)
    assert session.downloads == 7
    assert output.read_bytes() == data


def test_rejects_corrupt_download(tmp_path):
    session = FakeSession(tmp_path / "private", corrupt=True)
    source = tmp_path / "remote"
    source.write_bytes(b"a" * 1000)
    with pytest.raises(ValueError, match="chunk mismatch"):
        download_file(session, str(source), tmp_path / "out", max_bytes=4096, chunk_bytes=512)


def test_rejects_incorrect_upload_hash(tmp_path):
    session = FakeSession(tmp_path / "private")
    source = tmp_path / "original"
    source.write_bytes(b"test")
    with pytest.raises(ValueError, match="does not match"):
        upload_file(session, source, str(tmp_path / "remote"), "0" * 64, chunk_bytes=2)
    assert not (tmp_path / "remote").exists()
