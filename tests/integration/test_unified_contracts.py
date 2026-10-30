"""Local-safe request/transport ownership checks using opaque fake checkpoints."""

import json

import pytest

from ontogeny.execution.cli import qualification_request
from ontogeny.execution.contracts import ExecutionContext, Request
from ontogeny.execution.providers.colab import bootstrap


def test_current_contract_has_one_source_identity():
    request = qualification_request("a" * 40, "b" * 64)
    assert request.schema_version == 2
    assert request.source == {"ontogeny_sha": "a" * 40, "engine": "mini_gpt"}
    body = bootstrap(request, ExecutionContext("colab", "owned", "attempt-1"), "/content/work")
    assert "adaptive-mini-gpt" not in body and "--upstream" not in body


def test_historical_request_remains_readable_without_becoming_current():
    value = qualification_request("a" * 40, "b" * 64).to_dict()
    value["schema_version"] = 1
    value["source"] = {"ontogeny_sha": "a" * 40, "upstream_sha": "c" * 40}
    old = Request.parse(value)
    assert json.loads(json.dumps(old.to_dict())) == value
    with pytest.raises(RuntimeError, match="historical"):
        bootstrap(old, ExecutionContext("colab", "owned", "attempt-1"), "/content/work")


def test_request_defaults_to_current_internal_source(tmp_path, monkeypatch, capsys):
    import sys

    from ontogeny.execution import cli

    protocol = tmp_path / "protocol.md"
    protocol.write_text("test protocol")
    output = tmp_path / "request.json"
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: "a" * 40 + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ontogeny-exec",
            "qualification-request",
            "--protocol",
            str(protocol),
            "--output",
            str(output),
        ],
    )
    cli.main()
    request = Request.read(output)
    assert request.source == {"ontogeny_sha": "a" * 40, "engine": "mini_gpt"}
    assert json.loads(capsys.readouterr().out)["request_sha256"] == request.sha256
