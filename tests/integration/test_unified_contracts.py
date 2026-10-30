"""Local-safe request/transport ownership checks using opaque fake checkpoints."""

import json

import pytest

from mini_gpt.provenance import HISTORICAL_REVISION
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
    value["source"] = {"ontogeny_sha": "a" * 40, "upstream_sha": HISTORICAL_REVISION}
    old = Request.parse(value)
    assert json.loads(json.dumps(old.to_dict())) == value
    with pytest.raises(RuntimeError, match="historical"):
        bootstrap(old, ExecutionContext("colab", "owned", "attempt-1"), "/content/work")
