"""Bounded migration verification. Every real tensor/model check is remote CUDA only."""

import os
from pathlib import Path

import pytest

if not (
    (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and Path("/kaggle/working").is_dir())
    or (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir())
):
    pytest.skip("remote CUDA qualification only", allow_module_level=True)

import importlib
import json
import random
import subprocess
import sys
from typing import cast

import numpy as np
import torch

from experiments.integration.import_historical import HISTORICAL_REVISION, import_dense_checkpoint
from mini_gpt.checkpoint.comparison import capture_state, equal
from mini_gpt.checkpoint.identity import file_digest
from mini_gpt.learning.optimizer import optimizer_membership
from mini_gpt.learning.state import LearningState
from mini_gpt.learning.trainer import train_tokens
from mini_gpt.model.config import ModelConfig
from mini_gpt.model.transformer import MiniGPT
from mini_gpt.runtime import configure_determinism

pytestmark = pytest.mark.remote
configure_determinism()


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def make_owner(dropout=0.0):
    seed()
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=16,
        dropout=dropout,
    )
    return LearningState(
        MiniGPT(config).cuda(), lr=0.0005, external={"rng": random.Random(12), "cursor": 0}
    )


def batch():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
    return x, (x + 1) % 32


@pytest.mark.parametrize("encoding", ["absolute", "rope", "alibi"])
def test_dense_reference_equivalence(encoding):
    ReferenceConfig = importlib.import_module("adaptive_mini_gpt.model.config").ModelConfig
    ReferenceModel = importlib.import_module("adaptive_mini_gpt.model.transformer").MiniGPT
    ReferenceState = importlib.import_module(
        "adaptive_mini_gpt.training.learning_state"
    ).LearningState

    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=16,
        dropout=0.0,
        pos_encoding=encoding,
        tie_word_embeddings=True,
    )
    seed(51)
    reference = ReferenceState(
        ReferenceModel(ReferenceConfig(**config.to_dict())).cuda(), lr=0.0005
    )
    seed(51)
    candidate = LearningState(MiniGPT(config).cuda(), lr=0.0005)
    assert equal(reference.model.state_dict(), candidate.model.state_dict(), torch, np)
    x, y = batch()
    expected, old_loss = reference.model(x, y)
    actual, new_loss = candidate.model(x, y)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(new_loss, old_loss, atol=0, rtol=0)
    reference.optimizer.zero_grad(set_to_none=True)
    old_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.model.parameters(), 1.0)
    reference.optimizer.step()
    candidate.backward(new_loss)
    candidate.step()
    assert equal(reference.model.state_dict(), candidate.model.state_dict(), torch, np)
    assert equal(reference.optimizer.state_dict(), candidate.optimizer.state_dict(), torch, np)
    assert equal(
        [p.grad for p in reference.model.parameters()],
        [p.grad for p in candidate.model.parameters()],
        torch,
        np,
    )
    with pytest.raises(ValueError):
        candidate.model(x, torch.full_like(y, -100))
    # Future tokens cannot affect earlier outputs in the shared hidden traversal.
    changed = x.clone()
    changed[:, -1] = 19
    torch.testing.assert_close(
        candidate.model(x)[0][:, :-1], candidate.model(changed)[0][:, :-1], atol=0, rtol=0
    )


def test_optimizer_alias_rollback_and_immutable_publication(tmp_path):
    owner = make_owner()
    x, y = batch()
    train_tokens(owner, x, y, max_updates=3)
    owner.external["parameter"] = cast(MiniGPT, owner.model).wte.weight
    owner.external["model"] = owner.model
    owner.external["optimizer"] = owner.optimizer
    before = cast(MiniGPT, owner.model).wte.weight.detach().clone()
    assert not owner.candidate(lambda s: s.model.wte.weight.data.add_(1), lambda s: False)
    assert torch.equal(before, cast(MiniGPT, owner.model).wte.weight)
    path = tmp_path / "full.pt"
    owner.save(path)
    restored = LearningState.load(path)
    assert restored.external["model"] is restored.model
    assert restored.external["optimizer"] is restored.optimizer
    assert restored.external["parameter"] is cast(MiniGPT, restored.model).wte.weight
    assert not any(optimizer_membership(restored.model, restored.optimizer).values())
    embedding_ids = {
        id(m.weight) for m in restored.model.modules() if isinstance(m, torch.nn.Embedding)
    }
    assert all(
        group["weight_decay"] == 0
        for group in restored.optimizer.param_groups
        if any(id(p) in embedding_ids or p.ndim < 2 for p in group["params"])
    )
    with pytest.raises(FileExistsError):
        restored.save(path)


def test_nonempty_optimizer_fresh_process_continuation(tmp_path):
    owner = make_owner(dropout=0.1)
    owner.scheduler = torch.optim.lr_scheduler.StepLR(owner.optimizer, step_size=1, gamma=0.9)
    x, y = batch()
    for _ in range(2):
        train_tokens(owner, x, y, max_updates=3)
        owner.external["cursor"] += 1
    checkpoint = tmp_path / "pending.pt"
    owner.save(checkpoint)
    assert owner.optimizer.state and all(p.grad is not None for p in owner.model.parameters())
    train_tokens(owner, x, y, max_updates=3)
    owner.external["cursor"] += 1
    expected = capture_state(owner)
    output = tmp_path / "next.pt"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "experiments/integration/continue_neural.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, timeout=60)
    actual = torch.load(output, weights_only=False)
    assert equal(expected, actual, torch, np)


def test_trained_historical_checkpoint_conversion(tmp_path):
    checkpoint = Path(os.environ["MIGRATION_CHECKPOINT"])
    expected_hash = os.environ["MIGRATION_CHECKPOINT_SHA256"]
    assert file_digest(checkpoint) == expected_hash
    reference = tmp_path / "reference"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        os.environ["HISTORICAL_ONTOGENY_SRC"] + ":" + os.environ["REFERENCE_NEURAL_SRC"]
    )
    script = Path(__file__).resolve().parents[2] / "experiments/integration/legacy_reference.py"
    subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint), "--output", str(reference)],
        env=environment,
        check=True,
        timeout=60,
    )
    owner = import_dense_checkpoint(
        checkpoint, expected_sha256=expected_hash, source_revision=HISTORICAL_REVISION, trusted=True
    )
    before = list(capture_state(owner))
    before[4].pop("checkpoint_imports")
    assert equal(tuple(before), torch.load(reference / "before.pt", weights_only=False), torch, np)
    from ontogeny.learning.experience import Experience
    from ontogeny.neural.policy import PolicyCell
    from ontogeny.world.environment import CellWorld
    from ontogeny.world.state import WorldConfig

    world = CellWorld(WorldConfig(initial_energy=100, max_energy=100, max_steps=5), seed=93021)
    observation = world.observe()
    owner.external["driver"] = {"world": world.snapshot(), "observation": observation}

    class RecordedPolicy(PolicyCell):
        def probabilities(self, observation):
            values = super().probabilities(observation)
            self.latest_probabilities = values
            return values

    cell = RecordedPolicy(owner, max_updates=3)
    rows = []
    while True:
        action = cell.act(observation)
        probs = cell.latest_probabilities
        outcome = world.step(action)
        cell.learn(Experience(observation, action, outcome))
        owner.metadata["interactions"] += 1
        owner.external["driver"] = {"world": world.snapshot(), "observation": outcome.observation}
        rows.append(
            {"probabilities": probs, "action": int(action), "energy": outcome.observation.energy}
        )
        observation = outcome.observation
        if outcome.terminated or outcome.truncated:
            break
    owner.metadata["episode_cursor"] += 1
    after = list(capture_state(owner))
    after[4].pop("checkpoint_imports")
    assert equal(tuple(after), torch.load(reference / "after.pt", weights_only=False), torch, np)
    assert rows == json.loads((reference / "trace.json").read_text())
    published = tmp_path / "converted.pt"
    owner.save(published)
    restored = LearningState.load(published)
    assert equal(capture_state(owner), capture_state(restored), torch, np)
    assert file_digest(checkpoint) == expected_hash
    with pytest.raises(ValueError, match="hash"):
        import_dense_checkpoint(
            checkpoint, expected_sha256="0" * 64, source_revision=HISTORICAL_REVISION, trusted=True
        )


@pytest.fixture(autouse=True)
def allocation_budget(request):
    import time

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    yield
    peak = torch.cuda.max_memory_allocated()
    assert peak <= 256 * 1024 * 1024
    record = {
        "test": request.node.name,
        "peak_torch_memory_bytes": peak,
        "wall_seconds": time.monotonic() - started,
    }
    directory = os.environ.get("MIGRATION_TEST_RECORDS")
    if directory:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        (path / (request.node.name + ".json")).write_text(json.dumps(record))
