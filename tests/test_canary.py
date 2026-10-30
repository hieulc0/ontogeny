import pytest

from ontogeny.world.canary import CueWorld


def test_actions_have_observable_consequences_without_action_labels():
    outcomes = []
    for action in (0, 1):
        world = CueWorld(12)
        assert len(world.observe().tiles) == 9
        feedback = world.step(action)
        outcomes.append(feedback.reward)
        with pytest.raises(RuntimeError):
            world.step(0)
    assert sorted(outcomes) == [-1, 1]


def test_delayed_feedback_is_terminal_and_deterministic():
    a, b = CueWorld(18, delayed=True), CueWorld(18, delayed=True)
    for index, action in enumerate((0, 1, 0)):
        assert a.observe() == b.observe()
        feedback = a.step(action)
        assert feedback == b.step(action)
        assert feedback.done == (index == 2)
        if index < 2:
            assert feedback.reward == 0
