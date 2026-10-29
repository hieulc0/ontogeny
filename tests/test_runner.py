from ontogeny.evaluation.baselines import HeuristicPolicy, RandomPolicy
from ontogeny.organism.cell import run_episode
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WorldConfig


def test_controls_and_logging_are_reproducible():
    for cls in (RandomPolicy, HeuristicPolicy):
        log = []
        a = run_episode(CellWorld(seed=3), cls(7), record=log.append)
        b = run_episode(CellWorld(seed=3), cls(7))
        assert a == b and len(log) == a.duration
        assert a.duration <= 100 and a.terminated != a.truncated


def test_episode_resets_do_not_reset_learner():
    class FakeLearner(RandomPolicy):
        updates = 0

        def learn(self, experience):
            self.updates += 1

    policy = FakeLearner(8)
    for _ in range(2):
        run_episode(CellWorld(WorldConfig(max_steps=1)), policy, learning=True)
    assert policy.updates == 2
    run_episode(CellWorld(WorldConfig(max_steps=1)), policy, learning=False)
    assert policy.updates == 2
