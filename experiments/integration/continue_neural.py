"""Remote fresh-process continuation probe for a trained independent student."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from mini_gpt.runtime import configure_determinism

    torch, _ = configure_determinism()
    from mini_gpt.checkpoint.comparison import capture_state
    from mini_gpt.learning.state import LearningState
    from mini_gpt.learning.trainer import train_tokens

    owner = LearningState.load(args.checkpoint)
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
    train_tokens(owner, x, (x + 1) % 32, max_updates=3)
    owner.external["cursor"] += 1
    torch.save(capture_state(owner), args.output)


if __name__ == "__main__":
    main()
