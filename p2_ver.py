"""Smoke-test the Gymnasium wrapper: python phase2_verify.py."""

import numpy as np

from poker_env import PokerEnv


def main():
    env = PokerEnv(seed=42)

    for hand in range(10):
        observation, info = env.reset()
        first_position = "button" if observation[108] == 1 else "big blind"

        assert observation.shape == (113,)
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()

        terminated = False
        steps = 0

        while not terminated:
            legal = np.flatnonzero(info["action_mask"])
            assert len(legal) > 0

            # Check/call to exercise all four streets when possible.
            action = 1 if 1 in legal else int(legal[0])
            observation, reward, terminated, truncated, info = env.step(action)

            assert not truncated
            assert observation.shape == (113,)
            assert np.isfinite(observation).all()

            steps += 1
            assert steps < 100, "Hand took unexpectedly many decisions"

        print(
            f"Hand {hand + 1:2}: first position={first_position:9} "
            f"decisions={steps} payoff={info['payoff_chips']:+.0f} chips "
            f"reward={reward:+.2f}"
        )

    env.close()
    print("Phase 2 smoke test passed.")


if __name__ == "__main__":
    main()
