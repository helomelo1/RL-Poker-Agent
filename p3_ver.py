"""Smoke-test each opponent: python phase3_verify.py."""

import numpy as np

from opponents import CallingStation, RandomOpponent, RuleBasedOpponent
from poker_env import PokerEnv


def test_opponent(opponent):
    env = PokerEnv(opponent_policy=opponent, seed=42)
    payoffs = []

    for hand in range(20):
        observation, info = env.reset()
        done = False

        while not done:
            legal = np.flatnonzero(info["action_mask"])

            # Our test agent checks/calls. Opponent behavior varies.
            action = 1 if 1 in legal else int(legal[0])

            observation, reward, done, truncated, info = env.step(action)
            assert not truncated
            assert np.isfinite(reward)

        payoffs.append(info["payoff_chips"])

    env.close()
    print(
        f"{opponent.name:16} "
        f"hands={len(payoffs):2} "
        f"avg_payoff={np.mean(payoffs):+6.2f} chips "
        f"range=[{min(payoffs):+4.0f}, {max(payoffs):+4.0f}]"
    )


def main():
    test_opponent(CallingStation())
    test_opponent(RandomOpponent(seed=123))
    test_opponent(RuleBasedOpponent(seed=456))
    print("Phase 3 smoke test passed.")


if __name__ == "__main__":
    main()
