"""Inspect a heads-up RLCard no-limit Hold'em hand.

Run: python phase1_verify.py
"""

from pprint import pprint

import rlcard


def main():
    env = rlcard.make(
        "no-limit-holdem",
        config={
            "seed": 42,
            "game_num_players": 2,
            "chips_for_each": 100,
        },
    )

    observation, player_id = env.reset()
    raw = observation["raw_obs"]

    print("=== Environment ===")
    print("Players:", env.num_players)
    print("Actions:", env.num_actions)
    print("Configured starting chips:", env.game.init_chips)
    print("Blinds:", env.game.small_blind, "/", env.game.big_blind)
    print("Dealer ID:", env.game.dealer_id)
    print("First player to act:", player_id)

    print("\n=== First observation ===")
    print("Observation keys:", list(observation))
    print("RLCard's encoded observation shape:", observation["obs"].shape)
    print("Raw observation:")
    pprint(raw)

    # Confirm the agreed starting conditions.
    assert env.num_players == 2
    assert list(env.game.init_chips) == [100, 100]
    assert (env.game.small_blind, env.game.big_blind) == (1, 2)
    assert raw["pot"] == 3
    assert sorted(raw["stakes"]) == [98, 99]
    assert len(raw["hand"]) == 2
    assert len(raw["public_cards"]) == 0

    print("\n=== One complete hand (check/call when possible) ===")
    step_count = 0

    while not env.is_over():
        legal = list(observation["legal_actions"])  # encoded action IDs
        print(
            f"Step {step_count}: player={player_id}, "
            f"stage={raw['stage']}, "
            f"hand={raw['hand']}, "
            f"board={raw['public_cards']}, "
            f"pot={raw['pot']}, "
            f"stakes={raw['stakes']}, "
            f"legal={legal}"
        )

        # RLCard's no-limit action 1 is CHECK_CALL. Use a legal
        # fallback rather than assuming every action is always available.
        action = 1 if 1 in legal else legal[0]
        observation, player_id = env.step(action)
        raw = observation["raw_obs"]
        step_count += 1

        if step_count > 100:
            raise RuntimeError("Hand did not finish within 100 actions")

    print("\n=== Result ===")
    print("Hand finished after", step_count, "actions")
    print("Final board:", raw["public_cards"])
    print("Payoffs:", env.get_payoffs())
    print("\nPhase 1 checks passed.")


if __name__ == "__main__":
    main()
