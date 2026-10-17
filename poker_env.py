import gymnasium as gym
import numpy as np
import rlcard
from gymnasium import spaces

from state import STATE_SIZE, encode_state


STARTING_STACK = 100
NUM_ACTIONS = 5 # fold, check/call, bet: half-pot, bet: full-pot, bet: all-in

def check_call_opponent(observation):
    legal = observation["legal_actions"]
    return 1 if 1 in legal else next(iter(legal))


class PokerEnv(gym.Env):
    """Heads-up poker from the perspective of one learning player."""

    metadata = {"render_modes": []}

    def __init__(self, opponent_policy=check_call_opponent, seed=None):
        super().__init__()

        self.game = rlcard.make(
            "no-limit-holdem",
            config={
                "seed": seed,
                "game_num_players": 2,
                "chips_for_each": STARTING_STACK,
            },
        )
        self.opponent_policy = opponent_policy

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,  # chip values normalized by 100; pot can reach 200
            shape=(STATE_SIZE,),
            dtype=np.float32,
        )

        self.hero_id = 0
        self.observation = None
        self.player_id = None
        self.hand_number = 0


    def _legal_mask(self):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if not self.game.is_over():
            for action in self.observation["legal_actions"]:
                mask[int(action)] = 1

        return mask


    def _encode_observation(self):
        raw = self.observation["raw_obs"]

        raised = self.game.game.round.raised
        to_call = max(raised) - raised[self.hero_id]

        opponent_id = 1 - self.hero_id
        stakes = raw["stakes"]

        return encode_state(
            hole_cards=raw["hand"],
            community_cards=raw["public_cards"],
            is_button=(self.hero_id == self.button_id),
            pot=raw["pot"],
            my_chips=stakes[self.hero_id],
            opponent_chips=stakes[opponent_id],
            to_call=to_call,
            starting_stack=STARTING_STACK,
        )


    def _play_opponent_turns(self):
        while not self.game.is_over() and self.player_id != self.hero_id:
            action = int(self.opponent_policy(self.observation))
           
            if action not in self.observation["legal_actions"]:
                raise ValueError(f"Opponent chose illegal action: {action}")

            self.observation, self.player_id = self.game.step(action)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.game.seed(seed)

        self.game.game.dealer_id = self.hand_number % 2
        self.hand_number += 1

        self.observation, self.player_id = self.game.reset()

        self.button_id = (self.game.game.dealer_id + 1) % 2

        self._play_opponent_turns()

        if self.game.is_over():
            raise RuntimeError("Hand ended before the agent's first decision")

        info = {"action_mask": self._legal_mask()}
        return self._encode_observation(), info


    def step(self, action):
        action = int(action)

        if action not in self.observation["legal_actions"]:
            raise ValueError(
                f"Illegal action {action}; legal actions are "
                f"{list(self.observation['legal_actions'])}"
            )

        if self.game.is_over() or self.player_id != self.hero_id:
            raise RuntimeError("step() called when it is not the agent's turn")

        self.observation, self.player_id = self.game.step(action)
        self._play_opponent_turns()

        terminated = self.game.is_over()
        truncated = False

        if terminated:
            payoff_chips = float(self.game.get_payoffs()[self.hero_id])
            reward = payoff_chips / STARTING_STACK

            observation = np.zeros(STATE_SIZE, dtype=np.float32)
            info = {
                "action_mask": np.zeros(NUM_ACTIONS, dtype=np.int8),
                "payoff_chips": payoff_chips,
            }
        else:
            reward = 0.0
            observation = self._encode_observation()
            info = {"action_mask": self._legal_mask()}

        return observation, reward, terminated, truncated, info


    def close(self):
        pass