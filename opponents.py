import random

from state import RANKS, card_index


FOLD = 0
CHECK_CALL = 1
RAISE_HALF_POT = 2
RAISE_POT = 3
ALL_IN = 4


def legal_actions(observation):
    return list(observation["legal_actions"])


def rank_and_suit(card):
    index = card_index(card)
    rank = index % 13
    suit = index // 13

    return rank, suit


class CallingStation:
    "Type 1 Opponent: Never Raises or Folds"
    name = "calling_station"

    def __call__(self, observation):
        legal = legal_actions(observation)
        return CHECK_CALL if CHECK_CALL in legal else legal[0]


class RandomOpponent:
    "Type 2 Opponent: Choose randomly among legal actions"

    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, observation):
        return self.rng.choice(legal_actions(observation))


class RuleBasedOpponent:
    """
    Type 3 Opponent: Has a specific play style

    Preflop: Plays pairs and high cards aggresively;
                folds weak hands when has to call a large amount

    Postflop: Raises with a pair or better, sometimes can semi-bluff high cards;
                avoids large calls with nothing in hand
    """

    name = "rule_based"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)


    def __call__(self, observation):
        legal = legal_actions(observation)
        raw = observation["raw_obs"]

        hole = raw["hand"]
        board = raw["public_cards"]

        hole_ranks = [rank_and_suit(card)[0] for card in hole]
        hole_suits =  [rank_and_suit(card)[1] for card in hole]
        board_ranks = [rank_and_suit(card)[0] for card in board]

        to_call = max(raw["all_chips"]) - raw["my_chips"]
        pot = max(1, raw["pot"])

        if not board:
            pair = hole_ranks[0] == hole_ranks[1]
            high_card = max(hole_ranks)
            low_card = min(hole_ranks)
            suited = hole_suits[0] == hole_suits[1]

            strong = pair and high_card >= 8
            strong = strong or (high_card >= 11 and low_card >= 9)
            playable = pair or high_card >= 11 or (suited and high_card >= 9)

            if strong and RAISE_POT in legal:
                return RAISE_POT

            if playable and RAISE_HALF_POT in legal and self.rng.random() < 0.5:
                return RAISE_HALF_POT

            if not playable and to_call >= 4 and FOLD in legal:
                return FOLD

            return CHECK_CALL
            
        paired_board_with_hole = any(rank in board_ranks for rank in hole_ranks)
        high_cards = max(hole_ranks) >= RANKS.index("K")
        
        if paired_board_with_hole and RAISE_HALF_POT in legal:
            return RAISE_HALF_POT
    
        if high_cards and to_call == 0 and RAISE_HALF_POT in legal:
            if self.rng.random() <= 0.15:
                return RAISE_HALF_POT

        if not paired_board_with_hole and to_call > pot * 0.5:
            if FOLD in legal:
                return FOLD

        return CHECK_CALL