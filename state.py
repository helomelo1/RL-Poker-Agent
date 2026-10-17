"""
State Layout (113 float32 values):
    [0:52]    My hole cards: one bit per card in the 52-card deck
    [52:104]  Community cards: one bit per card
    [104:108] Street: one-hot [preflop, flop, turn, river]
    [108]     Am I the button? (1=yes, 0=no)
    [109]     Pot / starting stack
    [110]     My remaining chips / starting stack
    [111]     Opponent remaining chips / starting stack
    [112]     Amount I must contribute to call / starting stack
"""

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "SHDC"
STATE_SIZE = 113


def card_index(card: str) -> int:
    """Map a card such as 'SA' or 'AS' to an index in [0, 51].
    
    Accepts both suit-first ('SA') and rank-first ('AS') notation."""

    if len(card) != 2:
            raise ValueError(f"Expected a two-character card, got {card!r}")
    
    if card[0] in SUITS and card[1] in RANKS:
        suit, rank = card[0], card[1]
    elif card[0] in RANKS and card[1] in SUITS:
        rank, suit = card[0], card[1]
    else:
        raise ValueError(f"Unrecognized card: {card!r}")

    return SUITS.index(suit) * 13 + RANKS.index(rank)


def encode_state(
    hole_cards,
    community_cards,
    is_button: bool,
    pot: float,
    my_chips: float,
    opponent_chips: float,
    to_call: float,
    starting_stack: float,
) -> np.ndarray:
    if len(hole_cards) != 2:
        raise ValueError("Expected exactly 2 hole cards")
    if len(community_cards) not in (0, 3, 4, 5):
            raise ValueError("Expected 0, 3, 4, or 5 community cards")
    if starting_stack <= 0:
        raise ValueError("starting_stack must be positive")
    if min(pot, my_chips, opponent_chips, to_call) < 0:
        raise ValueError("Chip amounts cannot be negative")

    hole_indices = [card_index(card) for card in hole_cards]
    board_indices = [card_index(card) for card in community_cards]

    if len(set(hole_indices + board_indices)) != len(hole_indices + board_indices):
        raise ValueError("The same card appears more than once")

    state = np.zeros(STATE_SIZE, dtype=np.float32)

    state[hole_indices] = 1.0
    state[52 + np.array(board_indices, dtype=int)] = 1.0

    street = {0:0, 3: 1, 4: 2, 5: 3}[len(community_cards)]
    state[104 + street] = 1.0

    state[108] = float(is_button)
    state[109:113] = np.array(
        [pot, my_chips, opponent_chips, to_call], dtype=np.float32
    ) / starting_stack

    return state