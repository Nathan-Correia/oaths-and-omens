"""
TurtleAgent for engine: greedy_agent's buy phase, rectification, and
setup-phase policies as-is, heuristic_agent's battle targeting as-is (see
those modules' docstrings), but NEVER voluntarily initiates a fight -
movement is pure expansion, full stop. If there's nowhere left to expand
(OUTPOST_CAP reached, or no hex on the board is currently legal to found
on), it just sits still rather than falling back to attacking anything,
unlike every other agent in this package.

This exists to isolate a question heuristic_agent/greedy_agent's results
couldn't answer on their own: is any of the attacking those two do
actually profitable, or is a "denial" attack's average value close to (or
below) zero once you account for the units it costs, in a game this
tempo-driven? Comparing turtle head-to-head against heuristic (identical
in every other respect) puts a number on exactly that gap. Turtle can
obviously still end up fighting - moving into a contested tile, or being
attacked at its own capital/outposts - it just never chooses to start
one.
"""

import random

import numpy as np

from .greedy_agent import _move_toward, greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import _best_expansion_target, heuristic_target
from .random_agent import random_rectification


def turtle_move(state, faction, legal_mask):
    grid = state.grid
    origins = np.nonzero(legal_mask.any(axis=1))[0]
    if len(origins) == 0:
        return None

    sizes = state.army_units[origins].sum(axis=1)
    # kind="stable": see greedy_agent's note - numpy's default argsort leaves
    # ties in an unreproducible order, and army sizes tie constantly.
    ranked = [int(origins[i]) for i in np.argsort(-sizes, kind="stable")]

    home_target = _best_expansion_target(state, faction)
    if home_target is None:
        return None
    return _move_toward(grid, ranked, legal_mask, [home_target], skip_arrived=True)


def make_turtle_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    heuristic_agent's/greedy_agent's; movement is turtle_move (see
    module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return turtle_move(state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return turtle_move(state, faction, legal_mask)

    def decide_target(state, hex_index, faction):
        return heuristic_target(state, hex_index, faction)

    def decide_rectification(state, hex_index, winner_faction, cap):
        return random_rectification(state, hex_index, winner_faction, cap, rngs[winner_faction])

    def decide_resource_choice(state, faction, hex_index):
        return greedy_resource_choice(state, faction)

    def decide_placement(state, faction, legal_mask):
        return greedy_placement(state, legal_mask)

    def decide_draft(state, faction, legal_pool):
        return greedy_draft(state, legal_pool)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        return greedy_swap(state, leftover_hex, placer_hex)

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
        {f: decide_resource_choice for f in factions},
        {f: decide_placement for f in factions},
        {f: decide_draft for f in factions},
        {f: decide_swap for f in factions},
    )
