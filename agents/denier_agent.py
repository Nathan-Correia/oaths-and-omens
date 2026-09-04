"""
DenierAgent for engine: greedy_agent's buy phase, rectification, and
setup-phase policies as-is, heuristic_agent's battle targeting as-is
(see those modules' docstrings), but a different movement priority -
once this faction holds at least DENY_MIN_OWN_OUTPOSTS outposts of its
own (so it isn't sacrificing its very first foothold - there's nothing
worth denying early anyway, before anyone's built anything), it goes
after whichever outpost belongs to the CURRENT VP LEADER (ties broken toward the
weakest-defended, within ATTACK_TOLERANCE hexes of the nearest one - see
heuristic_agent's _best_attack_target for why distance dominates that
tie-break) ahead of continuing its own expansion. Destroying an outpost
is worth 2 VP immediately and cuts the leader's recurring per-round VP -
a double swing that a purely-expand-first strategy (greedy_agent,
heuristic_agent) never deliberately goes after, since both only ever
attack once expansion options run out. Falls back to heuristic_agent's
usual expand-then-attack-nearest priority whenever there's no leader (a
tie for first) or no reachable outpost belonging to them.

This is the flip side of turtle_agent's question: if turtle asks "is
attacking worth it at all," denier asks "is attacking the LEADER
specifically, ahead of your own growth, worth the tempo it costs."
"""

import random

import numpy as np

from .greedy_agent import _move_toward, greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import (
    ATTACK_TOLERANCE, OUTPOST_DEFENSE_POWER, _best_attack_target, _best_expansion_target, _power,
    heuristic_target,
)
from .random_agent import random_rectification
from engine.buy import _outpost_count
from engine.geometry import hex_distance

DENY_MIN_OWN_OUTPOSTS = 1
# How many extra hexes (beyond the board's own radius) a denial detour
# may cost before it's not worth abandoning expansion for - see
# _leader_attack_target's docstring for why this exists: on a bigger
# board, "go attack whoever's winning" can mean a trek clear across the
# map, burning many turns' worth of tempo that greedy's always-attack-
# nearest never has to pay. Scaled BY the board radius (not a flat hex
# count) since the same absolute distance means something very
# different on a radius-4 board than a radius-8 one.
MAX_DENY_DISTANCE_FACTOR = 0.6


def _current_leader(state, faction):
    """The faction (other than us) with the strictly highest VP total, or
    None if there's a tie for first (including a tie that involves us) -
    "the" leader has to be unambiguous for this to mean anything."""
    vp = state.victory_points
    top = int(np.max(vp))
    contenders = [f for f in range(state.num_factions) if int(vp[f]) == top]
    if len(contenders) != 1 or contenders[0] == faction:
        return None
    return contenders[0]


def _leader_attack_target(state, faction):
    """Nearest-to-our-biggest-army outpost belonging to the current
    leader, tie-broken by weakest-defended within ATTACK_TOLERANCE -
    same shape as heuristic_agent's _best_attack_target, just filtered
    to one faction's outposts instead of anyone's. None if there's no
    unambiguous leader, they hold no outposts, we have no mobile army, or
    even their NEAREST outpost is farther than MAX_DENY_DISTANCE_FACTOR *
    board radius away (see that constant's docstring - testing on a
    radius-8 board showed this chasing the leader clear across the map,
    stalling expansion for many turns in a way the smaller boards this
    was tuned on never exposed)."""
    leader = _current_leader(state, faction)
    if leader is None:
        return None
    grid = state.grid
    origins = np.nonzero((state.army_faction == faction) & ~state.locked)[0]
    if len(origins) == 0:
        return None
    sizes = state.army_units[origins].sum(axis=1)
    ref_coord = grid.coord_of(int(origins[int(np.argmax(sizes))]))

    outposts = np.nonzero((state.city_owner == leader) & ~state.is_capital)[0]
    if len(outposts) == 0:
        return None
    distances = {int(o): hex_distance(ref_coord, grid.coord_of(int(o))) for o in outposts}
    min_dist = min(distances.values())
    if min_dist > MAX_DENY_DISTANCE_FACTOR * grid.radius:
        return None
    candidates = [o for o, d in distances.items() if d <= min_dist + ATTACK_TOLERANCE]
    best = min(candidates, key=lambda o: _power(state.army_units[o]) + OUTPOST_DEFENSE_POWER)
    return [grid.coord_of(best)]


def denier_move(state, faction, legal_mask):
    grid = state.grid
    origins = np.nonzero(legal_mask.any(axis=1))[0]
    if len(origins) == 0:
        return None

    sizes = state.army_units[origins].sum(axis=1)
    # kind="stable": see greedy_agent's note - numpy's default argsort leaves
    # ties in an unreproducible order, and army sizes tie constantly.
    ranked = [int(origins[i]) for i in np.argsort(-sizes, kind="stable")]

    if _outpost_count(state, faction) >= DENY_MIN_OWN_OUTPOSTS:
        leader_target = _leader_attack_target(state, faction)
        if leader_target is not None:
            move = _move_toward(grid, ranked, legal_mask, leader_target)
            if move is not None:
                return move

    home_target = _best_expansion_target(state, faction)
    if home_target is not None:
        move = _move_toward(grid, ranked, legal_mask, [home_target], skip_arrived=True)
        if move is not None:
            return move

    attack_targets = _best_attack_target(state, faction)
    if attack_targets is None:
        return None
    return _move_toward(grid, ranked, legal_mask, attack_targets)


def make_denier_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    heuristic_agent's/greedy_agent's; movement is denier_move (see
    module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return denier_move(state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return denier_move(state, faction, legal_mask)

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
