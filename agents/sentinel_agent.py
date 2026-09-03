"""
SentinelAgent for engine: vanguard_agent's army-cycling movement (see
that module's docstring) plus a behavior none of this package's other
agents have at all - reinforcing a friendly OUTPOST that's currently
under attack. Capitals are deliberately excluded (see
_locked_own_cities): they can't be captured or destroyed by any means -
an attacker who wins there is just evicted with nothing to show for it -
so a capital under attack has nothing at stake worth diverting an
expanding army over, unlike an outpost, which really can be destroyed
and really does cost its owner VP.

Every other agent here is purely offense/expansion-facing; nothing ever
routes an army toward defending. That's a real gap the rules explicitly
support: "because armies lock in place as soon as a battle starts,
additional armies are free to move into a battle on a later step of the
same movement phase" - and since engine/turn.py's battle phase always
runs a battle to full resolution the same turn it starts (never spans
turn boundaries - see resolve_full_battle), a reinforcement only ever
has ONE turn's remaining movement/cavalry steps to matter, but within
that window it's a completely normal move: arriving at an
already-locked hex you're already a combatant on (your own capital or
outpost defending itself) just adds your units as another contribution
slot to YOUR side of that same fight (see movement.py's
_start_or_extend_battle / apply_movement_step - locked destinations
just extend the battle, no special reinforcement action needed).
Losing an outpost costs its owner not just the outpost itself but the
attacker's immediate 2 VP and the loser's recurring per-round VP going
forward - worth defending against a fight that's still winnable, given
a capital/outpost's own free defense shot is often not enough by itself
(11-20 on a d20, ~27.5% per shot).

sentinel_move checks every step whether any of this faction's own
outposts are currently locked (i.e. under attack right now) and, if a
mobile army can reach one this step, sends it there ahead of
vanguard_agent's normal expansion/attack rotation - a defended
foothold is what all that expansion was for in the first place. Falls
through to vanguard_move unchanged when nothing needs defending.

RESULT: no measurable benefit, possibly a small cost - a 150-game
head-to-head against plain vanguard_agent (identical except for this
behavior) came out at 10.7%, if anything slightly below the ~12.5%
baseline. Best guess, consistent with the pattern noted across this
package's other "smarter" experiments (denier_agent's leader-targeting
did nothing once layered on top of vanguard_agent either - see
warlord_agent; legion_agent's anti-redundancy claims actively backfired):
this game is tempo-driven enough that diverting an already-productive
expanding army to save one outpost - which still only has whatever
units happen to already be positioned nearby, arriving piecemeal over
however many steps are left in the turn, against an attacker who
chose that target BECAUSE it looked winnable - usually costs more in
lost expansion than it saves in preserved VP. Kept in the package as a
documented real capability difference (nothing else here ever
defends at all) rather than reverted, since it isn't a regression, just
an unproven one.
"""

import random

import numpy as np

from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .random_agent import random_rectification
from .vanguard_agent import _best_direction, vanguard_move
from engine.geometry import hex_distance
from engine.turn import CAVALRY_STEPS, MOVEMENT_STEPS


def _locked_own_cities(state, faction):
    """Coordinates of every OUTPOST (not capital - see module docstring)
    this faction owns that's currently locked in an active battle -
    i.e. worth reinforcing this turn, since it won't still be a valid
    target come next turn's movement phase (the battle will already be
    over by then)."""
    grid = state.grid
    hexes = np.nonzero((state.city_owner == faction) & state.locked & ~state.is_capital)[0]
    return [grid.coord_of(int(h)) for h in hexes]


def sentinel_move(state, faction, step, legal_mask, total_steps=1):
    grid = state.grid
    mobile = sorted(int(h) for h in np.nonzero(legal_mask.any(axis=1))[0])
    if not mobile:
        return None

    defense_targets = _locked_own_cities(state, faction)
    if defense_targets:
        steps_remaining = total_steps - step - 1
        ranked = sorted(mobile, key=lambda o: min(hex_distance(grid.coord_of(o), c) for c in defense_targets))
        for origin in ranked:
            origin_coord = grid.coord_of(origin)
            target = min(defense_targets, key=lambda c: hex_distance(origin_coord, c))
            legal_dirs = np.nonzero(legal_mask[origin])[0]
            if len(legal_dirs) == 0:
                continue
            return origin, _best_direction(state, grid, origin, legal_dirs, target, steps_remaining)

    return vanguard_move(state, faction, step, legal_mask, total_steps)


def make_sentinel_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's; movement is sentinel_move (see
    module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return sentinel_move(state, faction, step, legal_mask, total_steps=MOVEMENT_STEPS)

    def decide_cavalry(state, faction, step, legal_mask):
        return sentinel_move(state, faction, step, legal_mask, total_steps=CAVALRY_STEPS)

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
