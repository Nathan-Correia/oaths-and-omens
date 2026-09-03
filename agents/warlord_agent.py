"""
WarlordAgent for engine: vanguard_agent's army-cycling movement (fixes
the single-army-monopolizes-every-movement-step starvation problem
diagnosed in that module's docstring - rotating which ARMY gets first
pick each step instead of always the single largest one) combined with
denier_agent's insight (once a faction holds a foothold of its own,
deliberately keeping the CURRENT VP LEADER's outposts visible as
targets - not just whichever enemy outpost happens to be nearest - pays
for itself: 2 VP for the kill plus it cuts the leader's ongoing income,
a double swing greedy_agent/heuristic_agent never deliberately go
after).

The combination is a straightforward merge, not new mechanics: see
_all_targets, which is vanguard_agent's version (nearest-k legal
expansion sites + nearest-k enemy outposts) with the current leader's
own outposts folded into the same shared pool once this faction holds
DENY_MIN_OWN_OUTPOSTS outposts of its own - every mobile army still
just walks toward whichever pooled target is nearest to IT specifically
(see vanguard_move), so a leader outpost only actually gets chosen when
some army is well-positioned for it, same as any other target; it's
just guaranteed to be IN the pool (denier_agent's own MAX_DENY_DISTANCE
_FACTOR cap doesn't apply here for the same reason - an army only ever
picks a leader target when it's already its closest option, so there's
no separate cross-map-detour risk to cap against).
"""

import random

from .denier_agent import _current_leader
from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .random_agent import random_rectification
from .vanguard_agent import _best_direction, _ranked_attack_targets, _ranked_expansion_targets
from engine.buy import _outpost_count
from engine.geometry import hex_distance
from engine.state import NO_FACTION
from engine.turn import CAVALRY_STEPS, MOVEMENT_STEPS

DENY_MIN_OWN_OUTPOSTS = 1  # see denier_agent's constant of the same name for the reasoning


def _leader_targets(state, faction):
    """Every outpost belonging to the current unambiguous VP leader
    (see denier_agent._current_leader), once this faction holds at
    least DENY_MIN_OWN_OUTPOSTS of its own - empty otherwise (no
    leader, or too early to be worth the diversion)."""
    if int(_outpost_count(state, faction)[0]) < DENY_MIN_OWN_OUTPOSTS:
        return []
    leader = _current_leader(state, faction)
    if leader is None:
        return []
    grid = state.grid
    outposts = ((state.city_owner[0] == leader) & ~state.is_capital[0]).nonzero(as_tuple=False).flatten().tolist()
    return [grid.coord_of(o) for o in outposts]


def _all_targets(state, faction):
    """vanguard_agent's target pool (nearest-k legal expansion sites +
    nearest-k enemy outposts), with the current leader's own outposts
    folded in (see _leader_targets) - or every enemy capital as a last
    resort if the combined pool is empty (same fallback vanguard_agent/
    greedy_agent use, for the same reason)."""
    targets = _ranked_expansion_targets(state, faction) + _ranked_attack_targets(state, faction)
    targets += [c for c in _leader_targets(state, faction) if c not in targets]
    if targets:
        return targets

    grid = state.grid
    capitals = (
        state.is_capital[0] & (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction)
    ).nonzero(as_tuple=False).flatten().tolist()
    return [grid.coord_of(c) for c in capitals]


def warlord_move(state, faction, step, legal_mask, total_steps=1):
    """Identical mechanics to vanguard_agent's vanguard_move - rotate
    which mobile army gets first pick each step, each pursuing whichever
    pooled target (see _all_targets) is nearest to IT - just drawing
    from this module's target pool instead of vanguard_agent's."""
    grid = state.grid
    mobile = sorted(int(h) for h in legal_mask.any(dim=1).nonzero(as_tuple=False).flatten().tolist())
    if not mobile:
        return None

    targets = _all_targets(state, faction)
    if not targets:
        return None

    steps_remaining = total_steps - step - 1
    n = len(mobile)
    for offset in range(n):
        origin = mobile[(step + offset) % n]
        origin_coord = grid.coord_of(origin)
        target = min(targets, key=lambda c: hex_distance(origin_coord, c))
        if hex_distance(origin_coord, target) == 0:
            continue
        legal_dirs = legal_mask[origin].nonzero(as_tuple=False).flatten().tolist()
        if not legal_dirs:
            continue
        return origin, _best_direction(state, grid, origin, legal_dirs, target, steps_remaining)
    return None


def make_warlord_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's; movement is warlord_move (see
    module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return warlord_move(state, faction, step, legal_mask, total_steps=MOVEMENT_STEPS)

    def decide_cavalry(state, faction, step, legal_mask):
        return warlord_move(state, faction, step, legal_mask, total_steps=CAVALRY_STEPS)

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
