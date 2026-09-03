"""
LegionAgent for engine: vanguard_agent's army-cycling movement (see that
module's docstring for the starvation problem it fixes), plus one more
refinement aimed at a DIFFERENT inefficiency the same diagnostic process
surfaced: nothing stops two separate armies from both independently
deciding a SAME target is their closest option and both heading there
across different turns - one of them arriving is enough (only one unit
is ever consumed founding an outpost, or a battle only needs one army to
fight it), so the second army's trip is wasted effort that could have
covered a DIFFERENT opportunity instead.

LegionAgent tracks, per faction, which targets already have an army
assigned - a plain Python set closed over by make_legion_agents,
persisting across the WHOLE game (not just one turn: an expansion
target several hexes away can take multiple turns to reach, so
"already claimed" has to survive turn boundaries to mean anything -
see _prune_claims for how a claim is eventually released again). When
choosing a target for the army whose turn it is in the rotation, an
unclaimed target is always preferred over a claimed one, even if the
claimed one is technically closer - only once every reachable target is
already claimed does it double up.

RESULT: this backfired badly (8% vs greedy - below the ~12.5% baseline -
and crushed 2-to-1 head-to-head against plain vanguard_agent). Best
working explanation: _prune_claims only releases a claim once its
target disappears from the pool entirely (built/destroyed/fell out of
the nearest-k window) - it has no idea whether the army that claimed a
target is still alive or working on it. Combat is constant in this
game, so an assigned army dying en route (common) leaves its claim
stuck indefinitely, poisoning a perfectly good, totally uncontested
opportunity for as long as it stays inside the nearest-k window - which
can be most of the game. Meanwhile the "duplicate effort" this was
built to prevent turns out to be mostly self-correcting already:
vanguard_agent's plain version recomputes the target pool fresh every
single call, so the instant one army actually founds an outpost (or an
outpost gets destroyed), it drops out of _ranked_expansion_targets/
_ranked_attack_targets on its own and any OTHER army still routed
toward it naturally retargets next call - no persistent memory needed,
and therefore nothing to go stale. Left in the package as a documented
dead end (see the pattern noted in this repo's other reverted
experiments: danger-avoidance scoring, outpost garrisoning, weakest-
target-first attacking - stateful "smarter" refinements keep losing to
simple, fully-stateless-recomputed-every-call heuristics in this
particular, fast/chaotic game).
"""

import random

from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .random_agent import random_rectification
from .vanguard_agent import _all_targets
from engine.geometry import hex_distance


def _prune_claims(claimed, targets):
    """Drops any claimed target that's no longer in the current target
    pool - meaning it got resolved somehow (outpost founded/destroyed
    there, or it fell out of the nearest-k window) - so a claim is
    released once there's nothing left for it to represent, without
    needing to track WHICH army was pursuing it or whether that army
    died en route."""
    claimed.intersection_update(targets)


def legion_move(state, faction, step, legal_mask, claimed):
    """Same rotation as vanguard_agent's vanguard_move, except target
    selection prefers whichever of `targets` isn't already in `claimed`
    (falls back to the overall nearest if everything reachable is
    already claimed). Mutates `claimed` in place, adding whatever target
    this call ends up committing an army to."""
    grid = state.grid
    mobile = sorted(int(h) for h in legal_mask.any(dim=1).nonzero(as_tuple=False).flatten().tolist())
    if not mobile:
        return None

    targets = _all_targets(state, faction)
    if not targets:
        return None
    _prune_claims(claimed, targets)

    n = len(mobile)
    for offset in range(n):
        origin = mobile[(step + offset) % n]
        origin_coord = grid.coord_of(origin)
        unclaimed = [t for t in targets if t not in claimed]
        pool = unclaimed or targets
        target = min(pool, key=lambda c: hex_distance(origin_coord, c))
        if hex_distance(origin_coord, target) == 0:
            continue
        legal_dirs = legal_mask[origin].nonzero(as_tuple=False).flatten().tolist()
        if not legal_dirs:
            continue
        best_dir = min(
            legal_dirs,
            key=lambda d: hex_distance(grid.coord_of(int(grid.neighbor_table[origin, d])), target),
        )
        claimed.add(target)
        return origin, int(best_dir)
    return None


def make_legion_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's; movement is legion_move (see
    module docstring), with a per-faction `claimed` set closed over here
    and persisting for the whole game (see _prune_claims for how a
    claim is released again)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}
    claimed = {f: set() for f in range(num_factions)}

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return legion_move(state, faction, step, legal_mask, claimed[faction])

    def decide_cavalry(state, faction, step, legal_mask):
        return legion_move(state, faction, step, legal_mask, claimed[faction])

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
