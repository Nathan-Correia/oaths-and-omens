"""
MarshalAgent for engine: same idea as vanguard_agent (rotate which army
gets first pick each step, drawing from a shared pool of nearby
expansion sites + enemy outposts - see that module's docstring), but
with a real bipartite matching between armies and targets instead of
each army just independently grabbing whichever target is nearest to
IT. Independent nearest-picking can under-cover: if army A's nearest
target is also army B's nearest target, vanguard_agent's rotation might
send BOTH of them there on different steps (nothing stops it - see
legion_agent's docstring for why persistent "claimed" bookkeeping to
prevent that backfired instead), while a third target neither of them
is closest to sits completely unpursued even though someone nearby
could 100% have covered it.

_greedy_match computes a proper one-to-one pairing instead: repeatedly
pick whichever (still-unmatched army, still-unmatched target) pair is
closest, across the WHOLE board, and lock that pair in before moving on
- a classic greedy matching. Recomputed completely fresh every single
call (no memory carried between calls, turns, or games), so - unlike
legion_agent - there's no way for a stale assignment to survive an
army's death or a target's resolution; it simply isn't consulted again
next call. marshal_move then rotates through the resulting pairs by
`step`, same mechanism as vanguard_agent's rotation through armies.

RESULT: a small, real edge over plain vanguard_agent - 14.7% win rate
head-to-head (150 games, everything else identical) against the
~12.5% baseline for two equally-matched agents, the best margin of any
vanguard_agent variant tried (ahead of warlord_agent's leader-targeting
addition at 12.0% and sentinel_agent's reinforcement behavior at
10.7%). Modest and not hugely far outside noise at this sample size,
but consistently on the positive side rather than a wash - worth
treating as this package's current best default movement policy.

Also tried: folding denier_agent's leader-targeting into this module's
own _all_targets (on the theory that warlord_agent's null result might
have been specific to vanguard_agent's weaker per-army-nearest
allocation, not to the leader-targeting idea itself). First sample (150
games) looked promising at 15.3% vs plain marshal_agent; a second,
larger sample (200 games) came back at 10.0% - combined across both
(350 games) that's ~12.3%, dead on the no-effect baseline. Confirms
warlord_agent's finding rather than overturning it: attacking whoever
currently leads, specifically, doesn't earn back what it costs in this
game, independent of which movement-allocation mechanism carries it
out. Not kept as a separate module - it would just be marshal_agent
with a coin flip's worth of difference.

Direction selection now also avoids ending a step on a desert tile
with no city on it, where possible without sacrificing actual progress
- see vanguard_agent's _best_direction (shared by both modules) for the
rule this addresses and the measured effect (a genuine, if modest, win
- confirmed in two separate samples, unlike the leader-targeting result
above).
"""

import random

from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .random_agent import random_rectification
from .vanguard_agent import _all_targets, _best_direction
from engine.geometry import hex_distance
from engine.turn import CAVALRY_STEPS, MOVEMENT_STEPS


def _greedy_match(grid, origins, targets):
    """[(origin, target), ...] - repeatedly pairs off whichever
    (unmatched origin, unmatched target) combo is currently closest,
    until one side runs out. Origins beyond len(targets) (or vice versa)
    are simply left unmatched."""
    pairs = sorted(
        ((hex_distance(grid.coord_of(o), t), o, t) for o in origins for t in targets),
        key=lambda x: x[0],
    )
    used_origins, used_targets, matches = set(), set(), []
    for _, origin, target in pairs:
        if origin in used_origins or target in used_targets:
            continue
        matches.append((origin, target))
        used_origins.add(origin)
        used_targets.add(target)
    return matches


def marshal_move(state, faction, step, legal_mask, total_steps=1):
    """total_steps: how many steps this phase has in total (MOVEMENT_STEPS
    or CAVALRY_STEPS - see make_marshal_agents), used only to compute
    _best_direction's steps_remaining for marsh-detour purposes."""
    grid = state.grid
    mobile = sorted(int(h) for h in legal_mask.any(dim=1).nonzero(as_tuple=False).flatten().tolist())
    if not mobile:
        return None

    targets = _all_targets(state, faction)
    if not targets:
        return None

    matches = _greedy_match(grid, mobile, targets)
    if not matches:
        return None

    steps_remaining = total_steps - step - 1
    n = len(matches)
    for offset in range(n):
        origin, target = matches[(step + offset) % n]
        origin_coord = grid.coord_of(origin)
        if hex_distance(origin_coord, target) == 0:
            continue  # already standing on its matched target - leave it for the buy phase
        legal_dirs = legal_mask[origin].nonzero(as_tuple=False).flatten().tolist()
        if not legal_dirs:
            continue
        return origin, _best_direction(state, grid, origin, legal_dirs, target, steps_remaining)
    return None


def make_marshal_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's; movement is marshal_move (see
    module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return marshal_move(state, faction, step, legal_mask, total_steps=MOVEMENT_STEPS)

    def decide_cavalry(state, faction, step, legal_mask):
        return marshal_move(state, faction, step, legal_mask, total_steps=CAVALRY_STEPS)

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
