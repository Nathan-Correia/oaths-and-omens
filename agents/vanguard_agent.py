"""
VanguardAgent for engine: same buy phase, battle targeting, rectification,
and setup-phase policies as greedy_agent/heuristic_agent (see those
modules' docstrings) - the change here is entirely in movement, aimed at
a structural inefficiency none of this package's other agents address.

THE PROBLEM (diagnosed via tournament.py + direct game traces, not
theory): every other agent's movement (greedy_agent's greedy_rush_move,
heuristic_agent's heuristic_move, denier_agent's denier_move) computes
ONE shared objective per call and hands it to _move_toward, which tries
candidate armies in size-ranked order and returns the FIRST one with any
legal step toward it - so the single largest (ties broken toward the
lowest hex index) mobile army wins EVERY movement-phase step, every
turn, for as long as it has any legal move at all, regardless of how
many OTHER armies exist. A faction typically accumulates 4-6 separate
armies by mid-game (buy_infantry purchases land wherever there's room -
capital, outposts - not necessarily merged into one stack), and tracing
a real game turn-by-turn found individual armies sitting at the exact
same hex for 4+ CONSECUTIVE turns without ever being assigned anything,
purely because some other, bigger/lower-indexed army kept winning the
single per-step movement slot. Confirmed this isn't a gold/economy
problem: giving a faction MORE preserved gold (see heuristic_agent's
smart_buy revert) just buys more units that feed the same single-file
queue - it doesn't get them moving.

THE FIX (see vanguard_move): rotate which ARMY gets first pick each
step, not which objective does. An earlier version tried the reverse -
cycle through several ranked objectives, each claimed by whichever army
happened to be closest to IT - and it didn't actually help (tested
WORSE than doing nothing): the nearest-k objectives tend to cluster in
whatever single direction currently has the most opportunity, so the
same well-positioned army kept winning most of them anyway, while an
army parked in a different direction was still never anyone's closest
match to anything. Explicitly indexing into the mobile-army list by
`step` instead guarantees up to min(3, army count) distinct armies are
considered every regular-movement phase (likewise up to 2 for the
cavalry phase), each pursuing whichever of a shared target pool (the
nearest few legal expansion sites plus the nearest few enemy outposts -
see _all_targets) happens to be nearest to IT specifically. If an
army's own nearest target is unreachable this step, the next army in
the rotation is tried instead, so a step is only wasted when truly
nothing is actionable for anyone.

Direction selection (_best_direction, shared with marshal_agent) also
tie-breaks away from ending a step on a desert tile with no city on it
- "any army that ends its full turn on a desert tile loses 1 unit" is a
real rule none of this package's pathfinding otherwise accounted for at
all; see that function's docstring for the measured effect.

THREE ADDITIONAL FIXES (2026-09-02, prompted by the user directly tracing
a real 8-tactician game via board_state.json and finding all three by
eye - see each function's docstring for detail):

  - _ranked_expansion_targets used to rank purely by hex_distance to
    capital, with zero awareness of what a site would actually produce -
    a desert hex with no mountain/lake neighbor (heuristic_agent's
    _resource_bonus, reused here) generates NOTHING per turn (see
    engine/collect.py's _outpost_resource) and can still cost a unit to
    even stand on, but ranked identically to a wood/iron/fish-producing
    neighbor at the same distance. Now folds that same resource score in
    as a soft (not hard) adjustment - see EXPANSION_RESOURCE_WEIGHT.
  - _best_direction's tie-break (when multiple legal directions make
    equal progress) used to fall through to Python's min(), which always
    picks the lowest-numbered direction in CUBE_DIRECTIONS - a FIXED
    geometric direction, the same one every single time a tie occurs,
    anywhere on the board, every game. That's what the user was seeing
    as "they all march up and to the left": a real, systematic bias with
    nothing to do with actual position or value, just tuple comparison
    order. Replaced with a cheap position-varying hash so ties no longer
    all resolve the same way (see _direction_tiebreak).
  - marsh was only ever avoided as a tie-break among otherwise-equal
    options, never worth an actual detour - so an army whose single best
    direction happened to be the only one entering a marsh took it
    anyway, forfeiting every remaining step of that phase (marsh freezes
    on entry, not just at day's end, unlike desert) even when a
    1-hex-worse direction would've dodged it entirely. _best_direction
    now takes an optional steps_remaining and, only when stepping into
    marsh is otherwise the pick AND steps would actually be lost by it,
    accepts a small detour instead (MARSH_DETOUR_TOLERANCE) - see
    vanguard_move/marshal_move/sentinel_move for how steps_remaining
    gets threaded in from MOVEMENT_STEPS/CAVALRY_STEPS.
"""

import random

import numpy as np

from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import _resource_bonus, heuristic_target
from .random_agent import random_rectification
from engine.buy import OUTPOST_CAP, eligible_outpost_mask, _outpost_count
from engine.geometry import hex_distance
from engine.state import IMPASSABLE_BY_TERRAIN, NO_FACTION, TERRAIN_TO_INDEX
from engine.turn import CAVALRY_STEPS, MOVEMENT_STEPS

_DESERT_INDEX = TERRAIN_TO_INDEX["desert"]
_MARSH_INDEX = TERRAIN_TO_INDEX["marsh"]

# How many extra hexes of distance a detour around marsh is allowed to
# cost - see _best_direction's steps_remaining handling.
MARSH_DETOUR_TOLERANCE = 1


def _direction_tiebreak(origin, dest, turn_number):
    """A cheap position-varying number, NOT hex_distance-correlated and
    NOT a fixed direction index - used as _best_direction's last-resort
    tiebreak so equally-good directions don't all resolve toward the
    same fixed geometric direction (see that function's docstring for
    why the old plain min()-over-direction-index tiebreak was a real,
    systematic bug, not a harmless coin flip)."""
    return (origin * 92_821 + dest * 68_917 + int(turn_number) * 4_241) % 1_000_003


def _best_direction(state, grid, origin, legal_dirs, target, steps_remaining=0):
    """Whichever of `legal_dirs` makes the most progress toward `target`
    (hex-distance, same as ever) - tie-broken away from two terrain
    rules neither this module's nor marshal_agent's pathfinding
    otherwise knew about at all:
      - desert: "any army that ends its FULL TURN on a desert tile
        loses 1 unit" (see engine/terrain.py's apply_terrain_effects) -
        only the tile an army is STANDING on at the end of the whole
        turn matters, so merely passing through mid-turn (to keep
        moving on a later step) is unaffected.
      - marsh: entering one freezes the army for the rest of THAT turn
        (thaws again next turn - see engine/terrain.py's own unfreeze
        step) - unlike desert this costs something the moment you step
        in, whether or not it's your last move of the turn, since it
        cuts off whatever steps would've followed.
    Confirmed via direct game traces the desert case alone was firing
    ~12 times/game under marshal_agent play, entirely unnoticed; a
    version that only breaks ties this way (never sacrifices actual
    progress to dodge either) tested as a small, real win for desert
    alone (150+200 games, marshal_agent-vs-marshal_agent: 16.0% then
    14.5%, both above the ~12.5% baseline) - it only ever matters on
    steps that would otherwise be a coin flip anyway, so there's no
    real downside to preferring the sturdier option.

    Any remaining tie is broken by _direction_tiebreak rather than
    direction index (see module docstring - the index-order tiebreak
    was a real, observed bug, not a don't-care).

    steps_remaining (how many FURTHER steps this phase has after this
    one - 0 by default, meaning "never detour, tie-break only"): if the
    single best direction by progress would enter marsh, and at least
    one further step would otherwise happen this phase (so freezing
    actually costs something), a direction that avoids marsh is taken
    instead as long as it doesn't cost more than MARSH_DETOUR_TOLERANCE
    extra hexes of distance - see vanguard_move/marshal_move/
    sentinel_move for how this gets threaded in from MOVEMENT_STEPS/
    CAVALRY_STEPS."""
    def score(d):
        dest = int(grid.neighbor_table[origin, d])
        dist = hex_distance(grid.coord_of(dest), target)
        terrain = int(state.terrain[dest])
        lands_on_bad_desert = terrain == _DESERT_INDEX and state.city_owner[dest] == NO_FACTION
        lands_on_marsh = terrain == _MARSH_INDEX
        tiebreak = _direction_tiebreak(origin, dest, state.turn_number)
        return (dist, 1 if lands_on_bad_desert else 0, 1 if lands_on_marsh else 0, tiebreak)

    best = int(min(legal_dirs, key=score))
    if steps_remaining <= 0:
        return best

    best_dist, _, best_marsh, _ = score(best)
    if not best_marsh:
        return best

    non_marsh = [d for d in legal_dirs if not score(d)[2]]
    if not non_marsh:
        return best
    alt = int(min(non_marsh, key=score))
    alt_dist = score(alt)[0]
    return alt if alt_dist <= best_dist + MARSH_DETOUR_TOLERANCE else best

# How many of the nearest legal expansion sites / enemy outposts to track
# in parallel. Swept 2..200 in testing (win rate vs greedy climbed from
# 26% at (2,2) to 35% by (30,15), then EXACTLY plateaued - identical
# win rate, VP, and rank - all the way out to (200,100)): a ~169-hex
# board simply doesn't have more than about 30 eligible expansion sites
# or 15 enemy outposts worth distinguishing at once, so anything at or
# above that already sees the whole relevant board. Set noticeably above
# the observed plateau rather than exactly at it, as a safety margin for
# a bigger board (see tournament.py's SIZE_SWEEP) without materially
# changing behavior on the default one.
EXPANSION_OBJECTIVES = 50
ATTACK_OBJECTIVES = 25

# How strongly a site's resource value (heuristic_agent's _resource_bonus:
# +1/+2 for mountain/lake adjacency, -2 for a desert with neither) shifts
# its rank versus pure distance - weight 1.0 mirrors heuristic_agent's own
# EXPANSION_TOLERANCE=2 design (a site up to ~2 hexes farther is worth it
# for the richest bonus, one up to ~2 hexes closer isn't worth it for the
# worst penalty), just expressed as a continuous score instead of a hard
# tolerance band, since this module ranks a whole pool of targets at once
# rather than picking a single best one.
EXPANSION_RESOURCE_WEIGHT = 1.0


def _ranked_expansion_targets(state, faction, k=None):
    """Up to k (module constant EXPANSION_OBJECTIVES if not given - read
    at CALL time, not bound as a stale default, so tests/tuning can
    monkeypatch the module constant and see it take effect) of the
    best legal outpost sites, nearest-and-richest first - a wider net
    than heuristic_agent's _best_expansion_target (which returns only
    the single best), so several can be pursued by different armies at
    once. Ranked by distance to capital adjusted by each site's resource
    value (see EXPANSION_RESOURCE_WEIGHT) rather than raw distance alone
    - a zero-resource desert hex used to rank identically to a
    wood/iron/fish-producing one at the same distance, with nothing
    stopping it from being the one an army actually settles (see module
    docstring)."""
    if k is None:
        k = EXPANSION_OBJECTIVES
    if _outpost_count(state, faction) >= OUTPOST_CAP:
        return []
    grid = state.grid
    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital) == 0:
        return []
    capital_coord = grid.coord_of(int(own_capital[0]))

    eligible_mask = eligible_outpost_mask(state, faction) & ~IMPASSABLE_BY_TERRAIN[state.terrain]
    eligible = np.nonzero(eligible_mask)[0].tolist()
    if not eligible:
        return []

    def rank_score(i):
        dist = hex_distance(grid.coord_of(i), capital_coord)
        return dist - EXPANSION_RESOURCE_WEIGHT * _resource_bonus(state, i)

    nearest = sorted(eligible, key=rank_score)[:k]
    return [grid.coord_of(i) for i in nearest]


def _ranked_attack_targets(state, faction, k=None):
    """Up to k (module constant ATTACK_OBJECTIVES if not given - see
    _ranked_expansion_targets' docstring for why that's read at call
    time) of the nearest-to-capital enemy outposts, nearest first -
    empty if this faction has no capital on record or none exist yet
    (see _all_targets for the enemy-capital fallback in that case)."""
    if k is None:
        k = ATTACK_OBJECTIVES
    grid = state.grid
    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital) == 0:
        return []
    capital_coord = grid.coord_of(int(own_capital[0]))

    outposts = np.nonzero((state.city_owner != NO_FACTION) & (state.city_owner != faction) & ~state.is_capital)[0]
    if len(outposts) == 0:
        return []
    nearest = sorted(outposts, key=lambda o: hex_distance(grid.coord_of(int(o)), capital_coord))[:k]
    return [grid.coord_of(int(o)) for o in nearest]


def _all_targets(state, faction):
    """Every currently-worthwhile coordinate to have SOME army walking
    toward - the nearest few legal expansion sites plus the nearest few
    enemy outposts (see _ranked_expansion_targets/_ranked_attack_targets),
    or every enemy capital as a last resort if both of those are empty
    (same fallback greedy_agent's greedy_rush_move uses, for the same
    reason - see heuristic_agent's _best_attack_target docstring)."""
    targets = _ranked_expansion_targets(state, faction) + _ranked_attack_targets(state, faction)
    if targets:
        return targets

    grid = state.grid
    capitals = np.nonzero(state.is_capital & (state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    return [grid.coord_of(int(c)) for c in capitals]


def vanguard_move(state, faction, step, legal_mask, total_steps=1):
    """Cycles through ARMIES by `step`, not through objectives directly -
    see module docstring for why an earlier version (cycling objectives,
    each claimed by whichever army was closest to IT) didn't actually
    fix the starvation it was built to fix: the nearest-k objectives
    tend to cluster in whatever single direction currently has the most
    opportunity, so the same well-positioned army kept winning most of
    them anyway, while an army parked in a different direction was still
    never anyone's closest match to anything. Explicitly rotating which
    ARMY gets first pick each call guarantees up to min(3, army count)
    distinct armies are considered every regular-movement phase (and
    likewise up to 2 for the cavalry phase), each pursuing whichever of
    the shared target list is nearest to IT specifically.

    total_steps: how many steps this phase has in total (MOVEMENT_STEPS
    or CAVALRY_STEPS - see make_vanguard_agents), used only to compute
    _best_direction's steps_remaining for marsh-detour purposes."""
    grid = state.grid
    mobile = sorted(int(h) for h in np.nonzero(legal_mask.any(axis=1))[0])
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
            continue  # already standing on its own nearest target - leave it for the buy phase
        legal_dirs = np.nonzero(legal_mask[origin])[0]
        if len(legal_dirs) == 0:
            continue
        return origin, _best_direction(state, grid, origin, legal_dirs, target, steps_remaining)
    return None


def make_vanguard_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's; movement is vanguard_move (see
    module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return vanguard_move(state, faction, step, legal_mask, total_steps=MOVEMENT_STEPS)

    def decide_cavalry(state, faction, step, legal_mask):
        return vanguard_move(state, faction, step, legal_mask, total_steps=CAVALRY_STEPS)

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
