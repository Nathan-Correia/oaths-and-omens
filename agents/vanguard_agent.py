"""
VanguardAgent for engine: same buy phase, battle targeting, rectification,
and setup-phase policies as greedy_agent/heuristic_agent - the change
here is entirely in movement, aimed at a structural inefficiency none of
this package's other agents address.

THE PROBLEM (diagnosed via tournament.py + direct game traces, not
theory): every other agent's movement computes ONE shared objective per
call and hands it to _move_toward, which tries candidate armies in
size-ranked order and returns the FIRST one with any legal step toward
it - so the single largest mobile army wins EVERY movement-phase step,
every turn, for as long as it has any legal move at all, regardless of
how many OTHER armies exist. A faction typically accumulates 4-6
separate armies by mid-game, and individual armies can sit at the exact
same hex for many consecutive turns without ever being assigned anything.

THE FIX (see vanguard_move): rotate which ARMY gets first pick each
step, not which objective does. Explicitly indexing into the mobile-army
list by `step` guarantees up to min(3, army count) distinct armies are
considered every regular-movement phase (likewise up to 2 for the
cavalry phase), each pursuing whichever of a shared target pool (see
_all_targets) happens to be nearest to IT specifically.

Direction selection (_best_direction, shared with marshal_agent) tie-
breaks away from ending a step on a desert tile with no city on it, uses
a position-varying tiebreak instead of always resolving toward the same
fixed geometric direction (see _direction_tiebreak), and - when
steps_remaining says a further step would otherwise happen this phase -
accepts a small detour to avoid marsh (which freezes on entry, not just
at day's end) rather than only tie-breaking away from it.

_ranked_expansion_targets ranks candidate outpost sites by distance to
capital adjusted by each site's resource value (heuristic_agent's
_resource_bonus), not raw distance alone - a zero-resource desert hex
used to rank identically to a wood/iron/fish-producing one at the same
distance.
"""

import random

import torch

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
    same fixed geometric direction."""
    return (origin * 92_821 + dest * 68_917 + int(turn_number) * 4_241) % 1_000_003


def _best_direction(state, grid, origin, legal_dirs, target, steps_remaining=0):
    """Whichever of `legal_dirs` makes the most progress toward `target`
    (hex-distance) - tie-broken away from two terrain rules:
      - desert: any army that ends its FULL TURN on a desert tile loses
        1 unit - only the tile an army is STANDING on at the end of the
        whole turn matters, so merely passing through mid-turn is
        unaffected.
      - marsh: entering one freezes the army for the rest of THAT turn -
        unlike desert this costs something the moment you step in.
    Any remaining tie is broken by _direction_tiebreak rather than
    direction index. steps_remaining (0 by default, meaning "never
    detour, tie-break only"): if the single best direction by progress
    would enter marsh, and at least one further step would otherwise
    happen this phase, a direction that avoids marsh is taken instead as
    long as it doesn't cost more than MARSH_DETOUR_TOLERANCE extra hexes
    of distance."""
    def score(d):
        dest = int(grid.neighbor_table[origin, d])
        dist = hex_distance(grid.coord_of(dest), target)
        terrain = int(state.terrain[0, dest])
        lands_on_bad_desert = terrain == _DESERT_INDEX and int(state.city_owner[0, dest]) == NO_FACTION
        lands_on_marsh = terrain == _MARSH_INDEX
        tiebreak = _direction_tiebreak(origin, dest, int(state.turn_number[0]))
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
# in parallel. Swept 2..200 in testing: win rate vs greedy climbed then
# EXACTLY plateaued - a ~169-hex board simply doesn't have more than
# about 30 eligible expansion sites or 15 enemy outposts worth
# distinguishing at once. Set noticeably above the observed plateau as a
# safety margin for a bigger board.
EXPANSION_OBJECTIVES = 50
ATTACK_OBJECTIVES = 25

# How strongly a site's resource value shifts its rank versus pure
# distance - weight 1.0 mirrors heuristic_agent's own EXPANSION_TOLERANCE=2
# design, just expressed as a continuous score instead of a hard
# tolerance band, since this module ranks a whole pool of targets at once
# rather than picking a single best one.
EXPANSION_RESOURCE_WEIGHT = 1.0


def _ranked_expansion_targets(state, faction, k=None):
    """Up to k (module constant EXPANSION_OBJECTIVES if not given - read
    at CALL time, so tests/tuning can monkeypatch the module constant)
    of the best legal outpost sites, nearest-and-richest first - a wider
    net than heuristic_agent's _best_expansion_target, so several can be
    pursued by different armies at once."""
    if k is None:
        k = EXPANSION_OBJECTIVES
    if int(_outpost_count(state, faction)[0]) >= OUTPOST_CAP:
        return []
    grid = state.grid
    own_capital = torch.nonzero((state.city_owner[0] == faction) & state.is_capital[0], as_tuple=False).flatten()
    if len(own_capital) == 0:
        return []
    capital_coord = grid.coord_of(int(own_capital[0]))

    eligible_mask = eligible_outpost_mask(state, faction)[0] & ~IMPASSABLE_BY_TERRAIN.to(state.device)[state.terrain[0].long()]
    eligible = torch.nonzero(eligible_mask, as_tuple=False).flatten().tolist()
    if not eligible:
        return []

    def rank_score(i):
        dist = hex_distance(grid.coord_of(i), capital_coord)
        return dist - EXPANSION_RESOURCE_WEIGHT * _resource_bonus(state, i)

    nearest = sorted(eligible, key=rank_score)[:k]
    return [grid.coord_of(i) for i in nearest]


def _ranked_attack_targets(state, faction, k=None):
    """Up to k (module constant ATTACK_OBJECTIVES if not given) of the
    nearest-to-capital enemy outposts, nearest first - empty if this
    faction has no capital on record or none exist yet."""
    if k is None:
        k = ATTACK_OBJECTIVES
    grid = state.grid
    own_capital = torch.nonzero((state.city_owner[0] == faction) & state.is_capital[0], as_tuple=False).flatten()
    if len(own_capital) == 0:
        return []
    capital_coord = grid.coord_of(int(own_capital[0]))

    outposts = torch.nonzero(
        (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction) & ~state.is_capital[0], as_tuple=False
    ).flatten().tolist()
    if not outposts:
        return []
    nearest = sorted(outposts, key=lambda o: hex_distance(grid.coord_of(o), capital_coord))[:k]
    return [grid.coord_of(o) for o in nearest]


def _all_targets(state, faction):
    """Every currently-worthwhile coordinate to have SOME army walking
    toward - the nearest few legal expansion sites plus the nearest few
    enemy outposts, or every enemy capital as a last resort if both of
    those are empty."""
    targets = _ranked_expansion_targets(state, faction) + _ranked_attack_targets(state, faction)
    if targets:
        return targets

    grid = state.grid
    capitals = torch.nonzero(
        state.is_capital[0] & (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction), as_tuple=False
    ).flatten().tolist()
    return [grid.coord_of(c) for c in capitals]


def vanguard_move(state, faction, step, legal_mask, total_steps=1):
    """Cycles through ARMIES by `step`, not through objectives directly.
    total_steps: how many steps this phase has in total (MOVEMENT_STEPS
    or CAVALRY_STEPS), used only to compute _best_direction's
    steps_remaining for marsh-detour purposes."""
    grid = state.grid
    mobile = sorted(int(h) for h in torch.nonzero(legal_mask.any(dim=1), as_tuple=False).flatten().tolist())
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
        legal_dirs = torch.nonzero(legal_mask[origin], as_tuple=False).flatten().tolist()
        if not legal_dirs:
            continue
        return origin, _best_direction(state, grid, origin, legal_dirs, target, steps_remaining)
    return None


def make_vanguard_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's; movement is vanguard_move."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

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
