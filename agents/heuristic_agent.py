"""
HeuristicAgent for engine: reuses greedy_agent's buy phase, rectification,
and setup-phase (placement/draft/swap/resource-choice) policies as-is (see
that module's docstring for why those are already reasonable defaults),
and replaces only the two places greedy is purely myopic - battle
targeting and where armies go - with threat-aware versions:

  - decide_target (heuristic_target) attacks whichever rival in the fight
    currently has the FEWEST total units, not a uniform-random one -
    finishing off the weakest contributor first shrinks the number of
    rolls landing on you every subsequent round, the "kill the squishy
    target first" logic from any RTS.

  - Attack-phase movement (see _best_attack_target) scores every enemy
    outpost by estimated defender power (current garrison + its 1 free
    structure-defense shot) instead of nearest-only, and attacks
    whichever is weakest-defended - denying an opponent's easiest target
    first, rather than whichever happens to be closest. Falls back to
    greedy's nearest-enemy-capital move if no enemy outpost exists yet
    (see that function's docstring for why - a stricter version that
    required outmatching the target, or that dropped the capital
    fallback entirely on the theory an undefended capital is a wasted
    attack, both tested WORSE than just keeping an army advancing).

  - Home-expansion site scoring (see _score_expansion_site) extends
    greedy's pure nearest-to-capital metric with a resource tie-break
    (bonus for mountain/lake adjacency - the outpost will harvest
    iron/fish once built; penalty for a desert with neither, which
    produces nothing) among every site within EXPANSION_TOLERANCE hexes
    of the single nearest one.

Also tried, and reverted after testing: independent buy-phase tuning
(upgrade priority order, a target archers-vs-cavalry conversion split)
and a resource-richness-aware placement/draft. Neither survived: A/B
sweeps at the default board size found upgrade priority made no
measurable difference (outposts rarely get upgraded before a ~16-turn
game ends) and the archer/cavalry split was noise, while BOTH changes
turned out to actively cost win rate on the larger radius-8 board
specifically (confirmed by isolating each swap back to greedy's own
version one at a time). Reverted rather than kept "just in case" - see
git history if resource-aware placement/draft is worth revisiting with a
design that doesn't regress at scale.

Also tried: smart_buy, an AGENT-side workaround for a genuine engine bug -
engine/buy.py's _apply_one used to deduct a buy_infantry action's gold
BEFORE checking whether the target hex's army was already at the 6-unit
MAX_STACK_SIZE cap, so greedy_buy's "propose gold // INFANTRY_COST
purchases at one hex" pattern silently destroyed gold on every attempt
past the hex's actual remaining room - confirmed to waste 38 of a
faction's starting 50 gold on turn 1 alone (only 6 infantry fit at an
empty capital, greedy_buy proposes 25). A version of THIS agent that
instead capped proposals at each hex's real remaining room, spread
across every recruitable hex, preserved that gold exactly as intended -
and then made no measurable difference (two independent 100+/200-game
A/B runs against plain greedy_buy, everything else held identical, both
came out a couple points BELOW the 12.5% baseline, not above it). Best
working explanation: gold is essentially never this game's binding
constraint past turn 1 - outposts are gated by their 1-action-per-turn
cap, not by affordability, so recovered gold mostly just sits unspent
regardless. Reverted rather than kept for a benefit that didn't show up
in testing at the time.

UPDATE (2026-09-02): the underlying engine bug itself has now been fixed
at the source (engine/buy.py's _apply_one reorders the stack-cap check
before the gold deduction) - prompted by the user directly observing the
waste in a real traced game, not by the win-rate testing above (which
never found a benefit from preserving this gold, and still might not -
see the reasoning above for why gold isn't usually the binding
constraint). Fixed anyway since "spend gold and get nothing for it" is
simply incorrect engine behavior regardless of whether preserving that
gold measurably helps any given agent - every agent in this package now
gets it for free, no agent-side cap-awareness needed.

KNOWN LIMITATION: even after that revert, this agent (and denier_agent,
which is built on top of it) tests clearly ABOVE greedy at the default
board size (radius 7, 8 factions - see tournament.py's DEFAULT_SIZE) but
BELOW it on the larger radius-8/8-faction board, across multiple
isolation attempts (stripping EXPANSION_TOLERANCE/ATTACK_TOLERANCE to 0,
swapping _best_expansion_target for greedy's own function outright, and
the buy/placement revert above) that each ruled out one candidate cause
without closing the gap. A 20-seed all-greedy control (same faction
count, only board size varied) measured radius-8 games ending ~9%
sooner on average (14.9 vs 16.4 turns) than radius-7 ones - despite the
bigger board, 8 factions have less territory to contest per capital, so
outposts go up with less friction and VP accrues faster - which tracks
with a real, if unconfirmed, explanation: less time in the game for any
deviation from pure-speed-optimized play to earn back what it costs.
Not root-caused further than that; worth revisiting.
"""

import random

import numpy as np

from .greedy_agent import _move_toward, greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .random_agent import random_rectification
from engine.battle import faction_totals, get_legal_target_actions
from engine.buy import OUTPOST_CAP, eligible_outpost_mask, _outpost_count
from engine.geometry import hex_distance
from engine.state import NO_FACTION, IMPASSABLE_BY_TERRAIN, TERRAIN_TO_INDEX

MOUNTAIN_INDEX = TERRAIN_TO_INDEX["mountain"]
LAKE_INDEX = TERRAIN_TO_INDEX["lake"]
DESERT_INDEX = TERRAIN_TO_INDEX["desert"]

# Rough relative combat value per unit type - archers/cavalry outvalue a
# plain infantry unit thanks to their battle abilities (a free pre-round
# kill roll; a chance to dismount into a fresh infantry instead of just
# dying). Only used to compare stack strength for threat/target scoring,
# never for the actual battle math (which the engine resolves exactly -
# see engine/battle.py).
_UNIT_POWER = np.array([1.0, 1.2, 1.3])

EXPANSION_TOLERANCE = 2       # hexes of extra distance a richer expansion site is allowed to cost
                              # over the single nearest one - see _score_expansion_site's docstring
                              # for why this is a distance-dominated tie-break, not a free-form score
ATTACK_TOLERANCE = 2         # same idea for _best_attack_target: how many extra hexes a weaker-
                              # defended outpost is allowed to cost over the single nearest one
OUTPOST_DEFENSE_POWER = 0.5  # ~expected kills from an outpost's 1 free defense shot (11-20 on a d20)


def _power(units):
    return float(np.dot(units, _UNIT_POWER))


def heuristic_target(state, hex_index, faction):
    """Attacks whichever rival currently in the fight has the fewest
    total units - see module docstring."""
    legal = get_legal_target_actions(state, hex_index, faction)
    if not legal:
        return None
    totals = faction_totals(state, hex_index)
    return min(legal, key=lambda f: int(totals[f].sum()))


def _resource_bonus(state, hex_index):
    """+1 per mountain/lake neighbor the outpost would harvest (capped at
    2 total, matching the rule that an outpost only ever produces ONE
    resource regardless of how many qualifying neighbors it has - two
    neighbors just means a genuine choice, not double value), or a
    penalty if this hex is a desert with neither (produces nothing at
    all)."""
    grid = state.grid
    neighbors = grid.neighbor_table[hex_index]
    terrain = state.terrain
    has_mountain = any(j != -1 and terrain[j] == MOUNTAIN_INDEX for j in neighbors)
    has_lake = any(j != -1 and terrain[j] == LAKE_INDEX for j in neighbors)
    if has_mountain and has_lake:
        return 2.0
    if has_mountain or has_lake:
        return 1.0
    if int(terrain[hex_index]) == DESERT_INDEX:
        return -2.0
    return 0.0


def _score_expansion_site(state, hex_index, distance):
    """Distance (dominant - see _best_expansion_target) plus a resource
    tie-break. Deliberately NOT a function of nearby enemy army strength:
    an earlier version scored a "danger" term (nearby enemy army power)
    in here too, but that's recomputed fresh every call as enemy armies
    wander around the map, so the chosen "best" site could flip between
    calls even though nothing about OUR position changed - the target
    army would get redirected mid-walk, over and over, and in testing
    that thrashing left outpost count flatlined for many turns straight
    despite a growing army and plenty of eligible hexes (see
    _best_attack_target's docstring for the same instability in
    attack-target choice, which is tolerated there since combat state
    genuinely changing turn to turn is real signal, not noise). Distance
    and terrain are both static, so scoring on those alone is stable by
    construction."""
    resource = _resource_bonus(state, hex_index)
    return -distance + resource


def _best_expansion_target(state, faction):
    """Like greedy_agent's _home_expansion_target, but breaks ties among
    every site within EXPANSION_TOLERANCE hexes of the single nearest one
    by danger/resources (see _score_expansion_site) instead of always
    taking the strictly-nearest hex. VP accrues per round an outpost has
    stood, so early games are a tempo race - a safer or richer site only
    gets chosen over the nearest one if it doesn't cost more than a
    couple of extra hexes of travel; scoring every eligible hex on the
    board by distance alone (no tolerance band) regressed hard below
    greedy in testing, purely from landing the first outpost 1-2 turns
    later."""
    if _outpost_count(state, faction) >= OUTPOST_CAP:
        return None
    grid = state.grid
    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital) == 0:
        return None
    capital_index = int(own_capital[0])

    eligible_mask = eligible_outpost_mask(state, faction) & ~IMPASSABLE_BY_TERRAIN[state.terrain]
    eligible = np.nonzero(eligible_mask)[0]
    if len(eligible) == 0:
        return None
    dist_arr = np.abs(grid.coords_array[eligible] - grid.coords_array[capital_index]).max(axis=1)
    distances = dict(zip(eligible.tolist(), dist_arr.tolist()))
    min_dist = int(dist_arr.min())
    candidates = [i for i, d in distances.items() if d <= min_dist + EXPANSION_TOLERANCE]
    best = max(candidates, key=lambda i: _score_expansion_site(state, i, distances[i]))
    return grid.coord_of(best)


def _best_attack_target(state, faction):
    """Nearest enemy outpost to our biggest mobile army, tie-broken (within
    ATTACK_TOLERANCE hexes of that nearest one) by weakest-defended -
    same distance-dominates-with-a-tie-break shape as
    _best_expansion_target, for the same reason: an earlier version
    picked the single weakest-defended outpost on the WHOLE board with
    no regard for distance, and tested worse than greedy's pure-nearest
    fallback - chasing a far-off soft target costs tempo that chipping
    away at a closer, only slightly tougher one doesn't. Falls back to
    every enemy capital (nearest wins, via _move_toward) if no enemy
    outpost exists yet - same fallback greedy_agent's greedy_rush_move
    uses; dropping it entirely (on the theory that an undefended
    capital, the common case, nets zero kill-XP while still eating 2
    free defense shots) also tested worse, since that fallback is only
    ever reachable before anyone's built an outpost - too rare a case to
    be worth an army sitting idle over. Returns a target-hex list for
    _move_toward, or None if this faction has no mobile army, or there's
    truly nothing to attack."""
    grid = state.grid
    origins = np.nonzero((state.army_faction == faction) & ~state.locked)[0]
    if len(origins) == 0:
        return None
    sizes = state.army_units[origins].sum(axis=1)
    ref_coord = grid.coord_of(int(origins[int(np.argmax(sizes))]))

    outposts = np.nonzero((state.city_owner != NO_FACTION) & (state.city_owner != faction) & ~state.is_capital)[0]
    if len(outposts) > 0:
        distances = {int(o): hex_distance(ref_coord, grid.coord_of(int(o))) for o in outposts}
        min_dist = min(distances.values())
        candidates = [o for o, d in distances.items() if d <= min_dist + ATTACK_TOLERANCE]
        best = min(candidates, key=lambda o: _power(state.army_units[o]) + OUTPOST_DEFENSE_POWER)
        return [grid.coord_of(best)]

    capitals = np.nonzero(state.is_capital & (state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    if len(capitals) == 0:
        return None
    return [grid.coord_of(int(c)) for c in capitals]


def heuristic_move(state, faction, legal_mask):
    """Tried two variants of "leave a garrison at our own outposts" here
    (never move an army off one; de-prioritize but still allow it as a
    last resort) - both tested clearly worse than not bothering at all
    (0% and 7% win rate vs. ~12% for the plain version below), because
    the engine's fixed action shape can't express "leave 1 unit behind,
    move the rest": a regular-movement action always relocates a hex's
    WHOLE stack, so this function has no way to tell "a deliberate
    garrison" apart from "a unit that's merely passing through one of
    our own outposts on its way somewhere else, and happened to stop
    there for the turn" - the two look identical from state alone, and
    penalizing the latter to protect the former turned out to cost more
    than the extra outpost survivals were worth. Left un-implemented
    rather than resurrected with more tuning - see this function's git
    history if that trade-off is worth revisiting with actual per-hex
    memory (e.g. only garrison a hex that was JUST founded this turn)."""
    grid = state.grid
    origins = np.nonzero(legal_mask.any(axis=1))[0]
    if len(origins) == 0:
        return None

    sizes = state.army_units[origins].sum(axis=1)
    ranked = [int(origins[i]) for i in np.argsort(-sizes)]

    home_target = _best_expansion_target(state, faction)
    if home_target is not None:
        move = _move_toward(grid, ranked, legal_mask, [home_target], skip_arrived=True)
        if move is not None:
            return move
        # every mobile army already parked at the target (or none can
        # legally step toward it) - fall through to attacking instead of
        # doing nothing while expansion is still theoretically available

    attack_targets = _best_attack_target(state, faction)
    if attack_targets is None:
        return None
    return _move_toward(grid, ranked, legal_mask, attack_targets)


def make_heuristic_agents(num_factions, seed=0):
    """Same 9-callback shape as make_greedy_agents/make_random_agents.
    Buy/rectification/resource-choice/placement/draft/swap are greedy's
    as-is; movement and battle targeting are this module's (see
    docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return heuristic_move(state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return heuristic_move(state, faction, legal_mask)

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
