"""
TacticianAgent for engine: marshal_agent's buy phase, battle targeting,
rectification, setup-phase policies, and matching/direction logic (see
that module's docstring) as the FALLBACK for most decisions - but the
very first regular-movement decision of each turn (step 0, the most
consequential one: it's the first of 3 chances to advance the biggest
piece of that turn's plan) is chosen by genuine forward search instead
of a hand-tuned heuristic.

HOW: clone the actual ArrayState (see _clone_state), try each of
marshal_agent's candidate (origin, target) matches as OUR real move for
this step, then literally PLAY OUT the rest of that turn on the clone -
remaining movement/cavalry steps, the battle phase with real dice,
terrain effects, collect - using marshal_move for our own subsequent
decisions and a fixed opponent model for the other factions, then score
the resulting position (see _evaluate). Whichever candidate led to the
best evaluated outcome is the real move actually taken.

OPPONENT MODEL: random_agent's policy, not greedy_agent's - tried both
(plus marshal_agent's own policy) and the weaker, cheaper model won on
BOTH performance and cost: random-opponent rollouts scored 41-43% vs
greedy_agent (matching or beating the greedy-opponent-model version's
40-58%) and a clearly better 21.7% head-to-head against marshal_agent
directly (vs. ~14% with greedy opponents), while running about 3x
faster (random_agent skips a move outright half the time, so there's
less state to evolve and less RNG consumed per rollout). The
greedy-opponent version isn't wrong, exactly, but a stronger/more
"realistic" opponent model inside the rollout doesn't make the SEARCH
better - it just makes each rollout more expensive without improving
what it's being used for, which is comparing OUR candidates against
each other on a level playing field, not accurately predicting what any
specific real opponent will do.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE IN THIS PACKAGE: every other
"smarter" idea tried here (see heuristic_agent's, denier_agent's,
marshal_agent's docstrings for the long list) scores a candidate move by
some PROXY computed before anything actually happens - distance,
resource richness, army size, whether a target is currently "claimed"...
- and the pattern across a dozen-plus of those experiments was that
proxies depending on current dynamic state keep backfiring, in ways
that were often only findable by testing (see marshal_agent's overstack-
avoidance and incidental-contact-avoidance results for two that looked
obviously safe on paper and weren't). Actually simulating the
consequences, real battle dice included, sidesteps needing a proxy at
all - the "score" a candidate gets IS what (a plausible version of)
actually happens if it's played.

COST: expensive enough to only apply at one decision point per turn.
Each candidate rollout covers nearly a full turn's worth of engine work
for all `num_factions` factions (movement steps 1-2, both cavalry
steps, the full battle phase, terrain, collect) - with up to
MAX_CANDIDATES rollouts per real turn just for this one decision. Steps
1-2 of the regular phase and both cavalry steps fall back to plain
marshal_move (no additional search) to keep total cost bounded. Even
with the random opponent model (see above), a full game with only ONE
of 8 factions playing this agent (the other 7 cheap) runs at roughly
1s, versus well under 0.1s for an all-cheap-agent game - meaningfully
more expensive than everything else in this package, which is the real
trade-off being made here, not just "is it stronger."

RESULT: yes, clearly, by a wide margin over every hand-tuned heuristic
tried in this package. vs greedy_agent directly: 40-58% win rate across
several samples (a combined ~45% over 140 games with the original
greedy-opponent-model version; 41-43% with the current, cheaper random-
opponent-model version - see above), well above marshal_agent's own
already-strong ~33-38% at the default board size, and it generalizes -
40% at radius 8 (the board size that broke every purely heuristic
agent's edge over greedy, marshal_agent included, down to a much
smaller margin there), 68% at radius 4. Head-to-head against
marshal_agent specifically (its own fallback policy, so this isolates
the search's marginal value on top of an already-strong base) is a
smaller but still real edge - ~14% over 120 combined games with the
original greedy-opponent-model version; with the current random-
opponent-model one, two 60-game samples came back at 21.7% and 11.7%
(a reminder that even 60 games reads as "big edge" or "no edge"
depending on the draw - combined, ~16.7% over 120 games, roughly the
same ballpark as the greedy-opponent version, not a clear improvement
or regression on this specific comparison) - all against the ~12.5%
no-effect baseline. In an 8-agent free-for-all it placed a clear first
(31.2% win rate, best average rank), ahead of marshal_agent in second.

FURTHER SEARCH EXPANSIONS TRIED, THREE THAT DIDN'T EARN THEIR COST:
  - An explicit "do nothing this step" candidate alongside the moves
    from marshal_agent's matching: no effect either direction (11.7%
    vs the plain version, dead on baseline) - moving is essentially
    always at least as good as sitting still, so it almost never wins
    the comparison, meaning it's pure wasted rollout cost.
  - Extending the same search to cavalry-phase step 0 (a second,
    independent decision point per turn, same mechanics): a genuine
    regression, not just noise - 27.5% vs greedy_agent directly
    (clearly below the plain version's 40-58% range) and 3.3% head-to-
    head against the plain version. Candidate counts there are usually
    tiny (often exactly 1, since far fewer armies carry cavalry at
    all - see hussar_agent's docstring for why cavalry stays scarce),
    so there's little for a search to improve on, and reusing the same
    general target pool for a phase that's already downstream of the
    turn's real movement seems to actively confuse the choice rather
    than sharpen it.
  - Deepening the rollout to cover the ENTIRETY of next turn too (buy
    phase included), not just the rest of the current one, using
    ordinary policies (no additional search) for that second turn:
    a small, uncertain edge (14.0% head-to-head against the plain
    one-turn version) for roughly 3x the per-candidate cost - not
    worth it at this margin.
Net: the shipped design (one decision point, one turn of lookahead)
appears to be close to a local optimum for this style of search, at
least among the extensions tried so far.

FOUR MORE NON-SEARCH-DEPTH IDEAS TRIED (2026-09-02), NONE ADOPTED:
  - Hungarian-algorithm (true optimal bipartite) matching in place of
    marshal_agent's _greedy_match, everywhere that function is used
    (this module's candidate generation included): a genuine regression,
    not a wash - 10.7% head-to-head against plain marshal_agent (BELOW
    the ~12.5% no-effect baseline) and 31.3% vs greedy_agent directly
    (below marshal_agent's own established ~40%+ there). Best working
    explanation: minimizing TOTAL assignment distance across every
    army/target pair isn't the same objective as "get something useful
    done this step" - greedy matching's bias toward locking in the
    single cheapest pair first tends to resolve more targets sooner,
    which seems to matter more than global optimality does here.
  - Searching the BUY phase the same way movement step 0 is searched -
    a few candidate buy bundles (greedy_buy's own choice; the same
    minus buy_infantry; the same minus build_outpost/upgrade_outpost),
    each cloned/applied/rolled out through the rest of the turn and
    scored: also a clear regression (6.2% head-to-head against the
    plain version) at ~3-4x the per-game cost. Buy's value mostly pays
    off many turns later (an outpost compounds for the rest of the
    game), so a single one-turn rollout is a very noisy signal for
    comparing buy choices specifically - much noisier than it is for
    comparing movement targets, where the effect of "which hex did my
    army end up closer to" is closer to immediate.
  - Averaging each candidate's rollout score across TWO different
    simulated opponent models (random_agent's and greedy_agent's
    policies) instead of one, hoping for a choice more robust to which
    policy the real opponents turn out to run: no improvement (12.5%
    head-to-head against the plain version, dead on the no-effect
    baseline) at roughly 2x the cost. Consistent with the earlier
    finding (this docstring's RESULT section) that the weaker, cheaper
    single opponent model already outperforms the stronger one - this
    game's opponents apparently don't need modeling accurately enough
    for a mixture to pay for itself.
  - Sweeping every hand-picked constant in this module and
    vanguard_agent.py's shared helpers (RIVAL_WEIGHT, OUTPOST_WEIGHT,
    ARMY_POWER_WEIGHT, MAX_CANDIDATES, MARSH_DETOUR_TOLERANCE,
    EXPANSION_RESOURCE_WEIGHT) one at a time against greedy_agent: at
    n=60 nearly everything looked identical to the shipped defaults
    except MAX_CANDIDATES=6, which looked like a real win (48.3% vs the
    default's 40.0%) - it wasn't. A larger sweep (n=150) put every
    tested value from 3 to 10 in a tight 49-53% band with no clean
    trend, and a direct MAX_CANDIDATES=6-vs-10 head-to-head (n=200) came
    back at 15.0%, only ~1 standard deviation over the ~12.5% no-effect
    baseline - not a real signal either way. Left every constant at its
    existing value rather than chase a difference this small: n=60 is
    genuinely not enough to distinguish these configurations, a useful
    caution for reading any of this package's smaller-sample results.
Taken together with the three items above, every variation tried this
round on "spend the existing rollout budget differently" (more
candidates considered, more opponent models, more phases searched, a
provably-more-optimal matcher feeding the candidate pool) either made
things worse or made no measurable difference at real extra cost - a
much more consistent negative pattern than the three-item list above
saw individually. Suggests the bottleneck now is less "how the search
spends its compute" and more "what _evaluate rewards" or "which
targets/candidates make it into the pool at all" - directions that
change WHAT gets scored rather than how many times.
"""

import random

import numpy as np

from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .marshal_agent import _greedy_match, make_marshal_agents, marshal_move
from .random_agent import make_random_agents, random_rectification
from .vanguard_agent import _all_targets, _best_direction
from engine.collect import apply_collect_phase
from engine.geometry import hex_distance
from engine.movement import apply_movement_step, legal_cavalry_mask, legal_movement_mask
from engine.state import ArrayState
from engine.terrain import apply_terrain_effects
from engine.turn import CAVALRY_STEPS, MOVEMENT_STEPS, _run_battle_phase

# Cap on how many of marshal_agent's matched pairs get their own rollout, for cost. Originally swept
# 3/5/8 under a greedy_agent opponent model (40 games each vs greedy): 5 and 8 tied exactly, 3 measurably
# weaker. Re-swept 5/10/16 after switching the rollout's opponent model to random_agent (see this
# module's docstring - much cheaper per rollout, changing the cost/value tradeoff): 10 and 16 tied exactly
# (same win rate, VP, rank, and wall-clock time - marshal_agent's matching rarely produces more than ~10
# candidates on this board size anyway), both a bit ahead of 5. 10 is the default as the smaller of the
# two identical results.
MAX_CANDIDATES = 10


def _clone_state(state):
    """A deep-enough copy of `state` to simulate on without touching the
    real game: every array that CAN mutate during movement/battle/
    terrain/collect gets its own copy; `grid`, `terrain`, `city_placer`,
    and `capital_settle_order` are shared by reference since nothing in
    that phase range ever writes to them (terrain is fixed at game
    start; city_placer/capital_settle_order are setup-only bookkeeping -
    see state.py's field docstrings)."""
    return ArrayState(
        grid=state.grid,
        terrain=state.terrain,
        city_owner=state.city_owner.copy(),
        is_capital=state.is_capital.copy(),
        outpost_upgrade=state.outpost_upgrade.copy(),
        city_placer=state.city_placer,
        capital_settle_order=state.capital_settle_order,
        army_faction=state.army_faction.copy(),
        army_units=state.army_units.copy(),
        frozen=state.frozen.copy(),
        locked=state.locked.copy(),
        battle_faction=state.battle_faction.copy(),
        battle_origin=state.battle_origin.copy(),
        battle_units=state.battle_units.copy(),
        battle_moved=state.battle_moved.copy(),
        battle_round=state.battle_round.copy(),
        battle_order=list(state.battle_order),
        gold=state.gold.copy(),
        resources=state.resources.copy(),
        kill_xp=state.kill_xp.copy(),
        victory_points=state.victory_points.copy(),
        alive=state.alive.copy(),
        turn_number=state.turn_number,
        num_factions=state.num_factions,
    )


# Relative unit values for the army-power term below - same weights as
# vanguard_agent/heuristic_agent's threat-scoring (archers/cavalry
# outvalue infantry thanks to their battle abilities).
_UNIT_POWER = np.array([1.0, 1.2, 1.3])


def _army_power(state, faction):
    board = state.army_units[state.army_faction == faction]
    battle = state.battle_units[state.battle_faction == faction]
    total = np.zeros(3)
    if len(board):
        total += board.sum(axis=0)
    if len(battle):
        total += battle.sum(axis=0)
    return float(np.dot(total, _UNIT_POWER))


# _evaluate's term weights - never tuned before being shipped as the
# first guess below; see this function's docstring for a sweep.
RIVAL_WEIGHT = 0.5
OUTPOST_WEIGHT = 2.0
ARMY_POWER_WEIGHT = 0.05


def _evaluate(state, faction):
    """Higher is better for `faction`. Own VP dominates (it's literally
    the win condition), with a penalty for whoever's currently the
    biggest rival (denying/staying ahead of the leader matters even
    though chasing them specifically didn't pan out as a movement
    priority - see marshal_agent's leader-targeting result; this is
    just how good OUR position looks, not a targeting decision), plus
    smaller terms for outpost count (the thing that generates future
    VP) and remaining army strength (a proxy for how defensible this
    position is going forward)."""
    own_vp = int(state.victory_points[faction])
    rival_vp = [int(state.victory_points[f]) for f in range(state.num_factions) if f != faction]
    best_rival = max(rival_vp) if rival_vp else 0
    own_outposts = int(np.sum((state.city_owner == faction) & ~state.is_capital))
    return (
        own_vp - RIVAL_WEIGHT * best_rival + OUTPOST_WEIGHT * own_outposts
        + ARMY_POWER_WEIGHT * _army_power(state, faction)
    )


def _rollout_and_score(state, faction, first_action, my_decide, opp_decide, rng):
    """Clones `state`, applies `first_action` as `faction`'s move for
    the CURRENT regular-movement step (opponents use their own
    decide_movement for this same step), then plays out everything
    remaining in the turn - see module docstring. `my_decide`/
    `opp_decide`: 9-tuples in the same shape make_X_agents returns,
    for this faction and every other faction respectively. Returns
    _evaluate(...) of the resulting state."""
    sim = _clone_state(state)

    actions = {faction: first_action}
    for f in range(sim.num_factions):
        if f != faction:
            lm = legal_movement_mask(sim, f)
            actions[f] = opp_decide[1][f](sim, f, 0, lm)
    apply_movement_step(sim, actions, rng, cavalry_only=False)

    for step in range(1, MOVEMENT_STEPS):
        actions = {}
        for f in range(sim.num_factions):
            lm = legal_movement_mask(sim, f)
            fn = my_decide[1][f] if f == faction else opp_decide[1][f]
            actions[f] = fn(sim, f, step, lm)
        apply_movement_step(sim, actions, rng, cavalry_only=False)

    for step in range(CAVALRY_STEPS):
        actions = {}
        for f in range(sim.num_factions):
            lm = legal_cavalry_mask(sim, f)
            fn = my_decide[2][f] if f == faction else opp_decide[2][f]
            actions[f] = fn(sim, f, step, lm)
        apply_movement_step(sim, actions, rng, cavalry_only=True)

    decide_target = {f: (my_decide[3][f] if f == faction else opp_decide[3][f]) for f in range(sim.num_factions)}
    decide_rectification = {
        f: (my_decide[4][f] if f == faction else opp_decide[4][f]) for f in range(sim.num_factions)
    }
    decide_resource_choice = {
        f: (my_decide[5][f] if f == faction else opp_decide[5][f]) for f in range(sim.num_factions)
    }
    _run_battle_phase(sim, decide_target, decide_rectification, rng)
    apply_terrain_effects(sim)
    apply_collect_phase(sim, decide_resource_choice)

    return _evaluate(sim, faction)


def _search_first_move(state, faction, legal_mask, my_decide, opp_decide, rng):
    """The step-0 search itself: builds candidates from marshal_agent's
    matching (same pool it would otherwise rotate through), rolls each
    one out (see _rollout_and_score), and returns whichever scored best
    - or None if there's truly nothing to try."""
    grid = state.grid
    mobile = sorted(int(h) for h in np.nonzero(legal_mask.any(axis=1))[0])
    if not mobile:
        return None
    targets = _all_targets(state, faction)
    if not targets:
        return None
    matches = _greedy_match(grid, mobile, targets)
    if not matches:
        return None

    candidates = []
    for origin, target in matches[:MAX_CANDIDATES]:
        origin_coord = grid.coord_of(origin)
        if hex_distance(origin_coord, target) == 0:
            continue
        legal_dirs = np.nonzero(legal_mask[origin])[0]
        if len(legal_dirs) == 0:
            continue
        direction = _best_direction(state, grid, origin, legal_dirs, target, MOVEMENT_STEPS - 1)
        candidates.append((origin, direction))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    seed = rng.randrange(2 ** 31)  # same seed for every candidate this decision - fair apples-to-apples comparison
    best_action, best_score = None, None
    for action in candidates:
        score = _rollout_and_score(state, faction, action, my_decide, opp_decide, random.Random(seed))
        if best_score is None or score > best_score:
            best_action, best_score = action, score
    return best_action


def make_tactician_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Buy/target/
    rectification/resource-choice/placement/draft/swap match
    greedy_agent's/heuristic_agent's/marshal_agent's; movement/cavalry
    default to marshal_move, except regular-movement step 0, which uses
    _search_first_move (see module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}
    my_decide = make_marshal_agents(num_factions, seed=seed)
    opp_decide = make_random_agents(num_factions, seed=seed + 999_983)

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        if step == 0:
            move = _search_first_move(state, faction, legal_mask, my_decide, opp_decide, rngs[faction])
            if move is not None:
                return move
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
