"""
TacticianAgent for engine: marshal_agent's buy phase, battle targeting,
rectification, setup-phase policies, and matching/direction logic as the
FALLBACK for most decisions - but the very first regular-movement
decision of each turn (step 0) is chosen by genuine forward search
instead of a hand-tuned heuristic.

HOW: clone the actual ArrayState (see ArrayState.clone in state.py), try
each of marshal_agent's candidate (origin, target) matches as OUR real
move for this step, then literally PLAY OUT the rest of that turn on the
clone - remaining movement/cavalry steps, the battle phase with real
dice, terrain effects, collect - using marshal_move for our own
subsequent decisions and a fixed opponent model for the other factions,
then score the resulting position (see _evaluate). Whichever candidate
led to the best evaluated outcome is the real move actually taken.

BATCHED-ENGINE NOTE: the rollout clone is still a single-game (B=1)
ArrayState - this agent's search happens entirely within one game's
per-game decision, same as every other agent (see engine/turn.py's
module docstring: agents stay per-game Python functions even though the
engine itself is batched). Calling the engine's batched phase functions
(apply_movement_step, _run_battle_phase, apply_collect_phase) on a B=1
clone works the same as it does for the real B=1-or-more game; only the
decide_target/decide_rectification/decide_resource_choice callbacks need
wrapping in a length-1 list, matching those functions' decide_*_list
convention.

OPPONENT MODEL: random_agent's policy, not greedy_agent's - the weaker,
cheaper model won on both performance and cost (see prior test notes).

COST: expensive enough to only apply at one decision point per turn -
see MAX_CANDIDATES. Steps 1-2 of the regular phase and both cavalry
steps fall back to plain marshal_move (no additional search).

RESULT (pre-batching-rewrite numbers, not yet re-validated post-rewrite -
see the plan's RNG section for why fresh numbers are needed rather than
compared against these): 40-58% win rate vs greedy_agent across several
samples, well above marshal_agent's own ~33-38%, generalizing across
board sizes. Head-to-head against marshal_agent specifically: a smaller
but real edge, roughly 12-22% over several samples against the ~12.5%
no-effect baseline.

SEARCH EXPANSIONS TRIED AND NOT ADOPTED (see git history/prior session
notes for full detail): an explicit "do nothing" candidate (no effect);
extending the same search to cavalry-phase step 0 (a regression);
deepening the rollout to cover next turn too (marginal edge, not worth
~3x the cost); Hungarian-algorithm matching in place of marshal_agent's
greedy match (a regression); searching the buy phase the same way (a
regression, and expensive); averaging rollout score across two opponent
models (no improvement, ~2x cost); sweeping every hand-picked constant
in this module and vanguard_agent.py's shared helpers (nothing beat the
shipped defaults at real sample sizes). Net: the shipped design (one
decision point, one turn of lookahead, one opponent model) has held up
as a local optimum across a wide range of things tried against it.
"""

import random

import torch

from .greedy_agent import greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .marshal_agent import _greedy_match, make_marshal_agents, marshal_move
from .random_agent import make_random_agents, random_rectification
from .vanguard_agent import _all_targets, _best_direction
from engine.collect import apply_collect_phase
from engine.geometry import hex_distance
from engine.movement import actions_from_dicts, apply_movement_step, legal_cavalry_mask, legal_movement_mask
from engine.terrain import apply_terrain_effects
from engine.turn import CAVALRY_STEPS, MOVEMENT_STEPS, _run_battle_phase

# Cap on how many of marshal_agent's matched pairs get their own rollout,
# for cost - see prior sweep notes (git history): 10 and 16 tied exactly
# on this board size (marshal_agent's matching rarely produces more than
# ~10 candidates anyway); 10 is the default as the smaller of the two.
MAX_CANDIDATES = 10

# Relative unit values for the army-power term below - same weights as
# vanguard_agent/heuristic_agent's threat-scoring.
_UNIT_POWER = torch.tensor([1.0, 1.2, 1.3])


def _army_power(state, faction):
    board = state.army_units[0][state.army_faction[0] == faction]
    battle = state.battle_units[0][state.battle_faction[0] == faction]
    total = torch.zeros(3, device=state.device)
    if len(board):
        total = total + board.sum(dim=0).float()
    if len(battle):
        total = total + battle.sum(dim=0).float()
    return float((total * _UNIT_POWER.to(total.device)).sum())


# _evaluate's term weights - never tuned before being shipped as the
# first guess below; a sweep against greedy_agent found nothing that
# beat these at real sample sizes (see module docstring).
RIVAL_WEIGHT = 0.5
OUTPOST_WEIGHT = 2.0
ARMY_POWER_WEIGHT = 0.05


def _evaluate(state, faction):
    """Higher is better for `faction`. Own VP dominates (it's literally
    the win condition), with a penalty for whoever's currently the
    biggest rival, plus smaller terms for outpost count (the thing that
    generates future VP) and remaining army strength (a proxy for how
    defensible this position is going forward)."""
    own_vp = int(state.victory_points[0, faction])
    rival_vp = [int(state.victory_points[0, f]) for f in range(state.num_factions) if f != faction]
    best_rival = max(rival_vp) if rival_vp else 0
    own_outposts = int(((state.city_owner[0] == faction) & ~state.is_capital[0]).sum())
    return (
        own_vp - RIVAL_WEIGHT * best_rival + OUTPOST_WEIGHT * own_outposts
        + ARMY_POWER_WEIGHT * _army_power(state, faction)
    )


def _rollout_and_score(state, faction, first_action, my_decide, opp_decide, gen):
    """Clones `state` (ArrayState.clone - see state.py), applies
    `first_action` as `faction`'s move for the CURRENT regular-movement
    step (opponents use their own decide_movement for this same step),
    then plays out everything remaining in the turn - see module
    docstring. `my_decide`/`opp_decide`: 9-tuples in the same shape
    make_X_agents returns, for this faction and every other faction
    respectively. `gen`: a torch.Generator on the same device as
    `state`, consumed by the engine's own randomness (movement swap
    ties, battle dice) - separate from any Python-level random.Random
    the wrapped agent callbacks use for their own decisions. Returns
    _evaluate(...) of the resulting state."""
    sim = state.clone()
    F = sim.num_factions

    actions = {faction: first_action}
    for f in range(F):
        if f != faction:
            lm = legal_movement_mask(sim, f)[0]
            actions[f] = opp_decide[1][f](sim, f, 0, lm)
    from_hex, direction, has_action = actions_from_dicts([actions], F, sim.device)
    apply_movement_step(sim, from_hex, direction, has_action, gen, cavalry_only=False)

    for step in range(1, MOVEMENT_STEPS):
        actions = {}
        for f in range(F):
            lm = legal_movement_mask(sim, f)[0]
            fn = my_decide[1][f] if f == faction else opp_decide[1][f]
            actions[f] = fn(sim, f, step, lm)
        from_hex, direction, has_action = actions_from_dicts([actions], F, sim.device)
        apply_movement_step(sim, from_hex, direction, has_action, gen, cavalry_only=False)

    for step in range(CAVALRY_STEPS):
        actions = {}
        for f in range(F):
            lm = legal_cavalry_mask(sim, f)[0]
            fn = my_decide[2][f] if f == faction else opp_decide[2][f]
            actions[f] = fn(sim, f, step, lm)
        from_hex, direction, has_action = actions_from_dicts([actions], F, sim.device)
        apply_movement_step(sim, from_hex, direction, has_action, gen, cavalry_only=True)

    decide_target = {f: (my_decide[3][f] if f == faction else opp_decide[3][f]) for f in range(F)}
    decide_rectification = {f: (my_decide[4][f] if f == faction else opp_decide[4][f]) for f in range(F)}
    decide_resource_choice = {f: (my_decide[5][f] if f == faction else opp_decide[5][f]) for f in range(F)}
    _run_battle_phase(sim, [decide_target], [decide_rectification], gen)
    apply_terrain_effects(sim)
    apply_collect_phase(sim, [decide_resource_choice])

    return _evaluate(sim, faction)


def _search_first_move(state, faction, legal_mask, my_decide, opp_decide, rng):
    """The step-0 search itself: builds candidates from marshal_agent's
    matching (same pool it would otherwise rotate through), rolls each
    one out (see _rollout_and_score), and returns whichever scored best
    - or None if there's truly nothing to try. `rng`: this faction's own
    Python-level random.Random (used only to pick one shared seed for
    every candidate's rollout, for a fair apples-to-apples comparison -
    NOT the engine-level randomness inside the rollout itself, see
    _rollout_and_score)."""
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

    candidates = []
    for origin, target in matches[:MAX_CANDIDATES]:
        origin_coord = grid.coord_of(origin)
        if hex_distance(origin_coord, target) == 0:
            continue
        legal_dirs = legal_mask[origin].nonzero(as_tuple=False).flatten().tolist()
        if not legal_dirs:
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
        gen = torch.Generator(device=state.device)
        gen.manual_seed(seed)
        score = _rollout_and_score(state, faction, action, my_decide, opp_decide, gen)
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

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

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
