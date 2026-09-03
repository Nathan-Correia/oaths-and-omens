"""
RandomAgent for engine - uniformly samples from whatever legal actions/
mask the engine already computed. Its value isn't playing well - it's
exercising every code path in the engine (fuzzing) to surface crashes,
illegal states, and rules ambiguities early.

Every callback here operates on a single-game (batch_size=1) state VIEW
- see engine/turn.py's module docstring: agents stay per-game Python
functions even though the engine itself is batched. decide_movement/
decide_cavalry just return one (hex_index, direction) pulled from the
legal mask, or None - engine's action representation only ever moves a
fixed, phase-determined subset of a hex's army, so there's no partial-
split left to sample.

decide_buy's SHAPE changed with the batched rewrite: engine/buy.py no
longer offers a pre-computed list of legal actions to filter (that
variable-length representation is exactly what got replaced - see that
module's docstring) - decide_buy now inspects `state` directly and
returns the fixed-shape buy-decision dict itself. random_buy below picks
uniformly among a few coarse choices (whether to attempt an outpost
action, how many infantry/conversions to request and where) rather than
enumerating every atomic legal action the way the old list-based version
did - still random, still exercises the same code paths, just built
differently now that the underlying action space is.

decide_target/decide_rectification are reused as-is by greedy_agent.py -
neither battle targeting nor rectification needs to be smart.
"""

import random

import torch

from engine.battle import faction_totals, get_legal_target_actions
from engine.buy import OUTPOST_COST, eligible_outpost_mask

SKIP_CHANCE = 0.5  # chance to move nothing this step, matching v1's flavor


def random_movement(rng, legal_mask):
    if rng.random() < SKIP_CHANCE:
        return None
    rows, cols = torch.nonzero(legal_mask, as_tuple=True)
    if len(rows) == 0:
        return None
    i = rng.randrange(len(rows))
    return int(rows[i]), int(cols[i])


def random_target(state, hex_index, faction, rng):
    legal = get_legal_target_actions(state, hex_index, faction)
    if not legal:
        return None
    return rng.choice(legal)


def random_rectification(state, hex_index, winner_faction, cap, rng):
    totals = faction_totals(state, hex_index)[winner_faction]
    overflow = int(totals.sum()) - cap
    if overflow <= 0:
        return []
    K = state.battle_faction.shape[-1]
    origins = [
        int(state.battle_origin[0, hex_index, k]) for k in range(K)
        if int(state.battle_faction[0, hex_index, k]) == winner_faction
    ]
    if not origins:
        return []
    send_back = []
    remaining = [int(x) for x in totals]
    for ut in (0, 1, 2):
        while overflow > 0 and remaining[ut] > 0:
            units = [0, 0, 0]
            units[ut] = 1
            send_back.append({"origin_hex": rng.choice(origins), "units": units})
            remaining[ut] -= 1
            overflow -= 1
    return send_back


def random_buy(state, faction, rng, max_infantry_hexes=2, max_convert_hexes=2):
    """Picks uniformly among a few coarse buy choices - see module
    docstring for why this no longer filters a pre-computed legal-action
    list. Illegal/unaffordable requests (e.g. a hex that can't build an
    outpost, or more gold than the faction has) are silently dropped by
    apply_buy_phase_batch, same as every other agent's proposals."""
    decision = {}
    grid = state.grid
    N = state.num_hexes

    if rng.random() < 0.3:
        army_hexes = torch.nonzero(
            (state.army_faction[0] == faction) & ~state.locked[0], as_tuple=False
        ).flatten().tolist()
        if army_hexes and int(state.gold[0, faction]) >= OUTPOST_COST:
            eligible = eligible_outpost_mask(state, faction)[0]
            candidates = [h for h in army_hexes if bool(eligible[h])]
            if candidates:
                h = rng.choice(candidates)
                unit_type = rng.randrange(3)
                if int(state.army_units[0, h, unit_type]) > 0:
                    decision["outpost_type"] = 1
                    decision["outpost_hex"] = h
                    decision["outpost_unit_type"] = unit_type
        if "outpost_type" not in decision:
            outposts = torch.nonzero(
                (state.city_owner[0] == faction) & ~state.is_capital[0], as_tuple=False
            ).flatten().tolist()
            if outposts:
                h = rng.choice(outposts)
                decision["outpost_type"] = 2
                decision["outpost_hex"] = h
                decision["outpost_upgrade"] = rng.randrange(3)

    city_hexes = torch.nonzero((state.city_owner[0] == faction) & ~state.locked[0], as_tuple=False).flatten().tolist()
    if city_hexes:
        n = rng.randint(0, min(max_infantry_hexes, len(city_hexes)))
        decision["infantry_buy"] = {h: rng.randint(1, 3) for h in rng.sample(city_hexes, n)} if n else {}

    army_hexes = torch.nonzero(
        (state.army_faction[0] == faction) & (state.army_units[0, :, 0] > 0), as_tuple=False
    ).flatten().tolist()
    if army_hexes:
        n = rng.randint(0, min(max_convert_hexes, len(army_hexes)))
        chosen = rng.sample(army_hexes, n) if n else []
        cav, arc = {}, {}
        for h in chosen:
            (cav if rng.random() < 0.5 else arc)[h] = rng.randint(1, 2)
        if cav:
            decision["convert_cavalry"] = cav
        if arc:
            decision["convert_archers"] = arc

    return decision


def random_placement(rng, legal_mask):
    candidates = torch.nonzero(legal_mask, as_tuple=False).flatten()
    return int(rng.choice(candidates.tolist()))


def random_draft(rng, legal_pool):
    return rng.choice(legal_pool)


def random_swap(rng):
    return rng.random() < 0.5


def random_resource_choice(rng):
    return rng.choice(("iron", "fish"))


def make_random_agents(num_factions, seed=0):
    """Returns (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification, decide_resource_choice, decide_placement,
    decide_draft, decide_swap) - each {faction: callable}, matching
    engine.turn.run_turn's and engine.placement.run_city_setup's expected
    signatures (decide_buy(state, faction), decide_movement(state,
    faction, step, legal_mask) - see engine/turn.py's module docstring).
    Each faction gets its own random.Random (mirrors v1's per-agent rng),
    keyed off `seed` so a whole game's agent decisions are reproducible."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return random_buy(state, faction, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return random_movement(rngs[faction], legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return random_movement(rngs[faction], legal_mask)

    def decide_target(state, hex_index, faction):
        return random_target(state, hex_index, faction, rngs[faction])

    def decide_rectification(state, hex_index, winner_faction, cap):
        return random_rectification(state, hex_index, winner_faction, cap, rngs[winner_faction])

    def decide_resource_choice(state, faction, hex_index):
        return random_resource_choice(rngs[faction])

    def decide_placement(state, faction, legal_mask):
        return random_placement(rngs[faction], legal_mask)

    def decide_draft(state, faction, legal_pool):
        return random_draft(rngs[faction], legal_pool)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        return random_swap(rngs[faction])

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
