"""
Agent-driven capital setup for engine: colourless city placement followed
by a draft. Two phases, each its own independently-random faction order
drawn from `rng`:

  1. Placement: each faction, in turn, places one colourless (unowned)
     city on a hex it's legally allowed to (see legal_placement_mask) -
     nobody owns anything yet; state.city_placer just records who placed
     each hex's city, for the draft phase below to reason about.
  2. Draft: each faction, in turn, claims one already-placed city as its
     capital - never the one it placed itself. When only one faction and
     one city are left: if it's the city that faction placed, they just
     get it (no decision to make - rulebook's "if your city is the last
     city you just get it"). Otherwise the leftover city belongs to
     someone (`placer`) who already drafted something else earlier, and
     the last faction gets to choose whether to force a swap - take what
     `placer` currently holds, bumping `placer` onto the leftover.

Callback signatures (mirroring turn.py's module docstring style):

  decide_placement(state, faction, legal_mask) -> hex_index
  decide_draft(state, faction, legal_pool) -> hex_index         # legal_pool: [hex_index, ...]
  decide_swap(state, faction, leftover_hex, placer_faction, placer_hex) -> bool
    True forces the swap; anything falsy keeps the leftover as-is. Only
    ever called for the single last-drafting faction, and only when the
    leftover city isn't the one they placed themselves.

Like movement.py/buy.py, whatever a callback returns is re-validated
against the legal set actually computed and handed to it - an invalid
answer silently falls back to a random legal choice (via `rng`, the same
shared Random threaded through run_city_setup) rather than raising.

NOT BATCHED, DELIBERATELY (see setup.py's module docstring for the same
reasoning): `state` here is always a single (batch_size=1) game - this
runs once per game, before any turn loop, with a handful of sequential,
agent-driven decision points (not once per turn), so there's nothing to
gain from batching it. `state` fields are indexed at batch position 0
throughout; `legal_placement_mask`/legal_pool are plain [N]/list shapes,
matching the pre-rewrite single-game callback signatures exactly (this
module's callbacks are unaffected by the rest of the engine's batching -
see run_city_setup's docstring).
"""

from .geometry import hex_distance
from .state import IMPASSABLE_BY_TERRAIN, NO_FACTION

CAPITAL_MIN_DIST = 3  # minimum hex distance between any two placed cities
EDGE_BAN_MIN_FACTIONS = 5  # 5-7 players: capitals may not be placed on edge tiles
EDGE_BAN_MAX_FACTIONS = 7


def _passable_mask(state):
    """[N] bool."""
    return ~IMPASSABLE_BY_TERRAIN.to(state.device)[state.terrain[0].long()]


def _edge_mask(grid):
    return [grid.is_edge(i) for i in range(grid.num_hexes)]


def _min_dist_mask(grid, placed_hexes, min_dist):
    """[N] bool (plain Python list) - hexes at least `min_dist` from
    every hex already placed on."""
    if not placed_hexes or min_dist <= 0:
        return [True] * grid.num_hexes
    placed_coords = [grid.coord_of(p) for p in placed_hexes]
    return [all(hex_distance(grid.coord_of(i), p) >= min_dist for p in placed_coords) for i in range(grid.num_hexes)]


def legal_placement_mask(state, num_factions):
    """[N] bool of hexes a colourless city could legally be placed on
    right now: passable, not already placed on, at least
    CAPITAL_MIN_DIST from every already-placed city, and - only for 5-7
    factions - not on the board's edge ring. Relaxes in tiers if the
    strict mask would be empty: the edge ban drops first, then
    CAPITAL_MIN_DIST too; "not already placed" + "passable" never relax.
    Whichever tier ends up non-empty is what's actually handed to
    decide_placement, so an agent is never offered a hex that isn't
    really legal."""
    grid = state.grid
    passable = _passable_mask(state)
    not_placed = state.city_placer[0] == NO_FACTION
    base = passable & not_placed
    placed_hexes = [int(h) for h in (state.city_placer[0] != NO_FACTION).nonzero(as_tuple=False).flatten().tolist()]
    ban_edges = EDGE_BAN_MIN_FACTIONS <= num_factions <= EDGE_BAN_MAX_FACTIONS

    dist_ok_list = _min_dist_mask(grid, placed_hexes, CAPITAL_MIN_DIST)
    dist_ok = base.new_tensor(dist_ok_list)
    not_edge = base.new_tensor([not e for e in _edge_mask(grid)]) if ban_edges else base.new_ones(base.shape, dtype=base.dtype)

    strict = base & dist_ok & not_edge
    if bool(strict.any()):
        return strict
    no_edge_ban = base & dist_ok
    if bool(no_edge_ban.any()):
        return no_edge_ban
    return base


def run_city_setup(state, decide_placement, decide_draft, decide_swap, rng, log=None):
    """Mutates and returns `state` - see module docstring for the full
    placement-then-draft process and each callback's signature. Called
    once, before any turn is played.

    `log`: optional list - if given, receives every individual
    placement/draft step in order, for run.py to dump alongside
    city_placement_log.json. Each entry is {"type", "faction", "q", "r",
    "s", ...}:
      "place"      - faction placed a colourless city at (q, r, s).
      "draft"      - faction drafted the already-placed city at (q, r, s).
      "draft_auto" - faction was left with the city they placed themselves.
      "keep"       - faction was offered a swap and declined it.
      "swap"       - faction forced a swap.
    """
    grid = state.grid
    num_factions = state.num_factions

    placement_order = rng.sample(range(num_factions), num_factions)
    for faction in placement_order:
        legal_mask = legal_placement_mask(state, num_factions)
        legal_indices = legal_mask.nonzero(as_tuple=False).flatten().tolist()
        choice = decide_placement[faction](state, faction, legal_mask)
        if choice is None or not (0 <= choice < state.num_hexes) or not bool(legal_mask[choice]):
            choice = int(rng.choice(legal_indices))
        state.city_placer[0, choice] = faction
        if log is not None:
            q, r, s = grid.coord_of(choice)
            log.append({"type": "place", "faction": faction, "q": q, "r": r, "s": s})

    placed_hexes = (state.city_placer[0] != NO_FACTION).nonzero(as_tuple=False).flatten().tolist()
    draft_order = rng.sample(range(num_factions), num_factions)
    assigned = {}
    settle_counter = 0

    def finalize(faction, hex_index):
        nonlocal settle_counter
        state.city_owner[0, hex_index] = faction
        state.is_capital[0, hex_index] = True
        state.capital_settle_order[0, faction] = settle_counter
        settle_counter += 1
        assigned[faction] = hex_index

    for i, faction in enumerate(draft_order):
        pool = [h for h in placed_hexes if int(state.city_owner[0, h]) == NO_FACTION]
        if i < num_factions - 1:
            legal_pool = [h for h in pool if int(state.city_placer[0, h]) != faction]
            choice = decide_draft[faction](state, faction, legal_pool)
            if choice not in legal_pool:
                choice = rng.choice(legal_pool)
            finalize(faction, choice)
            if log is not None:
                q, r, s = grid.coord_of(choice)
                log.append({"type": "draft", "faction": faction, "q": q, "r": r, "s": s})
        else:
            leftover = pool[0]
            placer = int(state.city_placer[0, leftover])
            if placer == faction:
                finalize(faction, leftover)
                if log is not None:
                    q, r, s = grid.coord_of(leftover)
                    log.append({"type": "draft_auto", "faction": faction, "q": q, "r": r, "s": s})
            else:
                placer_hex = assigned[placer]
                swap = bool(decide_swap[faction](state, faction, leftover, placer, placer_hex))
                if swap:
                    if log is not None:
                        tq, tr, ts = grid.coord_of(placer_hex)
                        bq, br, bs = grid.coord_of(leftover)
                        log.append({
                            "type": "swap", "faction": faction, "q": tq, "r": tr, "s": ts,
                            "placer_faction": placer, "placer_q": bq, "placer_r": br, "placer_s": bs,
                        })
                    finalize(placer, leftover)
                    finalize(faction, placer_hex)
                else:
                    finalize(faction, leftover)
                    if log is not None:
                        q, r, s = grid.coord_of(leftover)
                        log.append({"type": "keep", "faction": faction, "q": q, "r": r, "s": s})

    return state
