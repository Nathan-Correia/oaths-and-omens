"""
Agent-driven capital setup for engine: colourless city placement followed
by a draft, replacing the old farthest-point-heuristic auto-placement that
used to live in setup.py (see that module's docstring). Two phases, each
its own independently-random faction order drawn from `rng`:

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

Callback signatures (mirroring turn.py's module docstring style - see
that module for the movement/buy/battle equivalents):

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
"""

import numpy as np

from .geometry import hex_distance
from .state import IMPASSABLE_TERRAIN_INDICES, NO_FACTION

CAPITAL_MIN_DIST = 3  # minimum hex distance between any two placed cities
EDGE_BAN_MIN_FACTIONS = 5  # 5-7 players: capitals may not be placed on edge tiles
EDGE_BAN_MAX_FACTIONS = 7


def _passable_mask(state):
    return ~np.isin(state.terrain, IMPASSABLE_TERRAIN_INDICES)


def _edge_mask(grid):
    return np.array([grid.is_edge(i) for i in range(grid.num_hexes)], dtype=bool)


def _min_dist_mask(grid, placed_hexes, min_dist):
    """bool[N]: hexes at least `min_dist` from every hex already placed on."""
    if not placed_hexes or min_dist <= 0:
        return np.ones(grid.num_hexes, dtype=bool)
    ok = np.empty(grid.num_hexes, dtype=bool)
    placed_coords = [grid.coord_of(p) for p in placed_hexes]
    for i in range(grid.num_hexes):
        coord = grid.coord_of(i)
        ok[i] = all(hex_distance(coord, p) >= min_dist for p in placed_coords)
    return ok


def legal_placement_mask(state, num_factions):
    """bool[N] of hexes a colourless city could legally be placed on right
    now: passable, not already placed on, at least CAPITAL_MIN_DIST from
    every already-placed city, and - only for 5-7 factions - not on the
    board's edge ring. Relaxes in tiers if the strict mask would be empty
    (mirrors the old engine/setup.py's "hard floor, fall back to every
    hex" precedent): the edge ban drops first, then CAPITAL_MIN_DIST too;
    "not already placed" + "passable" never relax. Whichever tier ends up
    non-empty is what's actually handed to decide_placement, so an agent
    is never offered a hex that isn't really legal."""
    grid = state.grid
    base = _passable_mask(state) & (state.city_placer == NO_FACTION)
    placed_hexes = np.nonzero(state.city_placer != NO_FACTION)[0].tolist()
    ban_edges = EDGE_BAN_MIN_FACTIONS <= num_factions <= EDGE_BAN_MAX_FACTIONS

    dist_ok = _min_dist_mask(grid, placed_hexes, CAPITAL_MIN_DIST)
    not_edge = ~_edge_mask(grid) if ban_edges else True

    strict = base & dist_ok & not_edge
    if strict.any():
        return strict
    no_edge_ban = base & dist_ok
    if no_edge_ban.any():
        return no_edge_ban
    return base


def run_city_setup(state, decide_placement, decide_draft, decide_swap, rng, log=None):
    """Mutates and returns `state` - see module docstring for the full
    placement-then-draft process and each callback's signature. Called
    once, before any turn is played.

    `log`: optional list - if given, receives every individual
    placement/draft step in order, for run.py to dump alongside
    city_placement_log.json (see city_placement_visualizer.py). Each
    entry is {"type", "faction", "q", "r", "s", ...}:
      "place"      - faction placed a colourless city at (q, r, s).
      "draft"      - faction drafted the already-placed city at (q, r, s).
      "draft_auto" - faction was left with the city they placed themselves
                     (only unclaimed city, no real choice - rulebook's
                     "if your city is the last city you just get it").
      "keep"       - faction was offered a swap and declined it, keeping
                     the leftover city at (q, r, s).
      "swap"       - faction forced a swap: they take (q, r, s) (what
                     "placer_faction" had already drafted), and
                     placer_faction is bumped onto their own original
                     placement at (placer_q, placer_r, placer_s) instead.
    """
    grid = state.grid
    num_factions = state.num_factions

    placement_order = rng.sample(range(num_factions), num_factions)
    for faction in placement_order:
        legal_mask = legal_placement_mask(state, num_factions)
        choice = decide_placement[faction](state, faction, legal_mask)
        if choice is None or not (0 <= choice < state.num_hexes) or not legal_mask[choice]:
            choice = int(rng.choice(np.nonzero(legal_mask)[0].tolist()))
        state.city_placer[choice] = faction
        if log is not None:
            q, r, s = grid.coord_of(choice)
            log.append({"type": "place", "faction": faction, "q": q, "r": r, "s": s})

    placed_hexes = np.nonzero(state.city_placer != NO_FACTION)[0].tolist()
    draft_order = rng.sample(range(num_factions), num_factions)
    assigned = {}
    settle_counter = 0

    def finalize(faction, hex_index):
        nonlocal settle_counter
        state.city_owner[hex_index] = faction
        state.is_capital[hex_index] = True
        state.capital_settle_order[faction] = settle_counter
        settle_counter += 1
        assigned[faction] = hex_index

    for i, faction in enumerate(draft_order):
        pool = [h for h in placed_hexes if state.city_owner[h] == NO_FACTION]
        if i < num_factions - 1:
            legal_pool = [h for h in pool if int(state.city_placer[h]) != faction]
            choice = decide_draft[faction](state, faction, legal_pool)
            if choice not in legal_pool:
                choice = rng.choice(legal_pool)
            finalize(faction, choice)
            if log is not None:
                q, r, s = grid.coord_of(choice)
                log.append({"type": "draft", "faction": faction, "q": q, "r": r, "s": s})
        else:
            leftover = pool[0]
            placer = int(state.city_placer[leftover])
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
