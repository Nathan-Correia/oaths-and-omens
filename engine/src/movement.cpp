#include "oo/movement.hpp"

#include <cassert>

namespace oo {

namespace {

struct Units {
    int16_t u[NUM_UNIT_TYPES];
    int total() const { return u[0] + u[1] + u[2]; }
};

struct CollectedMove {
    int faction;
    int from;
    int to;
    Units units;
};

// A battle contribution being added: who, from where, how much, and whether they
// MOVED into the hex to join the fight.
struct Contribution {
    int faction;
    int origin;
    Units units;
    bool moved;
};

void legal_mask_impl(const GameState& state, int faction, bool require_cavalry, LegalMask& out) {
    const HexGrid& grid = *state.grid;
    const int n = state.num_hexes;
    for (int h = 0; h < n; ++h) {
        bool own = state.army_faction[h] == faction && !state.locked[h] && !state.frozen[h];
        if (own && require_cavalry) own = state.army_units[h][kCavalry] > 0;
        for (int d = 0; d < NUM_DIRECTIONS; ++d) {
            const int j = grid.neighbour(h, d);
            out.cell[h][d] = own && j >= 0 && !kImpassableByTerrain[state.terrain[j]];
        }
    }
    // Hexes past num_hexes are never legal but must be deterministic - a caller
    // scanning the whole fixed array must not read uninitialised memory.
    for (int h = n; h < MAX_HEXES; ++h) {
        for (int d = 0; d < NUM_DIRECTIONS; ++d) out.cell[h][d] = false;
    }
}

Units units_at(const GameState& state, int hex_index) {
    Units u;
    for (int t = 0; t < NUM_UNIT_TYPES; ++t) u.u[t] = state.army_units[hex_index][t];
    return u;
}

// Removes `units` from a hex's army - the whole thing for a movement-phase
// action, or just the cavalry for a cavalry-phase one. A cavalry-only departure
// can leave infantry/archers behind, still owned by the same faction.
void subtract_departure(GameState& state, int hex_index, const Units& units) {
    for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.army_units[hex_index][t] -= units.u[t];
    if (state.units_at(hex_index) == 0) {
        state.army_faction[hex_index] = NO_FACTION;
        state.frozen[hex_index] = false;
    }
}

int first_empty_battle_slot(const GameState& state, int hex_index) {
    for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
        if (state.battle_faction[hex_index][k] == NO_FACTION) return k;
    }
    assert(false && "battle contributions exceeded MAX_BATTLE_CONTRIB - raise the cap");
    return -1;
}

// Appends contributions to whatever is already at `hex_index`, locking the hex if
// it was not already. The round number is reset only when the hex NEWLY locks.
// Records the hex into battle_order at that same moment - creation order, which
// matters because the dismount cap tally is shared across a turn's battles.
void start_or_extend_battle(GameState& state, int hex_index, const Contribution* contribs,
                            int n_contribs) {
    const bool was_locked = state.locked[hex_index];
    for (int i = 0; i < n_contribs; ++i) {
        const int slot = first_empty_battle_slot(state, hex_index);
        state.battle_faction[hex_index][slot] = static_cast<int8_t>(contribs[i].faction);
        state.battle_origin[hex_index][slot] = contribs[i].origin;
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
            state.battle_units[hex_index][slot][t] = contribs[i].units.u[t];
        }
        state.battle_moved[hex_index][slot] = contribs[i].moved;
        state.battle_nslots[hex_index] = static_cast<uint8_t>(slot + 1);
    }
    if (!was_locked) {
        state.battle_round[hex_index] = 0;
        assert(state.num_battles < MAX_ACTIVE_BATTLES);
        state.battle_order[state.num_battles++] = static_cast<int16_t>(hex_index);
    }
    state.locked[hex_index] = true;
    state.clear_army(hex_index);
}

// The movement phase moves the WHOLE army. The cavalry phase moves a different,
// fixed subset - only the cavalry count, always leaving infantry and archers
// behind - not a further narrowing of the same idea.
Units units_to_move(const GameState& state, int from_index, bool cavalry_only) {
    if (cavalry_only) {
        Units u{{0, 0, 0}};
        u.u[kCavalry] = state.army_units[from_index][kCavalry];
        return u;
    }
    return units_at(state, from_index);
}

}  // namespace

void legal_movement_mask(const GameState& state, int faction, LegalMask& out) {
    legal_mask_impl(state, faction, false, out);
}

void legal_cavalry_mask(const GameState& state, int faction, LegalMask& out) {
    legal_mask_impl(state, faction, true, out);
}

void apply_movement_step(GameState& state, const MoveActions& actions, Rng& rng, bool cavalry_only) {
    const HexGrid& grid = *state.grid;

    // --- validate and collect ------------------------------------------------
    // Faction order, ascending - engine_old iterates a dict built by turn.py in
    // range order, so this matches. Anything invalid is silently dropped.
    SmallVec<CollectedMove, MAX_FACTIONS> moves;
    for (int faction = 0; faction < state.num_factions; ++faction) {
        if (!actions.has[faction]) continue;
        const int from = actions.move[faction].hex;
        const int dir = actions.move[faction].dir;

        if (from < 0 || from >= state.num_hexes) continue;
        if (state.army_faction[from] != faction) continue;
        if (state.locked[from] || state.frozen[from]) continue;
        const Units units = units_to_move(state, from, cavalry_only);
        if (units.total() <= 0) continue;
        if (dir < 0 || dir >= NUM_DIRECTIONS) continue;
        const int to = grid.neighbour(from, dir);
        if (to < 0) continue;
        if (kImpassableByTerrain[state.terrain[to]]) continue;

        moves.push_back(CollectedMove{faction, from, to, units});
    }

    // --- pass 1: swap / line-battle detection --------------------------------
    bool handled[MAX_FACTIONS] = {};
    SmallVec<int, MAX_FACTIONS> remaining;

    for (int i = 0; i < moves.size(); ++i) {
        if (handled[i]) continue;
        int reverse = -1;
        for (int j = 0; j < moves.size(); ++j) {
            if (j == i || handled[j]) continue;
            if (moves[j].from == moves[i].to && moves[j].to == moves[i].from) {
                reverse = j;
                break;
            }
        }
        if (reverse >= 0 && moves[reverse].faction != moves[i].faction) {
            subtract_departure(state, moves[i].from, moves[i].units);
            subtract_departure(state, moves[reverse].from, moves[reverse].units);

            // A Line Battle has to be pinned to one of the two engaged tiles -
            // there is no third, neutral hex. The SMALLER army's own starting hex
            // hosts it (flavour: they got pushed back onto their own ground), with
            // a coin flip on an exact tie. This rng.random() draw's position in
            // the sequence is load-bearing for parity.
            const int m_total = moves[i].units.total();
            const int r_total = moves[reverse].units.total();
            int battle_hex;
            if (m_total < r_total) {
                battle_hex = moves[i].from;
            } else if (r_total < m_total) {
                battle_hex = moves[reverse].from;
            } else {
                battle_hex = (rng.random() < 0.5) ? moves[i].from : moves[reverse].from;
            }

            const Contribution contribs[2] = {
                {moves[i].faction, moves[i].from, moves[i].units, true},
                {moves[reverse].faction, moves[reverse].from, moves[reverse].units, true},
            };
            start_or_extend_battle(state, battle_hex, contribs, 2);
            handled[i] = true;
            handled[reverse] = true;
        } else {
            remaining.push_back(i);
        }
    }

    // --- pass 2: group remaining moves by destination -------------------------
    // Destination order is first-arrival order, matching Python's insertion-ordered
    // dict. Departures are subtracted as each move is grouped.
    SmallVec<int, MAX_FACTIONS> dests;
    SmallVec<int, MAX_FACTIONS> arrivals[MAX_FACTIONS];
    for (int idx = 0; idx < remaining.size(); ++idx) {
        const int mi = remaining[idx];
        int slot = -1;
        for (int d = 0; d < dests.size(); ++d) {
            if (dests[d] == moves[mi].to) {
                slot = d;
                break;
            }
        }
        if (slot < 0) {
            slot = dests.size();
            dests.push_back(moves[mi].to);
            arrivals[slot].clear();
        }
        arrivals[slot].push_back(mi);
        subtract_departure(state, moves[mi].from, moves[mi].units);
    }

    // Sends a reverted move's units back to its origin. PORTED AS-IS, including
    // the two edge cases: if the origin has since been claimed by a different
    // faction's peaceful merge this starts a battle there rather than merging or
    // vanishing; and if the origin was locked by an unrelated battle this same
    // step, it recreates a peaceful army on a locked hex.
    //
    // The moved flags in the battle-starting branch: the faction that peacefully
    // claimed `origin` this step really did move there (true), while `a`'s own
    // move was voided by the revert, so by the end of the step it never left
    // (false) - same as any other stationary occupant a battle triggers against.
    auto revert_departure = [&](const CollectedMove& a) {
        const int origin = a.from;
        if (state.army_faction[origin] == NO_FACTION) {
            state.army_faction[origin] = static_cast<int8_t>(a.faction);
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.army_units[origin][t] = a.units.u[t];
        } else if (state.army_faction[origin] == a.faction) {
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.army_units[origin][t] += a.units.u[t];
        } else {
            const Contribution contribs[2] = {
                {state.army_faction[origin], origin, units_at(state, origin), true},
                {a.faction, origin, a.units, false},
            };
            start_or_extend_battle(state, origin, contribs, 2);
        }
    };

    for (int d = 0; d < dests.size(); ++d) {
        const int dest = dests[d];
        const SmallVec<int, MAX_FACTIONS>& group = arrivals[d];

        if (state.locked[dest]) {
            Contribution contribs[MAX_FACTIONS];
            for (int i = 0; i < group.size(); ++i) {
                const CollectedMove& a = moves[group[i]];
                contribs[i] = Contribution{a.faction, a.from, a.units, true};
            }
            start_or_extend_battle(state, dest, contribs, group.size());
            continue;
        }

        bool arrival_faction_present[MAX_FACTIONS] = {};
        int distinct_arrival_factions = 0;
        for (int i = 0; i < group.size(); ++i) {
            const int f = moves[group[i]].faction;
            if (!arrival_faction_present[f]) {
                arrival_faction_present[f] = true;
                ++distinct_arrival_factions;
            }
        }

        const int existing_faction =
            state.army_faction[dest] != NO_FACTION ? state.army_faction[dest] : -1;
        const bool hostile_present =
            existing_faction >= 0 && !arrival_faction_present[existing_faction];
        const bool multiple_arrival_factions = distinct_arrival_factions > 1;
        const int dest_owner = state.city_owner[dest] != NO_FACTION ? state.city_owner[dest] : -1;
        const bool foreign_structure = dest_owner >= 0 && !arrival_faction_present[dest_owner];

        if (hostile_present || multiple_arrival_factions || foreign_structure) {
            Contribution contribs[MAX_FACTIONS + 1];
            int n = 0;
            for (int i = 0; i < group.size(); ++i) {
                const CollectedMove& a = moves[group[i]];
                contribs[n++] = Contribution{a.faction, a.from, a.units, true};
            }
            if (existing_faction >= 0) {
                contribs[n++] = Contribution{existing_faction, dest, units_at(state, dest), false};
            }
            start_or_extend_battle(state, dest, contribs, n);
            continue;
        }

        const int existing_total = existing_faction >= 0 ? state.units_at(dest) : 0;
        int arriving_total = 0;
        for (int i = 0; i < group.size(); ++i) arriving_total += moves[group[i]].units.total();

        if (existing_total + arriving_total > MAX_STACK_SIZE) {
            // Outside battle the 6-unit limit is strict: a peaceful merge that
            // would exceed it simply does not happen.
            for (int i = 0; i < group.size(); ++i) revert_departure(moves[group[i]]);
            continue;
        }

        // Only reachable when there is exactly one arriving faction, so "the
        // first arrival's faction" is unambiguous.
        const int resolved_faction =
            existing_faction >= 0 ? existing_faction : moves[group[0]].faction;
        state.army_faction[dest] = static_cast<int8_t>(resolved_faction);
        for (int i = 0; i < group.size(); ++i) {
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                state.army_units[dest][t] += moves[group[i]].units.u[t];
            }
        }
        if (state.terrain[dest] == kMarsh) state.frozen[dest] = true;
    }
}

}  // namespace oo
