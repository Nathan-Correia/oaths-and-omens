#include "oo/encode.hpp"

#include "oo/buy.hpp"

#include <cstring>

namespace oo {

namespace {

uint8_t sat8(int v) {
    if (v < 0) return 0;
    return v > 255 ? uint8_t(255) : static_cast<uint8_t>(v);
}

int16_t sat16(int32_t v) {
    if (v < -32768) return -32768;
    return v > 32767 ? int16_t(32767) : static_cast<int16_t>(v);
}

}  // namespace

void encode_position(const GameState& state, const DecisionRequest& request, EncodedPosition& out) {
    std::memset(&out, 0, sizeof(out));

    const int nf = state.num_factions;
    const int n = state.num_hexes;
    const int acting = request.faction >= 0 ? request.faction : 0;

    // rel[f] is where faction f sits in the encoding. Slot 0 is always the
    // acting faction, so the network never has to learn "which one am I".
    int rel[MAX_FACTIONS];
    for (int f = 0; f < MAX_FACTIONS; ++f) rel[f] = f;
    if (nf > 0) {
        for (int f = 0; f < nf; ++f) rel[f] = (f - acting + nf) % nf;
    }

    // --- header --------------------------------------------------------------
    int max_vp = 0;
    for (int f = 0; f < nf; ++f) {
        if (state.victory_points[f] > max_vp) max_vp = state.victory_points[f];
    }
    out.header.version = static_cast<uint8_t>(kEncodingVersion);
    out.header.radius = static_cast<uint8_t>(state.grid->radius());
    out.header.num_factions = static_cast<uint8_t>(nf);
    out.header.decision_kind = static_cast<uint8_t>(request.kind);
    out.header.step = static_cast<uint8_t>(request.step);
    out.header.num_hexes = static_cast<int16_t>(n);
    out.header.turn_number = sat16(state.turn_number);
    out.header.turns_remaining = -1;  // filled by the caller if a cap applies
    out.header.max_vp = sat16(max_vp);
    out.header.vp_to_win = static_cast<int16_t>(kVpToWin);

    // --- factions ------------------------------------------------------------
    int32_t units_in_play[MAX_FACTIONS][NUM_UNIT_TYPES] = {};
    for (int h = 0; h < n; ++h) {
        const int f = state.army_faction[h];
        if (f != NO_FACTION) {
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) units_in_play[f][t] += state.army_units[h][t];
        }
    }
    // Units sitting in a battle are still in play, and a policy that could not
    // see them would badly misjudge any turn with a fight running.
    for (int b = 0; b < state.num_battles; ++b) {
        const Battle& battle = state.battles[b];
        for (int s = 0; s < battle.nslots; ++s) {
            const BattleSlot& slot = battle.slots[s];
            if (slot.faction == NO_FACTION) continue;
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                units_in_play[slot.faction][t] += slot.units[t];
            }
        }
    }

    for (int f = 0; f < nf; ++f) {
        FactionRecord& fr = out.factions[rel[f]];
        fr.active = 1;
        fr.is_me = (f == acting) ? 1 : 0;
        fr.outposts = sat8(outpost_count(state, f));
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) fr.units[t] = sat8(units_in_play[f][t]);
        fr.settle_order = static_cast<int8_t>(state.capital_settle_order[f]);
        fr.gold = sat16(state.gold[f]);
        for (int r = 0; r < NUM_RESOURCES; ++r) fr.resources[r] = sat16(state.resources[f][r]);
        fr.victory_points = sat16(state.victory_points[f]);
        fr.kill_xp = sat16(state.kill_xp[f]);
        // Signed: positive means behind, negative means this faction IS the
        // leader and by how much. The VP race is the whole game, and making the
        // net derive this from ten absolute scores would be wasteful.
        fr.vp_behind_leader = sat16(max_vp - state.victory_points[f]);
    }

    // --- hexes ---------------------------------------------------------------
    for (int h = 0; h < n; ++h) {
        HexRecord& hr = out.hexes[h];
        hr.terrain = static_cast<uint8_t>(state.terrain[h]);

        uint8_t flags = kHexPresent;
        if (state.passable(h)) flags |= kHexPassable;
        if (state.frozen[h]) flags |= kHexFrozen;
        if (state.grid->is_edge(h)) flags |= kHexEdge;

        const int owner = state.city_owner[h];
        if (owner != NO_FACTION) {
            hr.city = static_cast<uint8_t>(1 + rel[owner]);
            uint8_t cf = state.is_capital[h] ? 1u : 0u;
            const int up = state.outpost_upgrade[h];
            cf |= static_cast<uint8_t>((up == NO_UPGRADE ? 0 : up + 1) << 1);
            hr.city_flags = cf;
        }

        const int af = state.army_faction[h];
        if (af != NO_FACTION) {
            hr.army = static_cast<uint8_t>(1 + rel[af]);
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) hr.units[t] = sat8(state.army_units[h][t]);
        }

        const Battle* b = state.battle_at(h);
        if (b != nullptr) {
            flags |= kHexBattle;
            hr.battle_round = sat8(b->round);
            uint16_t mask = 0;
            for (int s = 0; s < b->nslots; ++s) {
                const BattleSlot& slot = b->slots[s];
                if (slot.faction == NO_FACTION) continue;
                mask |= static_cast<uint16_t>(1u << rel[slot.faction]);
                // Units mid-battle are reported on the hex too, so a policy
                // reading army_units alone does not see an empty contested tile.
                for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                    hr.units[t] = sat8(hr.units[t] + slot.units[t]);
                }
            }
            hr.battle_factions = mask;
        }

        hr.flags = flags;
    }

    // --- which hex is the question about -------------------------------------
    if (request.hex >= 0 && request.hex < n) out.hexes[request.hex].flags |= kHexQuery;
    if (request.kind == DecisionKind::kSwap) {
        if (request.leftover_hex >= 0 && request.leftover_hex < n) {
            out.hexes[request.leftover_hex].flags |= kHexQuery;
        }
        if (request.placer_hex >= 0 && request.placer_hex < n) {
            out.hexes[request.placer_hex].flags |= kHexQuery2;
        }
    }
}

void encode_batch(const DecisionRequest* requests, int n, EncodedPosition* out) {
    for (int i = 0; i < n; ++i) {
        encode_position(*requests[i].state, requests[i], out[i]);
    }
}

}  // namespace oo
