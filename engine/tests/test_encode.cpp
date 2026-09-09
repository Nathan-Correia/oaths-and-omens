// The state encoder (M9b, PLAN.md §10.4).
//
// No golden file, and deliberately so: a byte-for-byte reference would freeze
// the layout, and §10.11 says outright that the encoder is not frozen until M9
// ships. What is asserted instead are the properties a training run actually
// depends on, all of which fail silently if wrong:
//
//   - Everything is relative to the acting faction. Get this wrong and the net
//     learns per-seat behaviour and never generalises; nothing crashes.
//   - Padding past num_hexes is marked ABSENT, not decoded as plains. terrain[]
//     is zeroed past the board and zero means plains, so a missing present bit
//     would silently teach the net that every board is radius 8 with a plains
//     border.
//   - Units mid-battle stay visible. A contested hex has army_faction cleared,
//     so a naive encoder shows an empty tile where a fight is happening.
//   - The query hex is marked. It is what lets one trunk serve all nine
//     decision kinds.

#include "oo/batch.hpp"
#include "oo/encode.hpp"
#include "oo/movement.hpp"
#include "oo/setup.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

oo::DecisionRequest req_for(const oo::GameState& s, int faction, oo::DecisionKind kind) {
    oo::DecisionRequest r;
    r.state = &s;
    r.kind = kind;
    r.faction = faction;
    return r;
}

}  // namespace

int main() {
    auto enc = std::make_unique<oo::EncodedPosition>();
    auto enc2 = std::make_unique<oo::EncodedPosition>();

    // A real mid-game board, not a hand-built one: real states have cities,
    // upgrades, mixed stacks and the occasional battle, and a hand-built board
    // would quietly skip whichever field the encoder forgot.
    auto state = std::make_unique<oo::GameState>();
    oo::create_initial_state(*state, 6, 6, 4242);
    {
        oo::AgentSet agents;
        oo::build_agents(agents, oo::AgentKind::kGreedy, 6, 4242);
        oo::Rng rng(4242);
        oo::SetupDecisions sd = oo::make_setup_decisions(agents);
        oo::run_city_setup(*state, sd, rng);
        oo::TurnDecisions td = oo::make_turn_decisions(agents);
        for (int t = 0; t < 8; ++t) oo::run_turn(*state, td, rng);
    }
    const int n = state->num_hexes;
    check(n == 127, "radius 6 board has 127 hexes");

    // Greedy does not upgrade anything in eight turns, so the upgrade check
    // below was silently vacuous the first time this ran. Force one of each
    // kind onto the board rather than hoping an agent produces them.
    {
        int forced = 0;
        for (int h = 0; h < n && forced < oo::NUM_UPGRADE_TYPES; ++h) {
            if (state->city_owner[h] == oo::NO_FACTION || state->is_capital[h]) continue;
            state->outpost_upgrade[h] = static_cast<int8_t>(forced);
            ++forced;
        }
        check(forced == oo::NUM_UPGRADE_TYPES,
              "the probe board has enough outposts to carry one of each upgrade");
    }

    // --- the header describes the board -------------------------------------
    {
        oo::encode_position(*state, req_for(*state, 0, oo::DecisionKind::kMovement), *enc);
        check(enc->header.version == oo::kEncodingVersion, "version stamped");
        check(enc->header.radius == 6, "radius recorded");
        check(enc->header.num_factions == 6, "faction count recorded");
        check(enc->header.num_hexes == n, "hex count recorded");
        check(enc->header.turn_number == state->turn_number, "turn number recorded");
        check(enc->header.vp_to_win == oo::kVpToWin, "vp target recorded");
    }

    // --- PERSPECTIVE: slot 0 is always the acting faction --------------------
    {
        bool ok_me = true, ok_active = true;
        for (int acting = 0; acting < 6; ++acting) {
            oo::encode_position(*state, req_for(*state, acting, oo::DecisionKind::kMovement), *enc);
            ok_me = ok_me && enc->factions[0].is_me == 1 &&
                    enc->factions[0].victory_points == state->victory_points[acting] &&
                    enc->factions[0].gold == state->gold[acting];
            for (int f = 1; f < 6; ++f) ok_me = ok_me && enc->factions[f].is_me == 0;
            for (int f = 0; f < 6; ++f) ok_active = ok_active && enc->factions[f].active == 1;
            for (int f = 6; f < oo::MAX_FACTIONS; ++f) {
                ok_active = ok_active && enc->factions[f].active == 0;
            }
        }
        check(ok_me, "slot 0 is always the acting faction");
        check(ok_active, "unused faction slots are marked inactive");
    }

    // Faction f's data must land at the SAME relative slot however you get
    // there: encoding for acting A puts faction B at (B-A) mod nf, and encoding
    // for B puts A at (A-B) mod nf. Checking one hex's army owner through both
    // encodings catches a rel[] that is merely self-consistent but wrong.
    {
        int probe = -1;
        for (int h = 0; h < n; ++h) {
            if (state->army_faction[h] != oo::NO_FACTION) {
                probe = h;
                break;
            }
        }
        check(probe >= 0, "the board has at least one army to probe");
        if (probe >= 0) {
            const int owner = state->army_faction[probe];
            bool ok = true;
            for (int acting = 0; acting < 6; ++acting) {
                oo::encode_position(*state, req_for(*state, acting, oo::DecisionKind::kMovement),
                                    *enc);
                const int expect = 1 + ((owner - acting + 6) % 6);
                ok = ok && enc->hexes[probe].army == expect;
            }
            check(ok, "an army's owner is encoded relative to the acting faction");
        }
    }

    // --- padding is ABSENT, not plains ---------------------------------------
    {
        oo::encode_position(*state, req_for(*state, 0, oo::DecisionKind::kMovement), *enc);
        bool present_ok = true, padding_ok = true;
        for (int h = 0; h < n; ++h) {
            present_ok = present_ok && (enc->hexes[h].flags & oo::kHexPresent);
        }
        for (int h = n; h < oo::MAX_HEXES; ++h) {
            padding_ok = padding_ok && (enc->hexes[h].flags & oo::kHexPresent) == 0;
        }
        check(present_ok, "every real hex is marked present");
        check(padding_ok, "every padding hex is marked absent");
    }

    // --- terrain, cities and upgrades round-trip ------------------------------
    {
        oo::encode_position(*state, req_for(*state, 2, oo::DecisionKind::kMovement), *enc);
        bool terrain_ok = true, city_ok = true, cap_ok = true, up_ok = true, pass_ok = true;
        int cities = 0, capitals = 0, upgraded = 0;
        for (int h = 0; h < n; ++h) {
            const oo::HexRecord& hr = enc->hexes[h];
            terrain_ok = terrain_ok && hr.terrain == state->terrain[h];
            pass_ok = pass_ok && bool(hr.flags & oo::kHexPassable) == state->passable(h);
            const int owner = state->city_owner[h];
            if (owner == oo::NO_FACTION) {
                city_ok = city_ok && hr.city == 0;
            } else {
                ++cities;
                city_ok = city_ok && hr.city == 1 + ((owner - 2 + 6) % 6);
                const bool is_cap = (hr.city_flags & 1) != 0;
                cap_ok = cap_ok && is_cap == state->is_capital[h];
                if (state->is_capital[h]) ++capitals;
                const int up = (hr.city_flags >> 1) & 0x3;
                const int want = state->outpost_upgrade[h] == oo::NO_UPGRADE
                                     ? 0
                                     : state->outpost_upgrade[h] + 1;
                up_ok = up_ok && up == want;
                if (want) ++upgraded;
            }
        }
        check(terrain_ok, "terrain round-trips");
        check(pass_ok, "passability round-trips");
        check(city_ok, "city ownership round-trips, relative");
        check(cap_ok, "capital flag round-trips");
        check(up_ok, "outpost upgrade round-trips");
        // Prove the loop above was not vacuous.
        check(cities >= 6, "the probe board actually has cities to check");
        check(capitals == 6, "the probe board has one capital per faction");
        check(upgraded == oo::NUM_UPGRADE_TYPES,
              "every upgrade kind was actually exercised, not just the none case");
        std::printf("  probe board: %d cities, %d capitals, %d upgraded\n", cities, capitals,
                    upgraded);
    }

    // --- the query hex is marked, and only that one ---------------------------
    {
        oo::DecisionRequest r = req_for(*state, 1, oo::DecisionKind::kResource);
        r.hex = 40;
        oo::encode_position(*state, r, *enc);
        int marked = 0;
        for (int h = 0; h < n; ++h) {
            if (enc->hexes[h].flags & oo::kHexQuery) ++marked;
        }
        check(marked == 1 && (enc->hexes[40].flags & oo::kHexQuery), "exactly the query hex is marked");

        oo::DecisionRequest sw = req_for(*state, 1, oo::DecisionKind::kSwap);
        sw.leftover_hex = 12;
        sw.placer_hex = 55;
        oo::encode_position(*state, sw, *enc2);
        check((enc2->hexes[12].flags & oo::kHexQuery) &&
                  (enc2->hexes[55].flags & oo::kHexQuery2),
              "a swap marks both of its hexes, distinguishably");

        // A decision with no hex must mark none - otherwise hex 0 would be
        // permanently flagged, which is the kind of thing that trains in.
        oo::encode_position(*state, req_for(*state, 1, oo::DecisionKind::kBuy), *enc2);
        int none = 0;
        for (int h = 0; h < n; ++h) {
            if (enc2->hexes[h].flags & (oo::kHexQuery | oo::kHexQuery2)) ++none;
        }
        check(none == 0, "a decision with no hex marks no hex");
    }

    // --- the decision kind reaches the header --------------------------------
    {
        bool ok = true;
        for (int k = 0; k < oo::kNumDecisionKinds; ++k) {
            oo::DecisionRequest r = req_for(*state, 0, static_cast<oo::DecisionKind>(k));
            r.step = 2;
            oo::encode_position(*state, r, *enc);
            ok = ok && enc->header.decision_kind == k && enc->header.step == 2;
        }
        check(ok, "every decision kind and the step reach the header");
    }

    // --- units mid-battle stay visible ---------------------------------------
    // A hex with a battle has army_faction cleared, so an encoder that read only
    // army_units would show an empty tile in the middle of a fight.
    {
        auto b = std::make_unique<oo::GameState>();
        oo::new_empty(*b, oo::HexGrid::shared(4), 2);
        for (int h = 0; h < b->num_hexes; ++h) b->terrain[h] = oo::kPlains;
        const int hex = 10;
        oo::Battle& battle = b->new_battle(hex);
        battle.round = 2;
        battle.nslots = 2;
        battle.slots[0] = oo::BattleSlot{5, {2, 1, 0}, 0, true};
        battle.slots[1] = oo::BattleSlot{7, {1, 0, 2}, 1, false};

        oo::DecisionRequest r = req_for(*b, 1, oo::DecisionKind::kTarget);
        r.hex = hex;
        oo::encode_position(*b, r, *enc);
        const oo::HexRecord& hr = enc->hexes[hex];
        check((hr.flags & oo::kHexBattle) != 0, "a contested hex is flagged");
        check(hr.battle_round == 2, "the battle round is encoded");
        check(hr.units[0] == 3 && hr.units[1] == 1 && hr.units[2] == 2,
              "units mid-battle are still counted on the hex");
        // Acting faction is 1, so faction 1 -> slot 0 and faction 0 -> slot 1.
        check(hr.battle_factions == 0x3, "both participants appear in the mask, relative");

        // ...and they count toward their owners' in-play totals.
        check(enc->factions[0].units[2] == 2, "acting faction's archers in battle are in play");
        check(enc->factions[1].units[0] == 2, "opponent's infantry in battle are in play");
    }

    // --- vp_behind_leader ------------------------------------------------------
    {
        auto b = std::make_unique<oo::GameState>();
        oo::new_empty(*b, oo::HexGrid::shared(4), 3);
        b->victory_points[0] = 10;
        b->victory_points[1] = 25;
        b->victory_points[2] = 4;
        oo::encode_position(*b, req_for(*b, 0, oo::DecisionKind::kBuy), *enc);
        check(enc->header.max_vp == 25, "max VP on the board reaches the header");
        check(enc->factions[0].vp_behind_leader == 15, "acting faction's deficit is encoded");
        check(enc->factions[1].vp_behind_leader == 0, "the leader is 0 behind");
        check(enc->factions[2].vp_behind_leader == 21, "a trailing faction's deficit is encoded");
    }

    // --- the wire is much smaller than expanded planes -------------------------
    {
        const int r8 = 3 * 8 * 9 + 1;
        std::printf("  encoded bytes: r5 %d, r6 %d, r7 %d, r8 %d (vs ~11800 for fp16 planes)\n",
                    oo::encoded_bytes(91), oo::encoded_bytes(127), oo::encoded_bytes(169),
                    oo::encoded_bytes(r8));
        check(oo::encoded_bytes(r8) < 3000, "an r8 position fits in under 3 KB on the wire");
    }

    std::printf("test_encode: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
