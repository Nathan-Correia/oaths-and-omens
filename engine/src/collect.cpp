#include "oo/collect.hpp"

namespace oo {

namespace {

// engine_old/collect.py: _remove_first_unit. Removes one unit (infantry ->
// cavalry -> archers priority) from the FIRST hex in board order holding a
// peaceful army for `faction`. Only peaceful board armies - never units currently
// locked in a pending battle.
void remove_first_unit(GameState& state, int faction) {
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.army_faction[h] != faction || state.units_at(h) <= 0) continue;

        for (int ut = 0; ut < NUM_UNIT_TYPES; ++ut) {
            if (state.army_units[h][ut] > 0) {
                state.army_units[h][ut] -= 1;
                break;
            }
        }
        if (state.units_at(h) == 0) {
            state.army_faction[h] = NO_FACTION;
            state.frozen[h] = false;
        }
        return;  // only the first such hex
    }
}

// engine_old/collect.py: _outpost_resource. Which resource this outpost produces
// this turn, or -1 for none (a desert outpost with no adjacent mountain or lake).
// Mountain/lake adjacency on any of the 6 neighbours overrides the outpost's own
// tile; adjacency to both asks the caller which one they want this turn. An
// outpost only ever produces ONE resource, however many qualifying neighbours it
// has.
int outpost_resource(const GameState& state, int hex_index, int faction, ResourceChoiceFn choose,
                     void* ctx) {
    const int16_t* nb = state.grid->neighbours_of(hex_index);
    bool has_mountain = false, has_lake = false;
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        if (nb[d] < 0) continue;
        if (state.terrain[nb[d]] == kMountain) has_mountain = true;
        else if (state.terrain[nb[d]] == kLake) has_lake = true;
    }

    if (has_mountain && has_lake) {
        return (choose(state, faction, hex_index, ctx) == kIron) ? kIron : kFish;
    }
    if (has_mountain) return kIron;
    if (has_lake) return kFish;

    if (state.terrain[hex_index] == kPlains) return kWood;
    if (state.terrain[hex_index] == kMarsh) return kClay;
    return -1;  // desert with no qualifying neighbour
}

}  // namespace

void apply_gold_income(GameState& state) {
    for (int f = 0; f < state.num_factions; ++f) {
        bool has_any_city = false;
        bool has_capital = false;
        int32_t gold = 0;

        for (int h = 0; h < state.num_hexes; ++h) {
            if (state.city_owner[h] != f) continue;
            has_any_city = true;
            if (state.is_capital[h]) {
                has_capital = true;
            } else {
                gold += (state.outpost_upgrade[h] == kBarracks) ? kOutpostGoldWithBarracks
                                                                : kOutpostGold;
            }
        }

        if (!has_any_city) {
            state.gold[f] = 0;
            remove_first_unit(state, f);
            continue;
        }
        if (has_capital) gold += kCapitalGold;
        state.gold[f] += gold;
    }
}

void apply_resource_income(GameState& state, ResourceChoiceFn choose, void* ctx) {
    // Faction-major, then hex order within a faction - matching engine_old's
    // nested loops. Only the ORDER of choose() calls is observable here (an agent
    // may consume RNG in it), but keep it identical regardless.
    for (int f = 0; f < state.num_factions; ++f) {
        for (int h = 0; h < state.num_hexes; ++h) {
            if (state.city_owner[h] != f || state.is_capital[h]) continue;
            const int resource = outpost_resource(state, h, f, choose, ctx);
            if (resource < 0) continue;
            const int32_t amount = (state.outpost_upgrade[h] == kWorkshop) ? 2 : 1;
            state.resources[f][resource] += amount;
        }
    }
}

void apply_victory_points(GameState& state) {
    for (int f = 0; f < state.num_factions; ++f) {
        int outposts = 0;
        int temples = 0;
        for (int h = 0; h < state.num_hexes; ++h) {
            if (state.city_owner[h] != f || state.is_capital[h]) continue;
            ++outposts;
            if (state.outpost_upgrade[h] == kTemple) ++temples;
        }
        const int recurring = (outposts > 1 ? outposts - 1 : 0) * kOutpostVpPerRound;
        state.victory_points[f] += recurring + temples * kTempleVpPerRound;
    }
}

void apply_collect_phase(GameState& state, ResourceChoiceFn choose, void* ctx) {
    apply_gold_income(state);
    apply_resource_income(state, choose, ctx);
    apply_victory_points(state);
}

}  // namespace oo
