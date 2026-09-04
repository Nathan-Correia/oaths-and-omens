#include "oo/buy.hpp"

#include <cstring>

namespace oo {

namespace {

bool can_afford_resources(const GameState& state, int faction, int upgrade) {
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        if (state.resources[faction][r] < kUpgradeCosts[upgrade][r]) return false;
    }
    return true;
}

void spend_resources(GameState& state, int faction, int upgrade) {
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        state.resources[faction][r] -= kUpgradeCosts[upgrade][r];
    }
}

int remaining_cap(const int32_t counts[NUM_UNIT_TYPES], int unit_index) {
    return kSpawnCaps[unit_index] - counts[unit_index];
}

// engine_old/buy.py: _adjacent_enemy_present. "Orthogonally adjacent" on a hex
// board means the 6 neighbours.
bool adjacent_enemy_present(const GameState& state, int hex_index, int faction) {
    const int16_t* nb = state.grid->neighbours_of(hex_index);
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        const int j = nb[d];
        if (j < 0) continue;
        const int8_t f = state.army_faction[j];
        if (f != NO_FACTION && f != faction) return true;
    }
    return false;
}

}  // namespace

void eligible_outpost_mask(const GameState& state, int faction, bool out[MAX_HEXES]) {
    const HexGrid& grid = *state.grid;
    const int n = state.num_hexes;

    for (int h = 0; h < n; ++h) out[h] = (state.city_owner[h] == NO_FACTION);

    // Gather the reference hexes once, then sweep the board per constraint -
    // O(hexes x cities), not O(hexes^2). Mirrors engine_old's three separate
    // masks, each skipped entirely when it has no reference hexes (an empty
    // `nonzero` result there leaves the mask untouched rather than zeroing it).
    SmallVec<int16_t, MAX_FACTIONS> own_capitals;
    SmallVec<int16_t, MAX_FACTIONS> enemy_capitals;
    SmallVec<int16_t, MAX_FACTIONS * kOutpostCap> all_outposts;
    for (int c = 0; c < n; ++c) {
        if (state.city_owner[c] == NO_FACTION) continue;
        if (state.is_capital[c]) {
            if (state.city_owner[c] == faction) {
                own_capitals.push_back(static_cast<int16_t>(c));
            } else {
                enemy_capitals.push_back(static_cast<int16_t>(c));
            }
        } else {
            all_outposts.push_back(static_cast<int16_t>(c));
        }
    }

    for (int16_t c : own_capitals) {
        for (int h = 0; h < n; ++h) {
            if (out[h] && grid.distance(h, c) < kOutpostMinDistOwnCapital) out[h] = false;
        }
    }
    for (int16_t c : enemy_capitals) {
        for (int h = 0; h < n; ++h) {
            if (out[h] && grid.distance(h, c) < kOutpostMinDistEnemyCapital) out[h] = false;
        }
    }
    for (int16_t c : all_outposts) {
        for (int h = 0; h < n; ++h) {
            if (out[h] && grid.distance(h, c) < kOutpostMinDistOtherOutpost) out[h] = false;
        }
    }
}

bool can_build_outpost(const GameState& state, int hex_index, int faction) {
    // Single-hex form of the same rules; avoids building the whole mask.
    if (state.city_owner[hex_index] != NO_FACTION) return false;
    const HexGrid& grid = *state.grid;
    for (int c = 0; c < state.num_hexes; ++c) {
        if (state.city_owner[c] == NO_FACTION) continue;
        int min_dist;
        if (state.is_capital[c]) {
            min_dist = (state.city_owner[c] == faction) ? kOutpostMinDistOwnCapital
                                                        : kOutpostMinDistEnemyCapital;
        } else {
            min_dist = kOutpostMinDistOtherOutpost;
        }
        if (grid.distance(hex_index, c) < min_dist) return false;
    }
    return true;
}

void get_legal_buy_actions(const GameState& state, int faction, LegalBuyActions& out) {
    out.clear();
    const int n = state.num_hexes;
    int32_t counts[NUM_UNIT_TYPES];
    count_all_units_in_play(state, faction, counts);

    // 1. Recruit infantry at an owned, unlocked city.
    if (remaining_cap(counts, kInfantry) > 0 && state.gold[faction] >= kInfantryCost) {
        for (int h = 0; h < n; ++h) {
            if (state.city_owner[h] != faction || state.locked(h)) continue;
            if (state.is_capital[h] || !adjacent_enemy_present(state, h, faction)) {
                out.push_back(BuyAction{BuyType::kBuyInfantry, static_cast<int16_t>(h), 0, 0});
            }
        }
    }

    // 2. Convert one of your infantry into cavalry or archers.
    if (state.kill_xp[faction] > 0 && state.gold[faction] >= 1) {
        for (int h = 0; h < n; ++h) {
            if (state.army_faction[h] != faction || state.army_units[h][kInfantry] <= 0) continue;
            for (int unit : {kCavalry, kArchers}) {
                if (remaining_cap(counts, unit) > 0) {
                    out.push_back(BuyAction{BuyType::kConvertToSpecial, static_cast<int16_t>(h),
                                            static_cast<int8_t>(unit), 0});
                }
            }
        }
    }

    // 3. Found an outpost, consuming a unit standing there.
    if (state.gold[faction] >= kOutpostCost && outpost_count(state, faction) < kOutpostCap) {
        bool eligible[MAX_HEXES];
        eligible_outpost_mask(state, faction, eligible);
        for (int h = 0; h < n; ++h) {
            if (state.army_faction[h] != faction || state.locked(h)) continue;
            if (!eligible[h]) continue;
            for (int unit = 0; unit < NUM_UNIT_TYPES; ++unit) {
                if (state.army_units[h][unit] > 0) {
                    out.push_back(BuyAction{BuyType::kBuildOutpost, static_cast<int16_t>(h),
                                            static_cast<int8_t>(unit), 0});
                }
            }
        }
    }

    // 4. Upgrade (or convert) an owned outpost. Upgrade order is barracks,
    // workshop, temple - engine_old iterates UPGRADE_COSTS, a dict in that order.
    for (int h = 0; h < n; ++h) {
        if (state.city_owner[h] != faction || state.is_capital[h] || state.locked(h)) continue;
        const int current = state.outpost_upgrade[h];
        for (int up = 0; up < NUM_UPGRADE_TYPES; ++up) {
            if (up == current) continue;
            if (can_afford_resources(state, faction, up)) {
                out.push_back(BuyAction{BuyType::kUpgradeOutpost, static_cast<int16_t>(h), 0,
                                        static_cast<int8_t>(up)});
            }
        }
    }
}

namespace {

// engine_old/buy.py: _apply_one. Returns whether the purchase actually happened.
bool apply_one(GameState& state, int faction, const BuyAction& action,
               int32_t counts[NUM_UNIT_TYPES], int8_t enemy_adjacent_cache[MAX_HEXES]) {
    switch (action.type) {
        case BuyType::kBuyInfantry: {
            const int h = action.hex;
            if (state.city_owner[h] != faction || state.locked(h)) return false;

            if (!state.is_capital[h]) {
                if (enemy_adjacent_cache[h] < 0) {
                    enemy_adjacent_cache[h] = adjacent_enemy_present(state, h, faction) ? 1 : 0;
                }
                if (enemy_adjacent_cache[h] == 1) return false;
            }

            if (state.gold[faction] < kInfantryCost || remaining_cap(counts, kInfantry) <= 0) {
                return false;
            }

            // Stack-cap check runs BEFORE the gold is deducted. engine_old used to
            // have it after, so a purchase that failed here still cost 2 gold -
            // measured at 38 of a faction's starting 50 gold wasted on turn 1
            // alone. An empty hex is fine (we would be the first army there);
            // anything else not ours cannot actually occur given the city_owner
            // check above, but is kept for parity.
            if ((state.army_faction[h] != NO_FACTION && state.army_faction[h] != faction) ||
                state.units_at(h) >= MAX_STACK_SIZE) {
                return false;
            }

            state.gold[faction] -= kInfantryCost;
            if (state.army_faction[h] == NO_FACTION) state.army_faction[h] = static_cast<int8_t>(faction);
            state.army_units[h][kInfantry] += 1;
            counts[kInfantry] += 1;
            return true;
        }

        case BuyType::kConvertToSpecial: {
            const int h = action.hex;
            const int unit = action.unit_type;
            if (state.army_faction[h] != faction || state.army_units[h][kInfantry] <= 0) return false;
            if (state.kill_xp[faction] <= 0 || state.gold[faction] < 1 ||
                remaining_cap(counts, unit) <= 0) {
                return false;
            }
            state.kill_xp[faction] -= 1;
            state.gold[faction] -= 1;
            state.army_units[h][kInfantry] -= 1;
            state.army_units[h][unit] += 1;
            counts[kInfantry] -= 1;
            counts[unit] += 1;
            return true;
        }

        case BuyType::kBuildOutpost: {
            const int h = action.hex;
            const int unit = action.unit_type;
            if (state.army_faction[h] != faction || state.locked(h)) return false;
            if (state.army_units[h][unit] <= 0) return false;
            if (state.gold[faction] < kOutpostCost || outpost_count(state, faction) >= kOutpostCap) {
                return false;
            }
            if (!can_build_outpost(state, h, faction)) return false;

            state.gold[faction] -= kOutpostCost;
            state.army_units[h][unit] -= 1;
            counts[unit] -= 1;
            if (state.units_at(h) == 0) state.army_faction[h] = NO_FACTION;
            state.city_owner[h] = static_cast<int8_t>(faction);
            return true;
        }

        case BuyType::kUpgradeOutpost: {
            const int h = action.hex;
            const int up = action.upgrade;
            if (state.city_owner[h] != faction || state.is_capital[h] || state.locked(h)) return false;
            if (state.outpost_upgrade[h] == up) return false;
            if (!can_afford_resources(state, faction, up)) return false;

            spend_resources(state, faction, up);
            state.outpost_upgrade[h] = static_cast<int8_t>(up);
            return true;
        }
    }
    return false;
}

}  // namespace

void apply_buy_phase(GameState& state, const ChosenBuyActions chosen[MAX_FACTIONS]) {
    for (int faction = 0; faction < state.num_factions; ++faction) {
        int32_t counts[NUM_UNIT_TYPES];
        count_all_units_in_play(state, faction, counts);

        // -1 = not yet computed. Deliberately a cache and not a recomputation:
        // engine_old computes adjacency once per hex per faction per buy phase and
        // keeps using it even after purchases change the board. Recomputing would
        // be "more correct" and would diverge.
        int8_t enemy_adjacent_cache[MAX_HEXES];
        std::memset(enemy_adjacent_cache, -1, sizeof(enemy_adjacent_cache));

        bool outpost_recruited[MAX_HEXES] = {};
        bool outpost_action_used = false;

        for (int i = 0; i < chosen[faction].size(); ++i) {
            const BuyAction& action = chosen[faction][i];

            // Per-turn batch caps, checked before the action is attempted.
            if (action.type == BuyType::kBuyInfantry) {
                const int h = action.hex;
                if (state.city_owner[h] == faction && !state.is_capital[h] &&
                    state.outpost_upgrade[h] != kBarracks && outpost_recruited[h]) {
                    continue;
                }
            }
            if ((action.type == BuyType::kBuildOutpost || action.type == BuyType::kUpgradeOutpost) &&
                outpost_action_used) {
                continue;
            }

            if (!apply_one(state, faction, action, counts, enemy_adjacent_cache)) continue;

            if (action.type == BuyType::kBuyInfantry) {
                const int h = action.hex;
                if (state.city_owner[h] == faction && !state.is_capital[h]) {
                    outpost_recruited[h] = true;
                }
            } else if (action.type == BuyType::kBuildOutpost ||
                       action.type == BuyType::kUpgradeOutpost) {
                outpost_action_used = true;
            }
        }
    }
}

}  // namespace oo
