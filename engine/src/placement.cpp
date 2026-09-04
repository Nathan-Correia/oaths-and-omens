#include "oo/placement.hpp"

#include <vector>

namespace oo {

namespace {

bool any_set(const bool* mask, int n) {
    for (int i = 0; i < n; ++i) {
        if (mask[i]) return true;
    }
    return false;
}

}  // namespace

void legal_placement_mask(const GameState& state, bool out[MAX_HEXES]) {
    const HexGrid& grid = *state.grid;
    const int n = state.num_hexes;
    const bool ban_edges = state.num_factions >= kEdgeBanMinFactions &&
                           state.num_factions <= kEdgeBanMaxFactions;

    // base: passable and not already placed on. Never relaxes.
    bool base[MAX_HEXES];
    for (int i = 0; i < n; ++i) {
        base[i] = state.passable(i) && state.city_placer[i] == NO_FACTION;
    }

    bool dist_ok[MAX_HEXES];
    for (int i = 0; i < n; ++i) dist_ok[i] = true;
    for (int p = 0; p < n; ++p) {
        if (state.city_placer[p] == NO_FACTION) continue;
        for (int i = 0; i < n; ++i) {
            if (grid.distance(i, p) < kCapitalMinDist) dist_ok[i] = false;
        }
    }

    // Tier 1: everything.
    for (int i = 0; i < n; ++i) {
        out[i] = base[i] && dist_ok[i] && !(ban_edges && grid.is_edge(i));
    }
    if (any_set(out, n)) return;

    // Tier 2: drop the edge ban.
    for (int i = 0; i < n; ++i) out[i] = base[i] && dist_ok[i];
    if (any_set(out, n)) return;

    // Tier 3: drop the distance rule too.
    for (int i = 0; i < n; ++i) out[i] = base[i];
}

void run_city_setup(GameState& state, const SetupDecisions& decisions, Rng& rng,
                    std::vector<PlacementLogEntry>* log) {
    const HexGrid& grid = *state.grid;
    const int num_factions = state.num_factions;

    auto log_entry = [&](PlacementLogEntry::Kind kind, int faction, int hex_index) {
        if (!log) return;
        const HexCoord& c = grid.coord_of(hex_index);
        log->push_back(PlacementLogEntry{kind, static_cast<int8_t>(faction), c.q, c.r, c.s});
    };

    // --- phase 1: colourless placement --------------------------------------
    const std::vector<int> placement_order = rng.sample_indices(num_factions, num_factions);
    bool legal[MAX_HEXES];
    for (int faction : placement_order) {
        legal_placement_mask(state, legal);
        int choice = decisions.placement(state, faction, legal, decisions.ctx);
        if (choice < 0 || choice >= state.num_hexes || !legal[choice]) {
            // Invalid answers fall back to a uniform legal choice rather than
            // raising - and the fallback consumes rng, so it is part of parity.
            SmallVec<int16_t, MAX_HEXES> options;
            for (int i = 0; i < state.num_hexes; ++i) {
                if (legal[i]) options.push_back(static_cast<int16_t>(i));
            }
            choice = options[static_cast<int>(rng.choice_index(static_cast<size_t>(options.size())))];
        }
        state.city_placer[choice] = static_cast<int8_t>(faction);
        log_entry(PlacementLogEntry::kPlace, faction, choice);
    }

    // --- phase 2: the draft --------------------------------------------------
    SmallVec<int16_t, MAX_HEXES> placed_hexes;
    for (int i = 0; i < state.num_hexes; ++i) {
        if (state.city_placer[i] != NO_FACTION) placed_hexes.push_back(static_cast<int16_t>(i));
    }

    const std::vector<int> draft_order = rng.sample_indices(num_factions, num_factions);
    int assigned[MAX_FACTIONS];
    for (int f = 0; f < MAX_FACTIONS; ++f) assigned[f] = -1;
    int settle_counter = 0;

    auto finalize = [&](int faction, int hex_index) {
        state.city_owner[hex_index] = static_cast<int8_t>(faction);
        state.is_capital[hex_index] = true;
        state.capital_settle_order[faction] = settle_counter++;
        assigned[faction] = hex_index;
    };

    for (int i = 0; i < num_factions; ++i) {
        const int faction = draft_order[static_cast<size_t>(i)];

        SmallVec<int16_t, MAX_HEXES> pool;
        for (int k = 0; k < placed_hexes.size(); ++k) {
            if (state.city_owner[placed_hexes[k]] == NO_FACTION) pool.push_back(placed_hexes[k]);
        }

        if (i < num_factions - 1) {
            SmallVec<int16_t, MAX_HEXES> legal_pool;
            for (int k = 0; k < pool.size(); ++k) {
                if (state.city_placer[pool[k]] != faction) legal_pool.push_back(pool[k]);
            }
            int choice = decisions.draft(state, faction, legal_pool.items, legal_pool.size(),
                                         decisions.ctx);
            bool ok = false;
            for (int k = 0; k < legal_pool.size(); ++k) {
                if (legal_pool[k] == choice) ok = true;
            }
            if (!ok) {
                choice = legal_pool[static_cast<int>(
                    rng.choice_index(static_cast<size_t>(legal_pool.size())))];
            }
            finalize(faction, choice);
            log_entry(PlacementLogEntry::kDraft, faction, choice);
        } else {
            const int leftover = pool[0];
            const int placer = state.city_placer[leftover];
            if (placer == faction) {
                // The only city left is the one they placed - no decision to make.
                finalize(faction, leftover);
                log_entry(PlacementLogEntry::kDraftAuto, faction, leftover);
            } else {
                const int placer_hex = assigned[placer];
                const bool swap =
                    decisions.swap(state, faction, leftover, placer, placer_hex, decisions.ctx);
                if (swap) {
                    if (log) {
                        const HexCoord& taken = grid.coord_of(placer_hex);
                        const HexCoord& bumped = grid.coord_of(leftover);
                        log->push_back(PlacementLogEntry{
                            PlacementLogEntry::kSwap, static_cast<int8_t>(faction), taken.q,
                            taken.r, taken.s, static_cast<int8_t>(placer), bumped.q, bumped.r,
                            bumped.s});
                    }
                    finalize(placer, leftover);
                    finalize(faction, placer_hex);
                } else {
                    finalize(faction, leftover);
                    log_entry(PlacementLogEntry::kKeep, faction, leftover);
                }
            }
        }
    }
}

}  // namespace oo
