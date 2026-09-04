#include "oo/setup.hpp"

#include <cstring>
#include <vector>

namespace oo {

namespace {

constexpr int8_t kUnset = -1;

bool is_impassable_type(int8_t t) {
    return t == kMountain || t == kLake;
}

int same_type_neighbour_count(const HexGrid& grid, const int8_t* terrain, int index,
                              int8_t type_index) {
    int count = 0;
    const int16_t* nb = grid.neighbours_of(index);
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        if (nb[d] != -1 && terrain[nb[d]] == type_index) ++count;
    }
    return count;
}

// True if marking `candidate` impassable would split the rest of the board's
// non-impassable hexes into more than one connected component - i.e. wall part of
// the board off into an unreachable island.
//
// Unset hexes count as non-impassable: every unset hex is guaranteed to become
// some terrain before generation ends, and only mountain/lake are ever impassable,
// so "currently non-impassable" is exactly "will end up passable, eventually".
//
// A plain DFS re-run from scratch each call is fine - this only runs during
// one-off map generation, never in the per-turn hot path. Being a pure reachability
// question, iteration order cannot affect the answer, so there is no parity hazard.
bool would_disconnect(const HexGrid& grid, const int8_t* terrain, int candidate,
                      std::vector<uint8_t>& passable, std::vector<uint8_t>& seen,
                      std::vector<int16_t>& stack) {
    const int n = grid.num_hexes();
    int total_passable = 0;
    int first = -1;
    passable.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        if (i == candidate || is_impassable_type(terrain[i])) continue;
        passable[static_cast<size_t>(i)] = 1;
        ++total_passable;
        if (first < 0) first = i;
    }
    if (total_passable == 0) return false;

    seen.assign(static_cast<size_t>(n), 0);
    stack.clear();
    seen[static_cast<size_t>(first)] = 1;
    stack.push_back(static_cast<int16_t>(first));
    int reached = 1;
    while (!stack.empty()) {
        const int h = stack.back();
        stack.pop_back();
        const int16_t* nb = grid.neighbours_of(h);
        for (int d = 0; d < NUM_DIRECTIONS; ++d) {
            const int j = nb[d];
            if (j == -1 || !passable[static_cast<size_t>(j)] || seen[static_cast<size_t>(j)]) {
                continue;
            }
            seen[static_cast<size_t>(j)] = 1;
            ++reached;
            stack.push_back(static_cast<int16_t>(j));
        }
    }
    return reached != total_passable;
}

// Whether `index` may become the (placed_so_far + 1)-th hex of this round.
bool can_place(const HexGrid& grid, const int8_t* terrain, int index, int8_t type_index,
               int placed_so_far, bool check_disconnect, std::vector<uint8_t>& passable,
               std::vector<uint8_t>& seen, std::vector<int16_t>& stack) {
    bool shape_ok;
    if (type_index == kMountain) {
        // The first mountain of a round is free; every one after it must extend
        // the chain by exactly one link, keeping mountains linear.
        shape_ok = (placed_so_far == 0) || same_type_neighbour_count(grid, terrain, index,
                                                                    type_index) == 1;
    } else if (type_index == kLake || type_index == kMarsh || type_index == kDesert) {
        // First two are free; from the third on, the hex must grow an existing
        // body rather than sprout a new one.
        shape_ok = (placed_so_far < 2) || same_type_neighbour_count(grid, terrain, index,
                                                                    type_index) >= 2;
    } else {
        shape_ok = true;
    }
    if (!shape_ok) return false;
    if (check_disconnect && is_impassable_type(type_index) &&
        would_disconnect(grid, terrain, index, passable, seen, stack)) {
        return false;
    }
    return true;
}

}  // namespace

void generate_terrain(const HexGrid& grid, Rng& rng, int8_t* terrain,
                      std::vector<TerrainLogEntry>* log) {
    const int n = grid.num_hexes();
    std::memset(terrain, kUnset, static_cast<size_t>(n));

    std::vector<uint8_t> unset(static_cast<size_t>(n), 1);
    int n_unset = n;

    int32_t bag[kBagSize];
    for (int i = 0; i < kBagSize; ++i) bag[i] = kBagCounts[i];

    // Scratch buffers, reused across the many would_disconnect calls.
    std::vector<uint8_t> passable, seen;
    std::vector<int16_t> stack;
    std::vector<int16_t> candidates;
    std::vector<uint8_t> is_candidate(static_cast<size_t>(n), 0);
    std::vector<int16_t> placed;

    // First round seeds from a random edge hex.
    candidates.clear();
    for (int i = 0; i < n; ++i) {
        if (grid.is_edge(i)) candidates.push_back(static_cast<int16_t>(i));
    }
    int start = candidates[rng.choice_index(candidates.size())];

    int round_index = 0;
    int stuck_rounds = 0;
    const int max_stuck_rounds = n * 4;
    bool check_disconnect = true;

    while (n_unset > 0) {
        // Draw a type, weighted by how much of it is left. The bag order here IS
        // engine_old's dict order - see the header's parity note.
        int types[kBagSize];
        int n_types = 0;
        std::vector<double> weights;
        for (int i = 0; i < kBagSize; ++i) {
            if (bag[i] > 0) {
                types[n_types++] = i;
                weights.push_back(static_cast<double>(bag[i]));
            }
        }
        const int bag_slot = types[rng.choices_index(weights)];
        const int8_t type_index = kBagOrder[bag_slot];

        // --- one round -----------------------------------------------------
        placed.clear();
        const bool seed_blocked =
            check_disconnect && is_impassable_type(type_index) &&
            would_disconnect(grid, terrain, start, passable, seen, stack);

        if (!seed_blocked) {
            // NOTE: the randint below is only consumed when the round actually
            // starts. A round blocked at its seed hex returns without drawing,
            // and that difference in RNG consumption is load-bearing.
            const int lo = kRoundCountLo[type_index];
            const int hi = kRoundCountHi[type_index];
            int target = static_cast<int>(rng.randint(lo, hi));
            if (bag[bag_slot] < target) target = bag[bag_slot];

            auto place = [&](int index) {
                terrain[index] = type_index;
                if (log) {
                    const HexCoord& c = grid.coord_of(index);
                    log->push_back(TerrainLogEntry{c.q, c.r, c.s, type_index, round_index});
                }
            };

            place(start);
            placed.push_back(static_cast<int16_t>(start));

            while (static_cast<int>(placed.size()) < target) {
                // Ascending hex order - engine_old sorts this list precisely so the
                // choice below is a property of the data rather than of CPython's
                // set layout (PLAN.md §3.2).
                for (int16_t h : placed) {
                    const int16_t* nb = grid.neighbours_of(h);
                    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
                        if (nb[d] != -1 && terrain[nb[d]] == kUnset) {
                            is_candidate[static_cast<size_t>(nb[d])] = 1;
                        }
                    }
                }
                candidates.clear();
                for (int i = 0; i < n; ++i) {
                    if (!is_candidate[static_cast<size_t>(i)]) continue;
                    is_candidate[static_cast<size_t>(i)] = 0;
                    if (can_place(grid, terrain, i, type_index, static_cast<int>(placed.size()),
                                  check_disconnect, passable, seen, stack)) {
                        candidates.push_back(static_cast<int16_t>(i));
                    }
                }
                if (candidates.empty()) break;
                const int choice = candidates[rng.choice_index(candidates.size())];
                place(choice);
                placed.push_back(static_cast<int16_t>(choice));
            }
            bag[bag_slot] -= static_cast<int32_t>(placed.size());
        }
        // -------------------------------------------------------------------

        ++round_index;
        if (!placed.empty()) {
            stuck_rounds = 0;
            for (int16_t h : placed) {
                if (unset[static_cast<size_t>(h)]) {
                    unset[static_cast<size_t>(h)] = 0;
                    --n_unset;
                }
            }
        } else {
            ++stuck_rounds;
            // Circuit breaker: if generation is genuinely wedged (every passable
            // type spent while unset hexes remain, forcing mountain/lake attempts
            // that keep landing somewhere disconnecting), stop enforcing the island
            // check. An occasional island beats a hang.
            if (stuck_rounds >= max_stuck_rounds) check_disconnect = false;
        }

        if (n_unset > 0) {
            candidates.clear();
            for (int i = 0; i < n; ++i) {
                if (!unset[static_cast<size_t>(i)]) continue;
                const int16_t* nb = grid.neighbours_of(i);
                bool touches = false;
                for (int d = 0; d < NUM_DIRECTIONS; ++d) {
                    if (nb[d] != -1 && terrain[nb[d]] != kUnset) touches = true;
                }
                if (touches) candidates.push_back(static_cast<int16_t>(i));
            }
            start = candidates[rng.choice_index(candidates.size())];
        }
    }
}

void create_initial_state(GameState& state, int radius, int num_factions, int64_t seed,
                          std::vector<TerrainLogEntry>* terrain_log) {
    Rng rng(seed);
    const HexGrid& grid = HexGrid::shared(radius);
    new_empty(state, grid, num_factions);

    generate_terrain(grid, rng, state.terrain, terrain_log);

    for (int f = 0; f < num_factions; ++f) {
        state.gold[f] = kStartingGold;
        state.kill_xp[f] = kStartingKillXp;
    }
}

}  // namespace oo
