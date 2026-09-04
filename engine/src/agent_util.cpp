#include "oo/agent_util.hpp"

#include <algorithm>
#include <cassert>

namespace oo {

void mobile_hexes(const GameState& state, const LegalMask& legal, HexList& out) {
    out.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (legal.any_for_hex(h)) out.push_back(static_cast<int16_t>(h));
    }
}

void mobile_hexes_by_size_desc(const GameState& state, const LegalMask& legal, HexList& out) {
    mobile_hexes(state, legal, out);
    // Stable, so equal-size armies keep ascending hex order - matching the
    // reference agents' np.argsort(..., kind="stable"). See the header note.
    std::stable_sort(out.items, out.items + out.count, [&](int16_t a, int16_t b) {
        return state.units_at(a) > state.units_at(b);
    });
}

double resource_bonus(const GameState& state, int hex_index) {
    const int16_t* nb = state.grid->neighbours_of(hex_index);
    bool has_mountain = false, has_lake = false;
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        if (nb[d] == -1) continue;
        if (state.terrain[nb[d]] == kMountain) has_mountain = true;
        if (state.terrain[nb[d]] == kLake) has_lake = true;
    }
    if (has_mountain && has_lake) return 2.0;
    if (has_mountain || has_lake) return 1.0;
    if (state.terrain[hex_index] == kDesert) return -2.0;
    return 0.0;
}

void enemy_outpost_coords(const GameState& state, int faction, CoordList& out) {
    out.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] != NO_FACTION && state.city_owner[h] != faction &&
            !state.is_capital[h]) {
            out.push_back(state.grid->coord_of(h));
        }
    }
}

void enemy_capital_coords(const GameState& state, int faction, CoordList& out) {
    out.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.is_capital[h] && state.city_owner[h] != NO_FACTION &&
            state.city_owner[h] != faction) {
            out.push_back(state.grid->coord_of(h));
        }
    }
}

int own_capital(const GameState& state, int faction) {
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] == faction && state.is_capital[h]) return h;
    }
    return -1;
}

void eligible_expansion_hexes(const GameState& state, int faction, HexList& out) {
    out.clear();
    bool mask[MAX_HEXES];
    eligible_outpost_mask(state, faction, mask);
    for (int h = 0; h < state.num_hexes; ++h) {
        if (mask[h] && state.passable(h)) out.push_back(static_cast<int16_t>(h));
    }
}

namespace {

// Distance from `coord` to the nearest of `targets`, and which one. First
// minimum wins, matching Python's min().
int nearest_target(const HexCoord& coord, const CoordList& targets, HexCoord& out) {
    int best = -1;
    for (int i = 0; i < targets.size(); ++i) {
        const int d = hex_distance(coord, targets[i]);
        if (best < 0 || d < best) {
            best = d;
            out = targets[i];
        }
    }
    return best;
}

}  // namespace

bool move_toward(const GameState& state, const HexList& ranked_origins, const LegalMask& legal,
                 const CoordList& targets, bool skip_arrived, Move& out) {
    if (targets.empty()) return false;
    const HexGrid& grid = *state.grid;

    for (int i = 0; i < ranked_origins.size(); ++i) {
        const int origin = ranked_origins[i];
        const HexCoord& origin_coord = grid.coord_of(origin);
        HexCoord target{};
        const int dist = nearest_target(origin_coord, targets, target);
        if (skip_arrived && dist == 0) continue;

        int best_dir = -1, best_dist = 0;
        for (int d = 0; d < NUM_DIRECTIONS; ++d) {
            if (!legal.cell[origin][d]) continue;
            const int dest = grid.neighbour(origin, d);
            const int dd = hex_distance(grid.coord_of(dest), target);
            if (best_dir < 0 || dd < best_dist) {  // strict: first minimum wins
                best_dir = d;
                best_dist = dd;
            }
        }
        if (best_dir < 0) continue;
        out.hex = static_cast<int16_t>(origin);
        out.dir = static_cast<int8_t>(best_dir);
        return true;
    }
    return false;
}

namespace {

struct DirScore {
    int dist;
    int bad_desert;
    int marsh;
    int tiebreak;

    bool operator<(const DirScore& o) const {
        if (dist != o.dist) return dist < o.dist;
        if (bad_desert != o.bad_desert) return bad_desert < o.bad_desert;
        if (marsh != o.marsh) return marsh < o.marsh;
        return tiebreak < o.tiebreak;
    }
};

DirScore score_direction(const GameState& state, int origin, int d, const HexCoord& target) {
    const HexGrid& grid = *state.grid;
    const int dest = grid.neighbour(origin, d);
    DirScore s{};
    s.dist = hex_distance(grid.coord_of(dest), target);
    s.bad_desert = (state.terrain[dest] == kDesert && state.city_owner[dest] == NO_FACTION) ? 1 : 0;
    s.marsh = (state.terrain[dest] == kMarsh) ? 1 : 0;
    s.tiebreak = direction_tiebreak(origin, dest, state.turn_number);
    return s;
}

}  // namespace

int best_direction(const GameState& state, int origin, const LegalMask& legal,
                   const HexCoord& target, int steps_remaining) {
    int best = -1;
    DirScore best_score{};
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        if (!legal.cell[origin][d]) continue;
        const DirScore s = score_direction(state, origin, d, target);
        if (best < 0 || s < best_score) {  // strict: first minimum wins
            best = d;
            best_score = s;
        }
    }
    if (best < 0) return -1;
    if (steps_remaining <= 0) return best;
    if (!best_score.marsh) return best;

    // The chosen step lands in a marsh, which freezes the army for the rest of
    // the turn. Take a non-marsh option instead if it costs little extra.
    int alt = -1;
    DirScore alt_score{};
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        if (!legal.cell[origin][d]) continue;
        const DirScore s = score_direction(state, origin, d, target);
        if (s.marsh) continue;
        if (alt < 0 || s < alt_score) {
            alt = d;
            alt_score = s;
        }
    }
    if (alt < 0) return best;
    return (alt_score.dist <= best_score.dist + kMarshDetourTolerance) ? alt : best;
}

void ranked_expansion_targets(const GameState& state, int faction, CoordList& out) {
    out.clear();
    if (outpost_count(state, faction) >= kOutpostCap) return;
    const int capital = own_capital(state, faction);
    if (capital < 0) return;

    HexList eligible;
    eligible_expansion_hexes(state, faction, eligible);
    if (eligible.empty()) return;

    const HexCoord& capital_coord = state.grid->coord_of(capital);
    SmallVec<double, MAX_HEXES> scores;
    scores.clear();
    for (int i = 0; i < eligible.size(); ++i) {
        const int h = eligible[i];
        scores.push_back(hex_distance(state.grid->coord_of(h), capital_coord) -
                         kExpansionResourceWeight * resource_bonus(state, h));
    }
    // Sort indices so the score stays paired with its hex; stable, so equal
    // scores keep ascending hex order.
    SmallVec<int16_t, MAX_HEXES> order;
    order.clear();
    for (int i = 0; i < eligible.size(); ++i) order.push_back(static_cast<int16_t>(i));
    std::stable_sort(order.items, order.items + order.count,
                     [&](int16_t a, int16_t b) { return scores[a] < scores[b]; });

    const int k = std::min(kExpansionObjectives, order.size());
    for (int i = 0; i < k; ++i) out.push_back(state.grid->coord_of(eligible[order[i]]));
}

void ranked_attack_targets(const GameState& state, int faction, CoordList& out) {
    out.clear();
    const int capital = own_capital(state, faction);
    if (capital < 0) return;

    HexList outposts;
    outposts.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] != NO_FACTION && state.city_owner[h] != faction &&
            !state.is_capital[h]) {
            outposts.push_back(static_cast<int16_t>(h));
        }
    }
    if (outposts.empty()) return;

    const HexCoord& capital_coord = state.grid->coord_of(capital);
    std::stable_sort(outposts.items, outposts.items + outposts.count, [&](int16_t a, int16_t b) {
        return hex_distance(state.grid->coord_of(a), capital_coord) <
               hex_distance(state.grid->coord_of(b), capital_coord);
    });

    const int k = std::min(kAttackObjectives, outposts.size());
    for (int i = 0; i < k; ++i) out.push_back(state.grid->coord_of(outposts[i]));
}

void all_targets(const GameState& state, int faction, CoordList& out) {
    CoordList expansion, attack;
    ranked_expansion_targets(state, faction, expansion);
    ranked_attack_targets(state, faction, attack);

    out.clear();
    for (int i = 0; i < expansion.size(); ++i) out.push_back(expansion[i]);
    for (int i = 0; i < attack.size(); ++i) out.push_back(attack[i]);
    if (!out.empty()) return;

    // Nothing to expand to and nothing to raid: march on enemy capitals.
    enemy_capital_coords(state, faction, out);
}

void greedy_match(const GameState& state, const HexList& origins, const CoordList& targets,
                  MatchList& out) {
    out.clear();
    const HexGrid& grid = *state.grid;

    struct Pair {
        int dist;
        int16_t origin_i;
        int16_t target_i;
    };
    // A faction can field at most 24 + 12 + 12 = 48 units, so at most 48 hexes
    // can hold one of its armies; targets are capped by CoordList. That bounds
    // the pair list without a heap allocation.
    static constexpr int kMaxOrigins = 48;
    SmallVec<Pair, kMaxOrigins * 128> pairs;
    pairs.clear();
    assert(origins.size() <= kMaxOrigins && "more mobile armies than units can exist");
    // Generation order is origin-major, then target - and since the sort below
    // keys on DISTANCE ONLY, that generation order is what breaks every tie.
    for (int o = 0; o < origins.size(); ++o) {
        for (int t = 0; t < targets.size(); ++t) {
            pairs.push_back(Pair{hex_distance(grid.coord_of(origins[o]), targets[t]),
                                 static_cast<int16_t>(o), static_cast<int16_t>(t)});
        }
    }
    std::stable_sort(pairs.items, pairs.items + pairs.count,
                     [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

    bool used_origin[kMaxOrigins] = {};
    bool used_target[128] = {};
    for (int i = 0; i < pairs.size(); ++i) {
        const Pair& p = pairs[i];
        if (used_origin[p.origin_i] || used_target[p.target_i]) continue;
        out.push_back(MatchPair{origins[p.origin_i], targets[p.target_i]});
        used_origin[p.origin_i] = true;
        used_target[p.target_i] = true;
    }
}

}  // namespace oo
