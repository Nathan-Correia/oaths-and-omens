// The five native agents of M6a: random, greedy, heuristic, vanguard, marshal.
//
// Ported from agents/*.py. They form a chain - greedy reuses random's targeting
// and rectification, heuristic reuses greedy's buy and setup policies, vanguard
// and marshal reuse heuristic's targeting and greedy's buy - so the shared pieces
// live here as free functions and each agent class is thin wiring, mirroring the
// Python layout.
//
// See agent_util.hpp for the tie-breaking rules every one of these depends on.
// The long design journals in agents/*.py record which ideas were tried and
// rejected, with measured win rates; the ones that explain a specific line of
// code are reproduced here, but the full history stays in the Python files until
// they are retired at M8.

#include "oo/agent.hpp"

#include "oo/agent_util.hpp"
#include "oo/battle.hpp"
#include "oo/buy.hpp"
#include "oo/movement.hpp"
#include "oo/terrain.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

namespace oo {

namespace {

// ============================ random ====================================

inline constexpr double kSkipChance = 0.5;  // chance to move nothing this step

bool random_movement(Rng& rng, const GameState& state, const LegalMask& legal, Move& out) {
    if (rng.random() < kSkipChance) return false;
    // Flattened (hex, dir) cells in row-major order, matching np.nonzero.
    SmallVec<int32_t, MAX_HEXES * NUM_DIRECTIONS> cells;
    cells.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        for (int d = 0; d < NUM_DIRECTIONS; ++d) {
            if (legal.cell[h][d]) cells.push_back(h * NUM_DIRECTIONS + d);
        }
    }
    if (cells.empty()) return false;
    const int pick = cells[static_cast<int>(rng.randbelow(static_cast<uint64_t>(cells.size())))];
    out.hex = static_cast<int16_t>(pick / NUM_DIRECTIONS);
    out.dir = static_cast<int8_t>(pick % NUM_DIRECTIONS);
    return true;
}

int random_target(Rng& rng, const GameState& state, int hex_index, int faction) {
    SmallVec<int8_t, MAX_FACTIONS> legal;
    get_legal_target_actions(state, hex_index, faction, legal);
    if (legal.empty()) return -1;
    return legal[static_cast<int>(rng.choice_index(static_cast<size_t>(legal.size())))];
}

void random_rectification(Rng& rng, const GameState& state, int hex_index, int winner, int cap,
                          SendBack& out) {
    out.clear();
    FactionTotals totals;
    faction_totals(state, hex_index, totals);
    const int wi = totals.index_of(winner);
    if (wi < 0) return;
    int overflow = totals.total_for(wi) - cap;
    if (overflow <= 0) return;

    SmallVec<int32_t, MAX_BATTLE_CONTRIB> origins;
    origins.clear();
    if (const Battle* b = state.battle_at(hex_index)) {
        for (int k = 0; k < b->nslots; ++k) {
            if (b->slots[k].faction == winner) origins.push_back(b->slots[k].origin);
        }
    }
    if (origins.empty()) return;

    int remaining[NUM_UNIT_TYPES];
    for (int t = 0; t < NUM_UNIT_TYPES; ++t) remaining[t] = totals.units[wi][t];
    for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
        while (overflow > 0 && remaining[t] > 0) {
            SendBackEntry e{};
            e.origin_hex = origins[static_cast<int>(
                rng.choice_index(static_cast<size_t>(origins.size())))];
            e.units[t] = 1;
            out.push_back(e);
            --remaining[t];
            --overflow;
        }
    }
}

void random_buy(Rng& rng, const LegalBuyActions& legal, ChosenBuyActions& out) {
    out.clear();
    if (legal.empty()) return;
    const int max_actions = std::min(3, legal.size());
    const int n = static_cast<int>(rng.randint(0, max_actions));
    if (n <= 0) return;
    for (int idx : rng.sample_indices(legal.size(), n)) out.push_back(legal[idx]);
}

// ============================ greedy ====================================

// Which unit to sacrifice first when founding an outpost, and which upgrade to
// grab first when several are affordable: Temple (direct VP - the actual win
// condition), then Barracks (compounds economy), then Workshop (compounds
// resources). A "greedy" ranking by how directly each pays off.
constexpr int kOutpostUnitPriority[NUM_UNIT_TYPES] = {kInfantry, kCavalry, kArchers};
constexpr int kUpgradePriority[NUM_UPGRADE_TYPES] = {kTemple, kBarracks, kWorkshop};

// Shared by greedy and hussar; they differ only in how conversions choose a
// unit type (see the `hussar_conversions` branch at the bottom).
void greedy_buy_common(Rng& rng, const GameState& state, int faction,
                       const LegalBuyActions& legal, ChosenBuyActions& out,
                       bool hussar_conversions) {
    out.clear();

    // Bucket by kind, preserving `legal`'s order.
    SmallVec<int16_t, kMaxLegalBuy> outpost_actions, upgrade_actions, infantry_actions,
        convert_actions;
    outpost_actions.clear();
    upgrade_actions.clear();
    infantry_actions.clear();
    convert_actions.clear();
    for (int i = 0; i < legal.size(); ++i) {
        switch (legal[i].type) {
            case BuyType::kBuildOutpost:
                outpost_actions.push_back(static_cast<int16_t>(i));
                break;
            case BuyType::kUpgradeOutpost:
                // Only ever gives a BARE outpost its first upgrade. The engine
                // also offers converting an already-upgraded outpost to a
                // different one, but re-paying full price to swap is not worth
                // it for a strategy this simple.
                if (state.outpost_upgrade[legal[i].hex] == NO_UPGRADE) {
                    upgrade_actions.push_back(static_cast<int16_t>(i));
                }
                break;
            case BuyType::kBuyInfantry:
                infantry_actions.push_back(static_cast<int16_t>(i));
                break;
            case BuyType::kConvertToSpecial:
                convert_actions.push_back(static_cast<int16_t>(i));
                break;
        }
    }

    // Only ONE outpost action goes through per turn (buy.cpp's apply_buy_phase),
    // so upgrading ground already held beats founding new ground.
    auto pick_by_hex = [&](const SmallVec<int16_t, kMaxLegalBuy>& bucket, bool is_upgrade) {
        // Distinct hexes in first-appearance order, matching Python's dict keys.
        SmallVec<int16_t, kMaxLegalBuy> hexes;
        hexes.clear();
        for (int i = 0; i < bucket.size(); ++i) {
            const int16_t h = legal[bucket[i]].hex;
            bool seen = false;
            for (int j = 0; j < hexes.size(); ++j) {
                if (hexes[j] == h) seen = true;
            }
            if (!seen) hexes.push_back(h);
        }
        if (hexes.empty()) return;

        // rng.shuffle over the hex list, then take the first. Shuffled in place
        // on the fixed buffer - no allocation in the turn loop.
        rng.shuffle(hexes.items, static_cast<size_t>(hexes.count));
        const int16_t chosen_hex = hexes[0];

        const int n_pref = is_upgrade ? NUM_UPGRADE_TYPES : NUM_UNIT_TYPES;
        for (int p = 0; p < n_pref; ++p) {
            const int want = is_upgrade ? kUpgradePriority[p] : kOutpostUnitPriority[p];
            for (int i = 0; i < bucket.size(); ++i) {
                const BuyAction& a = legal[bucket[i]];
                if (a.hex != chosen_hex) continue;
                if (is_upgrade ? (a.upgrade == want) : (a.unit_type == want)) {
                    out.push_back(a);
                    return;
                }
            }
        }
    };

    if (!upgrade_actions.empty()) {
        pick_by_hex(upgrade_actions, /*is_upgrade=*/true);
    } else {
        pick_by_hex(outpost_actions, /*is_upgrade=*/false);
    }

    if (!infantry_actions.empty()) {
        const int num_purchases = state.gold[faction] / kInfantryCost;
        for (int i = 0; i < num_purchases; ++i) {
            out.push_back(legal[infantry_actions[static_cast<int>(
                rng.choice_index(static_cast<size_t>(infantry_actions.size())))]]);
        }
    }

    if (!convert_actions.empty()) {
        const int num_conversions = state.kill_xp[faction];
        int cav_count = count_units_in_play(state, faction, kCavalry);
        const int arc_count = count_units_in_play(state, faction, kArchers);

        for (int i = 0; i < num_conversions; ++i) {
            if (out.size() >= out.capacity()) break;
            int want;
            if (hussar_conversions) {
                // hussar wants cavalry until the concurrent cap, and DOES update
                // the running count, so one batch can switch to archers midway.
                want = (cav_count < kSpawnCaps[kCavalry]) ? kCavalry : kArchers;
            } else {
                // greedy balances the two, and counts ONCE before the loop -
                // never updating - so every conversion this turn picks the same
                // type. Ported as-is.
                want = (cav_count <= arc_count) ? kCavalry : kArchers;
            }

            SmallVec<int16_t, kMaxLegalBuy> matching;
            matching.clear();
            for (int j = 0; j < convert_actions.size(); ++j) {
                if (legal[convert_actions[j]].unit_type == want) {
                    matching.push_back(convert_actions[j]);
                }
            }
            const SmallVec<int16_t, kMaxLegalBuy>& pool =
                matching.empty() ? convert_actions : matching;
            out.push_back(
                legal[pool[static_cast<int>(rng.choice_index(static_cast<size_t>(pool.size())))]]);
            if (hussar_conversions && want == kCavalry) ++cav_count;
        }
    }
}

void greedy_buy(Rng& rng, const GameState& state, int faction, const LegalBuyActions& legal,
                ChosenBuyActions& out) {
    greedy_buy_common(rng, state, faction, legal, out, /*hussar_conversions=*/false);
}

void hussar_buy(Rng& rng, const GameState& state, int faction, const LegalBuyActions& legal,
                ChosenBuyActions& out) {
    greedy_buy_common(rng, state, faction, legal, out, /*hussar_conversions=*/true);
}

// Distance from `hex_index` to the nearest of `others`.
int nearest_dist(const GameState& state, int hex_index, const HexList& others) {
    int best = -1;
    for (int i = 0; i < others.size(); ++i) {
        const int d = state.grid->distance(hex_index, others[i]);
        if (best < 0 || d < best) best = d;
    }
    return best;
}

int greedy_placement(const GameState& state, const bool legal[MAX_HEXES]) {
    HexList candidates, placed;
    candidates.clear();
    placed.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (legal[h]) candidates.push_back(static_cast<int16_t>(h));
        if (state.city_placer[h] != NO_FACTION) placed.push_back(static_cast<int16_t>(h));
    }
    if (candidates.empty()) return -1;
    if (placed.empty()) return candidates[0];

    int best = candidates[0], best_d = nearest_dist(state, candidates[0], placed);
    for (int i = 1; i < candidates.size(); ++i) {
        const int d = nearest_dist(state, candidates[i], placed);
        if (d > best_d) {  // strict: first maximum wins
            best_d = d;
            best = candidates[i];
        }
    }
    return best;
}

int greedy_draft(const GameState& state, const int16_t* pool, int pool_size) {
    HexList claimed;
    claimed.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] != NO_FACTION) claimed.push_back(static_cast<int16_t>(h));
    }
    if (pool_size <= 0) return -1;
    if (claimed.empty()) return pool[0];

    int best = pool[0], best_d = nearest_dist(state, pool[0], claimed);
    for (int i = 1; i < pool_size; ++i) {
        const int d = nearest_dist(state, pool[i], claimed);
        if (d > best_d) {
            best_d = d;
            best = pool[i];
        }
    }
    return best;
}

Resource greedy_resource_choice(const GameState& state, int faction) {
    return (state.resources[faction][kIron] <= state.resources[faction][kFish]) ? kIron : kFish;
}

bool greedy_swap(const GameState& state, int leftover_hex, int placer_hex) {
    HexList others;
    others.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] != NO_FACTION && h != leftover_hex && h != placer_hex) {
            others.push_back(static_cast<int16_t>(h));
        }
    }
    if (others.empty()) return false;
    return nearest_dist(state, placer_hex, others) > nearest_dist(state, leftover_hex, others);
}

// greedy_agent's _home_expansion_target: the eligible site closest to our own
// capital, by Chebyshev distance over cube coordinates.
bool home_expansion_target(const GameState& state, int faction, HexCoord& out) {
    if (outpost_count(state, faction) >= kOutpostCap) return false;
    const int capital = own_capital(state, faction);
    if (capital < 0) return false;
    HexList eligible;
    eligible_expansion_hexes(state, faction, eligible);
    if (eligible.empty()) return false;

    int best = eligible[0];
    int best_d = state.grid->distance(eligible[0], capital);
    for (int i = 1; i < eligible.size(); ++i) {
        const int d = state.grid->distance(eligible[i], capital);
        if (d < best_d) {  // np.argmin: first minimum wins
            best_d = d;
            best = eligible[i];
        }
    }
    out = state.grid->coord_of(best);
    return true;
}

bool greedy_rush_move(const GameState& state, int faction, const LegalMask& legal, Move& out) {
    HexList ranked;
    mobile_hexes_by_size_desc(state, legal, ranked);
    if (ranked.empty()) return false;

    HexCoord home{};
    if (home_expansion_target(state, faction, home)) {
        CoordList targets;
        targets.clear();
        targets.push_back(home);
        if (move_toward(state, ranked, legal, targets, /*skip_arrived=*/true, out)) return true;
        // Every mobile army is already parked at the target, or none can legally
        // step toward it - fall through to attacking rather than doing nothing.
    }

    CoordList targets;
    enemy_outpost_coords(state, faction, targets);
    if (targets.empty()) enemy_capital_coords(state, faction, targets);
    if (targets.empty()) return false;
    return move_toward(state, ranked, legal, targets, /*skip_arrived=*/false, out);
}

// ============================ heuristic =================================

inline constexpr int kExpansionTolerance = 2;
inline constexpr int kAttackTolerance = 2;
inline constexpr double kOutpostDefensePower = 0.5;  // ~expected kills from one free shot

int heuristic_target(const GameState& state, int hex_index, int faction) {
    SmallVec<int8_t, MAX_FACTIONS> legal;
    get_legal_target_actions(state, hex_index, faction, legal);
    if (legal.empty()) return -1;
    FactionTotals totals;
    faction_totals(state, hex_index, totals);

    int best = legal[0];
    int best_total = totals.total_for(totals.index_of(legal[0]));
    for (int i = 1; i < legal.size(); ++i) {
        const int t = totals.total_for(totals.index_of(legal[i]));
        if (t < best_total) {  // strict: first minimum wins - go for the weakest
            best_total = t;
            best = legal[i];
        }
    }
    return best;
}

// Distance-dominated tie-break, not a free-form score: among sites within
// kExpansionTolerance of the nearest, prefer the resource-richest.
bool best_expansion_target(const GameState& state, int faction, HexCoord& out) {
    if (outpost_count(state, faction) >= kOutpostCap) return false;
    const int capital = own_capital(state, faction);
    if (capital < 0) return false;
    HexList eligible;
    eligible_expansion_hexes(state, faction, eligible);
    if (eligible.empty()) return false;

    int min_dist = state.grid->distance(eligible[0], capital);
    for (int i = 1; i < eligible.size(); ++i) {
        min_dist = std::min<int>(min_dist, state.grid->distance(eligible[i], capital));
    }

    int best = -1;
    double best_score = 0.0;
    for (int i = 0; i < eligible.size(); ++i) {
        const int d = state.grid->distance(eligible[i], capital);
        if (d > min_dist + kExpansionTolerance) continue;
        const double score = -static_cast<double>(d) + resource_bonus(state, eligible[i]);
        if (best < 0 || score > best_score) {  // strict: first maximum wins
            best_score = score;
            best = eligible[i];
        }
    }
    if (best < 0) return false;
    out = state.grid->coord_of(best);
    return true;
}

// Among enemy outposts within kAttackTolerance of the nearest (measured from our
// biggest army), pick the most weakly defended.
bool best_attack_target(const GameState& state, int faction, CoordList& out) {
    out.clear();
    HexList origins;
    origins.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.army_faction[h] == faction && !state.locked(h)) {
            origins.push_back(static_cast<int16_t>(h));
        }
    }
    if (origins.empty()) return false;

    int ref = origins[0], ref_size = state.units_at(origins[0]);
    for (int i = 1; i < origins.size(); ++i) {
        const int s = state.units_at(origins[i]);
        if (s > ref_size) {  // np.argmax: first maximum wins
            ref_size = s;
            ref = origins[i];
        }
    }
    const HexCoord& ref_coord = state.grid->coord_of(ref);

    HexList outposts;
    outposts.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] != NO_FACTION && state.city_owner[h] != faction &&
            !state.is_capital[h]) {
            outposts.push_back(static_cast<int16_t>(h));
        }
    }

    if (!outposts.empty()) {
        int min_dist = hex_distance(ref_coord, state.grid->coord_of(outposts[0]));
        for (int i = 1; i < outposts.size(); ++i) {
            min_dist = std::min(min_dist, hex_distance(ref_coord, state.grid->coord_of(outposts[i])));
        }
        int best = -1;
        double best_power = 0.0;
        for (int i = 0; i < outposts.size(); ++i) {
            const int d = hex_distance(ref_coord, state.grid->coord_of(outposts[i]));
            if (d > min_dist + kAttackTolerance) continue;
            const double p = army_power(state.army_units[outposts[i]]) + kOutpostDefensePower;
            if (best < 0 || p < best_power) {  // strict: first minimum wins
                best_power = p;
                best = outposts[i];
            }
        }
        if (best >= 0) {
            out.push_back(state.grid->coord_of(best));
            return true;
        }
    }

    enemy_capital_coords(state, faction, out);
    return !out.empty();
}

bool heuristic_move(const GameState& state, int faction, const LegalMask& legal, Move& out) {
    HexList ranked;
    mobile_hexes_by_size_desc(state, legal, ranked);
    if (ranked.empty()) return false;

    HexCoord home{};
    if (best_expansion_target(state, faction, home)) {
        CoordList targets;
        targets.clear();
        targets.push_back(home);
        if (move_toward(state, ranked, legal, targets, /*skip_arrived=*/true, out)) return true;
    }

    CoordList attack;
    if (!best_attack_target(state, faction, attack)) return false;
    return move_toward(state, ranked, legal, attack, /*skip_arrived=*/false, out);
}

// ============================ vanguard / marshal ========================

// Rotates through mobile armies by step so up to min(3, army count) DISTINCT
// armies act over a turn, rather than the biggest one hogging every step.
bool vanguard_move(const GameState& state, int faction, int step, const LegalMask& legal,
                   int total_steps, Move& out) {
    HexList mobile;
    mobile_hexes(state, legal, mobile);
    if (mobile.empty()) return false;
    CoordList targets;
    all_targets(state, faction, targets);
    if (targets.empty()) return false;

    const int steps_remaining = total_steps - step - 1;
    const int n = mobile.size();
    for (int offset = 0; offset < n; ++offset) {
        const int origin = mobile[(step + offset) % n];
        const HexCoord& origin_coord = state.grid->coord_of(origin);

        HexCoord target{};
        int best_d = -1;
        for (int i = 0; i < targets.size(); ++i) {
            const int d = hex_distance(origin_coord, targets[i]);
            if (best_d < 0 || d < best_d) {
                best_d = d;
                target = targets[i];
            }
        }
        if (best_d == 0) continue;  // already on its nearest target - leave it for the buy phase
        const int dir = best_direction(state, origin, legal, target, steps_remaining);
        if (dir < 0) continue;
        out.hex = static_cast<int16_t>(origin);
        out.dir = static_cast<int8_t>(dir);
        return true;
    }
    return false;
}

// Like vanguard, but assigns armies to targets by greedy bipartite matching
// first, so two armies do not chase the same objective.
bool marshal_move(const GameState& state, int faction, int step, const LegalMask& legal,
                  int total_steps, Move& out) {
    HexList mobile;
    mobile_hexes(state, legal, mobile);
    if (mobile.empty()) return false;
    CoordList targets;
    all_targets(state, faction, targets);
    if (targets.empty()) return false;

    MatchList matches;
    greedy_match(state, mobile, targets, matches);
    if (matches.empty()) return false;

    const int steps_remaining = total_steps - step - 1;
    const int n = matches.size();
    for (int offset = 0; offset < n; ++offset) {
        const MatchPair& m = matches[(step + offset) % n];
        const int origin = m.origin;
        if (hex_distance(state.grid->coord_of(origin), m.target) == 0) continue;
        const int dir = best_direction(state, origin, legal, m.target, steps_remaining);
        if (dir < 0) continue;
        out.hex = static_cast<int16_t>(origin);
        out.dir = static_cast<int8_t>(dir);
        return true;
    }
    return false;
}

// ============================ M6b leaf agents ===========================

// turtle: expand and nothing else. With no expansion site it simply does not
// move - no attacking fallback, unlike greedy/heuristic.
bool turtle_move(const GameState& state, int faction, const LegalMask& legal, Move& out) {
    HexList ranked;
    mobile_hexes_by_size_desc(state, legal, ranked);
    if (ranked.empty()) return false;
    HexCoord home{};
    if (!best_expansion_target(state, faction, home)) return false;
    CoordList targets;
    targets.clear();
    targets.push_back(home);
    return move_toward(state, ranked, legal, targets, /*skip_arrived=*/true, out);
}

// The single faction with the strictly highest VP, or -1 if that is tied or is us.
int current_leader(const GameState& state, int faction) {
    int top = state.victory_points[0];
    for (int f = 1; f < state.num_factions; ++f) top = std::max(top, state.victory_points[f]);
    int leader = -1, count = 0;
    for (int f = 0; f < state.num_factions; ++f) {
        if (state.victory_points[f] == top) {
            ++count;
            leader = f;
        }
    }
    if (count != 1 || leader == faction) return -1;
    return leader;
}

inline constexpr int kDenyMinOwnOutposts = 1;
// How far a denial detour may go, as a fraction of the board radius. Scaled BY
// radius rather than a flat hex count: on a big board "go attack whoever is
// winning" can mean a trek across the map, burning turns of tempo that
// always-attack-nearest never pays.
inline constexpr double kMaxDenyDistanceFactor = 0.6;

// The leader's weakest nearby outpost, measured from our biggest mobile army.
bool leader_attack_target(const GameState& state, int faction, CoordList& out) {
    out.clear();
    const int leader = current_leader(state, faction);
    if (leader < 0) return false;

    HexList origins;
    origins.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.army_faction[h] == faction && !state.locked(h)) {
            origins.push_back(static_cast<int16_t>(h));
        }
    }
    if (origins.empty()) return false;
    int ref = origins[0], ref_size = state.units_at(origins[0]);
    for (int i = 1; i < origins.size(); ++i) {
        const int s = state.units_at(origins[i]);
        if (s > ref_size) {  // np.argmax: first maximum wins
            ref_size = s;
            ref = origins[i];
        }
    }
    const HexCoord& ref_coord = state.grid->coord_of(ref);

    HexList outposts;
    outposts.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] == leader && !state.is_capital[h]) {
            outposts.push_back(static_cast<int16_t>(h));
        }
    }
    if (outposts.empty()) return false;

    int min_dist = hex_distance(ref_coord, state.grid->coord_of(outposts[0]));
    for (int i = 1; i < outposts.size(); ++i) {
        min_dist = std::min(min_dist, hex_distance(ref_coord, state.grid->coord_of(outposts[i])));
    }
    if (min_dist > kMaxDenyDistanceFactor * state.grid->radius()) return false;

    int best = -1;
    double best_power = 0.0;
    for (int i = 0; i < outposts.size(); ++i) {
        const int d = hex_distance(ref_coord, state.grid->coord_of(outposts[i]));
        if (d > min_dist + kAttackTolerance) continue;
        const double p = army_power(state.army_units[outposts[i]]) + kOutpostDefensePower;
        if (best < 0 || p < best_power) {  // strict: first minimum wins
            best_power = p;
            best = outposts[i];
        }
    }
    if (best < 0) return false;
    out.push_back(state.grid->coord_of(best));
    return true;
}

bool denier_move(const GameState& state, int faction, const LegalMask& legal, Move& out) {
    HexList ranked;
    mobile_hexes_by_size_desc(state, legal, ranked);
    if (ranked.empty()) return false;

    if (outpost_count(state, faction) >= kDenyMinOwnOutposts) {
        CoordList leader_target;
        if (leader_attack_target(state, faction, leader_target) &&
            move_toward(state, ranked, legal, leader_target, false, out)) {
            return true;
        }
    }
    HexCoord home{};
    if (best_expansion_target(state, faction, home)) {
        CoordList targets;
        targets.clear();
        targets.push_back(home);
        if (move_toward(state, ranked, legal, targets, /*skip_arrived=*/true, out)) return true;
    }
    CoordList attack;
    if (!best_attack_target(state, faction, attack)) return false;
    return move_toward(state, ranked, legal, attack, false, out);
}

// warlord: vanguard's objective pool plus every outpost of the current leader,
// deduped and appended after the usual targets.
void warlord_all_targets(const GameState& state, int faction, CoordList& out) {
    CoordList expansion, attack;
    ranked_expansion_targets(state, faction, expansion);
    ranked_attack_targets(state, faction, attack);
    out.clear();
    for (int i = 0; i < expansion.size(); ++i) out.push_back(expansion[i]);
    for (int i = 0; i < attack.size(); ++i) out.push_back(attack[i]);

    if (outpost_count(state, faction) >= kDenyMinOwnOutposts) {
        const int leader = current_leader(state, faction);
        if (leader >= 0) {
            for (int h = 0; h < state.num_hexes; ++h) {
                if (state.city_owner[h] != leader || state.is_capital[h]) continue;
                const HexCoord c = state.grid->coord_of(h);
                bool present = false;
                for (int i = 0; i < out.size(); ++i) {
                    if (out[i] == c) present = true;
                }
                if (!present && out.size() < out.capacity()) out.push_back(c);
            }
        }
    }
    if (out.empty()) enemy_capital_coords(state, faction, out);
}

// The rotate-through-mobile-armies loop shared by vanguard, warlord, legion and
// sentinel's fallback. `claimed`, when given (legion only), reserves a target so
// two armies do not chase the same one - and persists across the whole game.
bool rotating_move(const GameState& state, int faction, int step, const LegalMask& legal,
                   const CoordList& targets, int total_steps, bool use_best_direction,
                   CoordList* claimed, Move& out) {
    (void)faction;
    HexList mobile;
    mobile_hexes(state, legal, mobile);
    if (mobile.empty() || targets.empty()) return false;

    const int steps_remaining = total_steps - step - 1;
    const int n = mobile.size();
    for (int offset = 0; offset < n; ++offset) {
        const int origin = mobile[(step + offset) % n];
        const HexCoord& origin_coord = state.grid->coord_of(origin);

        const CoordList* pool = &targets;
        CoordList unclaimed;
        if (claimed != nullptr) {
            unclaimed.clear();
            for (int i = 0; i < targets.size(); ++i) {
                bool taken = false;
                for (int j = 0; j < claimed->size(); ++j) {
                    if ((*claimed)[j] == targets[i]) taken = true;
                }
                if (!taken) unclaimed.push_back(targets[i]);
            }
            if (!unclaimed.empty()) pool = &unclaimed;
        }

        HexCoord target{};
        int best_d = -1;
        for (int i = 0; i < pool->size(); ++i) {
            const int d = hex_distance(origin_coord, (*pool)[i]);
            if (best_d < 0 || d < best_d) {  // strict: first minimum wins
                best_d = d;
                target = (*pool)[i];
            }
        }
        if (best_d == 0) continue;  // already standing on its nearest target

        int dir;
        if (use_best_direction) {
            dir = best_direction(state, origin, legal, target, steps_remaining);
        } else {
            // legion picks the plain nearest step, with no terrain weighting.
            dir = -1;
            int best_dist = 0;
            for (int d = 0; d < NUM_DIRECTIONS; ++d) {
                if (!legal.cell[origin][d]) continue;
                const int dd =
                    hex_distance(state.grid->coord_of(state.grid->neighbour(origin, d)), target);
                if (dir < 0 || dd < best_dist) {
                    dir = d;
                    best_dist = dd;
                }
            }
        }
        if (dir < 0) continue;
        if (claimed != nullptr) {
            // Python's `claimed` is a SET, so re-claiming a target is idempotent.
            // Appending unconditionally accumulated duplicates, which eventually
            // filled the buffer and silently changed behaviour ~30 turns in.
            bool already = false;
            for (int j = 0; j < claimed->size(); ++j) {
                if ((*claimed)[j] == target) already = true;
            }
            if (!already) {
                assert(claimed->size() < claimed->capacity() && "claimed set overflow");
                claimed->push_back(target);
            }
        }
        out.hex = static_cast<int16_t>(origin);
        out.dir = static_cast<int8_t>(dir);
        return true;
    }
    return false;
}

// legion's _prune_claims: drop claims on targets that are no longer objectives.
void prune_claims(CoordList& claimed, const CoordList& targets) {
    CoordList kept;
    kept.clear();
    for (int i = 0; i < claimed.size(); ++i) {
        for (int j = 0; j < targets.size(); ++j) {
            if (claimed[i] == targets[j]) {
                kept.push_back(claimed[i]);
                break;
            }
        }
    }
    claimed = kept;
}

// sentinel: relieve any of our own outposts currently under attack; otherwise
// play vanguard.
void locked_own_cities(const GameState& state, int faction, CoordList& out) {
    out.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] == faction && state.locked(h) && !state.is_capital[h]) {
            out.push_back(state.grid->coord_of(h));
        }
    }
}

bool sentinel_move(const GameState& state, int faction, int step, const LegalMask& legal,
                   int total_steps, Move& out) {
    HexList mobile;
    mobile_hexes(state, legal, mobile);
    if (mobile.empty()) return false;

    CoordList defense;
    locked_own_cities(state, faction, defense);
    if (!defense.empty()) {
        const int steps_remaining = total_steps - step - 1;
        auto nearest = [&](int h) {
            int best = hex_distance(state.grid->coord_of(h), defense[0]);
            for (int i = 1; i < defense.size(); ++i) {
                best = std::min(best, hex_distance(state.grid->coord_of(h), defense[i]));
            }
            return best;
        };
        HexList ranked = mobile;
        std::stable_sort(ranked.items, ranked.items + ranked.count,
                         [&](int16_t a, int16_t b) { return nearest(a) < nearest(b); });

        for (int i = 0; i < ranked.size(); ++i) {
            const int origin = ranked[i];
            const HexCoord& origin_coord = state.grid->coord_of(origin);
            HexCoord target{};
            int best_d = -1;
            for (int t = 0; t < defense.size(); ++t) {
                const int d = hex_distance(origin_coord, defense[t]);
                if (best_d < 0 || d < best_d) {
                    best_d = d;
                    target = defense[t];
                }
            }
            const int dir = best_direction(state, origin, legal, target, steps_remaining);
            if (dir < 0) continue;
            out.hex = static_cast<int16_t>(origin);
            out.dir = static_cast<int8_t>(dir);
            return true;
        }
    }

    CoordList targets;
    all_targets(state, faction, targets);
    return rotating_move(state, faction, step, legal, targets, total_steps, true, nullptr, out);
}

// ============================ tactician =================================
//
// The only agent that SIMULATES. For regular-movement step 0 - the most
// consequential decision of a turn - it clones the real state, plays each
// candidate move out to the end of the turn with real dice, and keeps whichever
// scored best. Every other decision falls back to marshal.
//
// Its rollouts are why GameState is a trivially-copyable POD: cloning is a
// memcpy here, where the Python original ran a dozen np.copy calls per rollout.

inline constexpr int kMaxCandidates = 10;
inline constexpr double kRivalWeight = 0.5;
inline constexpr double kOutpostWeight = 2.0;
inline constexpr double kArmyPowerWeight = 0.05;

// Own VP dominates (it is literally the win condition), minus a penalty for
// whoever is currently the biggest rival, plus smaller terms for outposts (future
// VP) and army strength (how defensible the position is).
double evaluate(const GameState& state, int faction) {
    const int own_vp = state.victory_points[faction];
    int best_rival = 0;
    for (int f = 0; f < state.num_factions; ++f) {
        if (f != faction) best_rival = std::max(best_rival, state.victory_points[f]);
    }
    int own_outposts = 0;
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.city_owner[h] == faction && !state.is_capital[h]) ++own_outposts;
    }
    double total[NUM_UNIT_TYPES] = {0, 0, 0};
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.army_faction[h] == faction) {
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) total[t] += state.army_units[h][t];
        }
    }
    for (int i = 0; i < state.num_battles; ++i) {
        const Battle& bt = state.battles[i];
        for (int k = 0; k < bt.nslots; ++k) {
            if (bt.slots[k].faction != faction) continue;
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) total[t] += bt.slots[k].units[t];
        }
    }
    const double power =
        total[0] * kUnitPower[0] + total[1] * kUnitPower[1] + total[2] * kUnitPower[2];
    return own_vp - kRivalWeight * best_rival + kOutpostWeight * own_outposts +
           kArmyPowerWeight * power;
}

// The two policy sets a rollout runs on, shared by every tactician in a game
// exactly as make_tactician_agents shares one marshal set and one random set
// across all factions. Their RNGs are mutated by rollouts and persist, so how
// many rollouts run is itself part of the state.
struct TacticianShared {
    AgentSet mine;   // marshal, seeded like the game
    AgentSet opp;    // random, seeded seed + 999_983 - a deliberately weak, cheap
                     // opponent model that measured BETTER than a stronger one
};

// Routes a rollout's decisions: `self_faction` uses `mine`, everyone else `opp`.
struct RolloutCtx {
    const TacticianShared* shared;
    int self_faction;

    Agent* agent_for(int faction) const {
        return faction == self_faction ? shared->mine.get(faction) : shared->opp.get(faction);
    }
};

bool rc_movement(const GameState& s, int faction, int step, const LegalMask& legal, Move& out,
                 void* ctx) {
    return static_cast<RolloutCtx*>(ctx)->agent_for(faction)->decide_movement(s, faction, step,
                                                                             legal, out);
}
bool rc_cavalry(const GameState& s, int faction, int step, const LegalMask& legal, Move& out,
                void* ctx) {
    return static_cast<RolloutCtx*>(ctx)->agent_for(faction)->decide_cavalry(s, faction, step,
                                                                            legal, out);
}
int rc_target(const GameState& s, int hex_index, int faction, void* ctx) {
    return static_cast<RolloutCtx*>(ctx)->agent_for(faction)->decide_target(s, hex_index, faction);
}
void rc_rectification(const GameState& s, int hex_index, int winner, int cap, SendBack& out,
                      void* ctx) {
    static_cast<RolloutCtx*>(ctx)->agent_for(winner)->decide_rectification(s, hex_index, winner,
                                                                          cap, out);
}
Resource rc_resource(const GameState& s, int faction, int hex_index, void* ctx) {
    return static_cast<RolloutCtx*>(ctx)->agent_for(faction)->decide_resource_choice(s, faction,
                                                                                    hex_index);
}

// Plays `first_action` as our move for the CURRENT step, then the rest of the
// turn, and scores the result. `rng` is freshly seeded per candidate so every
// candidate faces identical dice - an apples-to-apples comparison.
double rollout_and_score(const GameState& state, int faction, const Move& first_action,
                         const TacticianShared& shared, Rng& rng, GameState& sim) {
    sim = state;  // memcpy - the whole point of the POD layout
    RolloutCtx ctx{&shared, faction};

    TurnDecisions td;
    td.movement = &rc_movement;
    td.cavalry = &rc_cavalry;
    td.target = &rc_target;
    td.rectification = &rc_rectification;
    td.resource_choice = &rc_resource;
    td.ctx = &ctx;

    LegalMask legal;
    {
        // Step 0. OUR action is submitted FIRST, before the opponents' - Python
        // builds `{faction: first_action}` and only then adds the rest, and that
        // submission order decides battle creation order (see MoveActions).
        MoveActions actions;
        actions.clear();
        actions.set(faction, first_action.hex, first_action.dir);
        for (int f = 0; f < sim.num_factions; ++f) {
            if (f == faction) continue;
            legal_movement_mask(sim, f, legal);
            Move mv{};
            if (shared.opp.get(f)->decide_movement(sim, f, 0, legal, mv)) {
                actions.set(f, mv.hex, mv.dir);
            }
        }
        apply_movement_step(sim, actions, rng, /*cavalry_only=*/false);
    }

    for (int step = 1; step < kMovementSteps; ++step) {
        MoveActions actions;
        actions.clear();
        for (int f = 0; f < sim.num_factions; ++f) {
            legal_movement_mask(sim, f, legal);
            Move mv{};
            if (ctx.agent_for(f)->decide_movement(sim, f, step, legal, mv)) {
                actions.set(f, mv.hex, mv.dir);
            }
        }
        apply_movement_step(sim, actions, rng, /*cavalry_only=*/false);
    }

    for (int step = 0; step < kCavalrySteps; ++step) {
        MoveActions actions;
        actions.clear();
        for (int f = 0; f < sim.num_factions; ++f) {
            legal_cavalry_mask(sim, f, legal);
            Move mv{};
            if (ctx.agent_for(f)->decide_cavalry(sim, f, step, legal, mv)) {
                actions.set(f, mv.hex, mv.dir);
            }
        }
        apply_movement_step(sim, actions, rng, /*cavalry_only=*/true);
    }

    run_battle_phase(sim, td, rng);
    apply_terrain_effects(sim);
    apply_collect_phase(sim, td.resource_choice, td.ctx);
    return evaluate(sim, faction);
}

// ============================ agent classes =============================

class RandomAgent : public Agent {
public:
    void decide_buy(const GameState&, int, const LegalBuyActions& legal,
                    ChosenBuyActions& out) override {
        random_buy(rng, legal, out);
    }
    bool decide_movement(const GameState& s, int, int, const LegalMask& legal,
                         Move& out) override {
        return random_movement(rng, s, legal, out);
    }
    bool decide_cavalry(const GameState& s, int, int, const LegalMask& legal, Move& out) override {
        return random_movement(rng, s, legal, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return random_target(rng, s, hex_index, faction);
    }
    void decide_rectification(const GameState& s, int hex_index, int winner, int cap,
                              SendBack& out) override {
        random_rectification(rng, s, hex_index, winner, cap, out);
    }
    Resource decide_resource_choice(const GameState&, int, int) override {
        // rng.choice(("iron", "fish")) is seq[_randbelow(2)] - a randbelow draw,
        // NOT random(). The two consume the generator differently.
        return rng.choice_index(2) == 0 ? kIron : kFish;
    }
    int decide_placement(const GameState& s, int, const bool legal[MAX_HEXES]) override {
        HexList candidates;
        candidates.clear();
        for (int h = 0; h < s.num_hexes; ++h) {
            if (legal[h]) candidates.push_back(static_cast<int16_t>(h));
        }
        if (candidates.empty()) return -1;
        return candidates[static_cast<int>(rng.choice_index(static_cast<size_t>(candidates.size())))];
    }
    int decide_draft(const GameState&, int, const int16_t* pool, int pool_size) override {
        if (pool_size <= 0) return -1;
        return pool[static_cast<int>(rng.choice_index(static_cast<size_t>(pool_size)))];
    }
    bool decide_swap(const GameState&, int, int, int, int) override { return rng.random() < 0.5; }
};

// Everything below shares greedy's buy and setup policies; only movement and
// battle targeting differ. Mirrors the Python layout exactly.
class GreedyBase : public Agent {
public:
    void decide_buy(const GameState& s, int faction, const LegalBuyActions& legal,
                    ChosenBuyActions& out) override {
        greedy_buy(rng, s, faction, legal, out);
    }
    void decide_rectification(const GameState& s, int hex_index, int winner, int cap,
                              SendBack& out) override {
        random_rectification(rng, s, hex_index, winner, cap, out);
    }
    Resource decide_resource_choice(const GameState& s, int faction, int) override {
        return greedy_resource_choice(s, faction);
    }
    int decide_placement(const GameState& s, int, const bool legal[MAX_HEXES]) override {
        return greedy_placement(s, legal);
    }
    int decide_draft(const GameState& s, int, const int16_t* pool, int pool_size) override {
        return greedy_draft(s, pool, pool_size);
    }
    bool decide_swap(const GameState& s, int, int leftover_hex, int, int placer_hex) override {
        return greedy_swap(s, leftover_hex, placer_hex);
    }
};

class GreedyAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int, const LegalMask& legal,
                         Move& out) override {
        return greedy_rush_move(s, faction, legal, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int, const LegalMask& legal,
                        Move& out) override {
        return greedy_rush_move(s, faction, legal, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return random_target(rng, s, hex_index, faction);
    }
};

class HeuristicAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int, const LegalMask& legal,
                         Move& out) override {
        return heuristic_move(s, faction, legal, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int, const LegalMask& legal,
                        Move& out) override {
        return heuristic_move(s, faction, legal, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class VanguardAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        return vanguard_move(s, faction, step, legal, kMovementSteps, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        return vanguard_move(s, faction, step, legal, kCavalrySteps, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class MarshalAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        return marshal_move(s, faction, step, legal, kMovementSteps, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        return marshal_move(s, faction, step, legal, kCavalrySteps, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class TurtleAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int, const LegalMask& legal,
                         Move& out) override {
        return turtle_move(s, faction, legal, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int, const LegalMask& legal,
                        Move& out) override {
        return turtle_move(s, faction, legal, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class DenierAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int, const LegalMask& legal,
                         Move& out) override {
        return denier_move(s, faction, legal, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int, const LegalMask& legal,
                        Move& out) override {
        return denier_move(s, faction, legal, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class WarlordAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        CoordList targets;
        warlord_all_targets(s, faction, targets);
        return rotating_move(s, faction, step, legal, targets, kMovementSteps, true, nullptr, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        CoordList targets;
        warlord_all_targets(s, faction, targets);
        return rotating_move(s, faction, step, legal, targets, kCavalrySteps, true, nullptr, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

// legion carries `claimed_` for the WHOLE GAME - the one piece of agent state
// that is not an RNG. This is why agents are per-game objects (PLAN.md §6.3).
class LegionAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        return legion_move(s, faction, step, legal, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        return legion_move(s, faction, step, legal, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }

private:
    bool legion_move(const GameState& s, int faction, int step, const LegalMask& legal,
                     Move& out) {
        // ORDER MATTERS: bail out on no mobile armies BEFORE pruning, exactly as
        // legion_move does. Pruning drops claims on hexes that are no longer
        // objectives, so an extra prune on a step this faction cannot move at all
        // permanently forgets claims Python still holds - which showed up ~13
        // turns later as a different target and a different move.
        HexList mobile;
        mobile_hexes(s, legal, mobile);
        if (mobile.empty()) return false;

        CoordList targets;
        all_targets(s, faction, targets);
        if (targets.empty()) return false;
        prune_claims(claimed_, targets);
        return rotating_move(s, faction, step, legal, targets, /*total_steps=*/1,
                             /*use_best_direction=*/false, &claimed_, out);
    }
    CoordList claimed_;
};

// hussar: vanguard's movement, but called WITHOUT a total_steps argument in the
// Python original - so it defaults to 1 and steps_remaining is never positive,
// meaning hussar never takes the marsh detour that vanguard does.
class HussarAgent : public GreedyBase {
public:
    void decide_buy(const GameState& s, int faction, const LegalBuyActions& legal,
                    ChosenBuyActions& out) override {
        hussar_buy(rng, s, faction, legal, out);
    }
    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        CoordList targets;
        all_targets(s, faction, targets);
        return rotating_move(s, faction, step, legal, targets, 1, true, nullptr, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        CoordList targets;
        all_targets(s, faction, targets);
        return rotating_move(s, faction, step, legal, targets, 1, true, nullptr, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class SentinelAgent : public GreedyBase {
public:
    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        return sentinel_move(s, faction, step, legal, kMovementSteps, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        return sentinel_move(s, faction, step, legal, kCavalrySteps, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }
};

class TacticianAgent : public GreedyBase {
public:
    TacticianAgent(std::shared_ptr<TacticianShared> shared, std::shared_ptr<GameState> scratch)
        : shared_(std::move(shared)), scratch_(std::move(scratch)) {}

    bool decide_movement(const GameState& s, int faction, int step, const LegalMask& legal,
                         Move& out) override {
        // Only step 0 is searched: it is the first of three chances to advance
        // the biggest piece of the turn's plan, and searching more phases was
        // measured as a regression (see tactician_agent.py's docstring).
        if (step == 0 && search_first_move(s, faction, legal, out)) return true;
        return marshal_move(s, faction, step, legal, kMovementSteps, out);
    }
    bool decide_cavalry(const GameState& s, int faction, int step, const LegalMask& legal,
                        Move& out) override {
        return marshal_move(s, faction, step, legal, kCavalrySteps, out);
    }
    int decide_target(const GameState& s, int hex_index, int faction) override {
        return heuristic_target(s, hex_index, faction);
    }

private:
    bool search_first_move(const GameState& state, int faction, const LegalMask& legal,
                           Move& out) {
        HexList mobile;
        mobile_hexes(state, legal, mobile);
        if (mobile.empty()) return false;
        CoordList targets;
        all_targets(state, faction, targets);
        if (targets.empty()) return false;
        MatchList matches;
        greedy_match(state, mobile, targets, matches);
        if (matches.empty()) return false;

        SmallVec<Move, kMaxCandidates> candidates;
        candidates.clear();
        const int n = std::min(kMaxCandidates, matches.size());
        for (int i = 0; i < n; ++i) {
            const MatchPair& m = matches[i];
            if (hex_distance(state.grid->coord_of(m.origin), m.target) == 0) continue;
            const int dir = best_direction(state, m.origin, legal, m.target, kMovementSteps - 1);
            if (dir < 0) continue;
            candidates.push_back(Move{m.origin, static_cast<int8_t>(dir)});
        }
        if (candidates.empty()) return false;
        if (candidates.size() == 1) {
            // NOTE: no rng draw in this branch. Consuming one here would desync
            // this agent's generator from the Python original.
            out = candidates[0];
            return true;
        }

        // One seed for the whole decision, reused for every candidate, so they
        // are compared against identical dice rather than different luck.
        const int64_t seed = rng.randrange(static_cast<int64_t>(1) << 31);
        bool have_best = false;
        double best_score = 0.0;
        for (int i = 0; i < candidates.size(); ++i) {
            Rng rollout_rng(seed);
            const double score =
                rollout_and_score(state, faction, candidates[i], *shared_, rollout_rng, *scratch_);
            if (!have_best || score > best_score) {  // strict: first maximum wins
                have_best = true;
                best_score = score;
                out = candidates[i];
            }
        }
        return have_best;
    }

    std::shared_ptr<TacticianShared> shared_;
    std::shared_ptr<GameState> scratch_;  // reused rollout buffer; GameState is ~65 KB
};

// --- callback adapters ------------------------------------------------------

void ad_buy(const GameState& s, int faction, const LegalBuyActions& legal, ChosenBuyActions& out,
            void* ctx) {
    static_cast<const AgentSet*>(ctx)->get(faction)->decide_buy(s, faction, legal, out);
}
bool ad_movement(const GameState& s, int faction, int step, const LegalMask& legal, Move& out,
                 void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_movement(s, faction, step, legal,
                                                                           out);
}
bool ad_cavalry(const GameState& s, int faction, int step, const LegalMask& legal, Move& out,
                void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_cavalry(s, faction, step, legal,
                                                                          out);
}
int ad_target(const GameState& s, int hex_index, int faction, void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_target(s, hex_index, faction);
}
void ad_rectification(const GameState& s, int hex_index, int winner, int cap, SendBack& out,
                      void* ctx) {
    // engine_old indexes decide_rectification by the WINNER, so the winner's own
    // generator is the one consumed.
    static_cast<const AgentSet*>(ctx)->get(winner)->decide_rectification(s, hex_index, winner, cap,
                                                                        out);
}
Resource ad_resource(const GameState& s, int faction, int hex_index, void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_resource_choice(s, faction,
                                                                                  hex_index);
}
int ad_placement(const GameState& s, int faction, const bool* legal, void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_placement(s, faction, legal);
}
int ad_draft(const GameState& s, int faction, const int16_t* pool, int pool_size, void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_draft(s, faction, pool,
                                                                        pool_size);
}
bool ad_swap(const GameState& s, int faction, int leftover, int placer, int placer_hex,
             void* ctx) {
    return static_cast<const AgentSet*>(ctx)->get(faction)->decide_swap(s, faction, leftover,
                                                                       placer, placer_hex);
}

}  // namespace

TurnDecisions make_turn_decisions(const AgentSet& agents) {
    TurnDecisions td;
    td.buy = &ad_buy;
    td.movement = &ad_movement;
    td.cavalry = &ad_cavalry;
    td.target = &ad_target;
    td.rectification = &ad_rectification;
    td.resource_choice = &ad_resource;
    td.ctx = const_cast<AgentSet*>(&agents);
    return td;
}

SetupDecisions make_setup_decisions(const AgentSet& agents) {
    SetupDecisions sd;
    sd.placement = &ad_placement;
    sd.draft = &ad_draft;
    sd.swap = &ad_swap;
    sd.ctx = const_cast<AgentSet*>(&agents);
    return sd;
}

void build_agents(AgentSet& out, AgentKind kind, int num_factions, int64_t seed) {
    out.num_factions = num_factions;

    // tactician's rollout policies are built ONCE and SHARED by every faction's
    // agent, exactly as make_tactician_agents does. Their RNGs are mutated by
    // rollouts and persist across the game, so sharing (rather than one set per
    // faction) is load-bearing, not an optimisation.
    std::shared_ptr<TacticianShared> shared;
    std::shared_ptr<GameState> scratch;
    if (kind == AgentKind::kTactician) {
        shared = std::make_shared<TacticianShared>();
        build_agents(shared->mine, AgentKind::kMarshal, num_factions, seed);
        build_agents(shared->opp, AgentKind::kRandom, num_factions, seed + 999983);
        scratch = std::make_shared<GameState>();
    }

    for (int f = 0; f < num_factions; ++f) {
        std::unique_ptr<Agent> a;
        switch (kind) {
            case AgentKind::kRandom: a = std::make_unique<RandomAgent>(); break;
            case AgentKind::kGreedy: a = std::make_unique<GreedyAgent>(); break;
            case AgentKind::kHeuristic: a = std::make_unique<HeuristicAgent>(); break;
            case AgentKind::kVanguard: a = std::make_unique<VanguardAgent>(); break;
            case AgentKind::kMarshal: a = std::make_unique<MarshalAgent>(); break;
            case AgentKind::kTurtle: a = std::make_unique<TurtleAgent>(); break;
            case AgentKind::kDenier: a = std::make_unique<DenierAgent>(); break;
            case AgentKind::kWarlord: a = std::make_unique<WarlordAgent>(); break;
            case AgentKind::kLegion: a = std::make_unique<LegionAgent>(); break;
            case AgentKind::kHussar: a = std::make_unique<HussarAgent>(); break;
            case AgentKind::kSentinel: a = std::make_unique<SentinelAgent>(); break;
            case AgentKind::kTactician:
                a = std::make_unique<TacticianAgent>(shared, scratch);
                break;
        }
        // Same per-faction seeding as every make_X_agents in agents/.
        a->rng.seed(seed * 1000003 + f);
        out.agents[f] = std::move(a);
    }
}

bool agent_kind_from_name(const char* name, AgentKind& out) {
    const std::string n(name);
    if (n == "random") { out = AgentKind::kRandom; return true; }
    if (n == "greedy") { out = AgentKind::kGreedy; return true; }
    if (n == "heuristic") { out = AgentKind::kHeuristic; return true; }
    if (n == "vanguard") { out = AgentKind::kVanguard; return true; }
    if (n == "marshal") { out = AgentKind::kMarshal; return true; }
    if (n == "turtle") { out = AgentKind::kTurtle; return true; }
    if (n == "denier") { out = AgentKind::kDenier; return true; }
    if (n == "warlord") { out = AgentKind::kWarlord; return true; }
    if (n == "legion") { out = AgentKind::kLegion; return true; }
    if (n == "hussar") { out = AgentKind::kHussar; return true; }
    if (n == "sentinel") { out = AgentKind::kSentinel; return true; }
    if (n == "tactician") { out = AgentKind::kTactician; return true; }
    return false;
}

}  // namespace oo
