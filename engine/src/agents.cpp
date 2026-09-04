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

#include <algorithm>
#include <cstring>
#include <string>
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
    for (int k = 0; k < state.battle_nslots[hex_index]; ++k) {
        if (state.battle_faction[hex_index][k] == winner) {
            origins.push_back(state.battle_origin[hex_index][k]);
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

void greedy_buy(Rng& rng, const GameState& state, int faction, const LegalBuyActions& legal,
                ChosenBuyActions& out) {
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
        // Counted ONCE before the loop and never updated, so every conversion
        // this turn picks the same type. Ported as-is.
        const int cav_count = count_units_in_play(state, faction, kCavalry);
        const int arc_count = count_units_in_play(state, faction, kArchers);
        const int want = (cav_count <= arc_count) ? kCavalry : kArchers;

        SmallVec<int16_t, kMaxLegalBuy> matching;
        matching.clear();
        for (int i = 0; i < convert_actions.size(); ++i) {
            if (legal[convert_actions[i]].unit_type == want) matching.push_back(convert_actions[i]);
        }
        const SmallVec<int16_t, kMaxLegalBuy>& pool =
            matching.empty() ? convert_actions : matching;

        for (int i = 0; i < num_conversions; ++i) {
            if (out.size() >= out.capacity()) break;
            out.push_back(
                legal[pool[static_cast<int>(rng.choice_index(static_cast<size_t>(pool.size())))]]);
        }
    }
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
        if (state.army_faction[h] == faction && !state.locked[h]) {
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
    for (int f = 0; f < num_factions; ++f) {
        std::unique_ptr<Agent> a;
        switch (kind) {
            case AgentKind::kRandom: a = std::make_unique<RandomAgent>(); break;
            case AgentKind::kGreedy: a = std::make_unique<GreedyAgent>(); break;
            case AgentKind::kHeuristic: a = std::make_unique<HeuristicAgent>(); break;
            case AgentKind::kVanguard: a = std::make_unique<VanguardAgent>(); break;
            case AgentKind::kMarshal: a = std::make_unique<MarshalAgent>(); break;
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
    return false;
}

}  // namespace oo
