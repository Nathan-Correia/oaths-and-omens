#include "oo/battle.hpp"

#include "oo/log.hpp"

#include <cassert>

namespace oo {

namespace {

// engine_old iterates DEATH_PRIORITY = (0, 1, 2): infantry, then cavalry, then
// archers. Same order as the enum, spelled out because it is a rule.
constexpr int kDeathPriority[NUM_UNIT_TYPES] = {kInfantry, kCavalry, kArchers};

void totals_impl(const GameState& state, int hex_index, bool moved_only, FactionTotals& out) {
    out.count = 0;
    const int nslots = state.battle_nslots[hex_index];
    for (int k = 0; k < nslots; ++k) {
        const int8_t f = state.battle_faction[hex_index][k];
        if (f == NO_FACTION) continue;
        int i = out.index_of(f);
        if (i < 0) {
            // First appearance - this is what fixes the order.
            i = out.count++;
            out.faction[i] = f;
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) out.units[i][t] = 0;
        }
        if (moved_only && !state.battle_moved[hex_index][k]) continue;
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
            out.units[i][t] += state.battle_units[hex_index][k][t];
        }
    }
}

int kills_for_roll(int roll, int attacker_total_units) {
    if (roll <= 5) return 0;
    if (roll <= 15) return 1;
    return attacker_total_units == 1 ? 1 : 2;
}

// Removes up to num_kills units from target_faction's presence in this battle -
// infantry, then cavalry, then archers, taking from the earliest contribution
// slots first. Credits the killer's kill-XP as it goes (engine_old builds a death
// log and credits from it afterwards; crediting inline is equivalent, and the
// structured log lands with the replay logging in M6c).
void apply_kills_to_faction(GameState& state, int hex_index, int target_faction, int num_kills,
                            int killer_faction, std::vector<DeathEntry>* deaths) {
    int remaining = num_kills;
    const int nslots = state.battle_nslots[hex_index];
    for (int p = 0; p < NUM_UNIT_TYPES && remaining > 0; ++p) {
        const int ut = kDeathPriority[p];
        for (int k = 0; k < nslots && remaining > 0; ++k) {
            if (state.battle_faction[hex_index][k] != target_faction) continue;
            const int available = state.battle_units[hex_index][k][ut];
            const int take = available < remaining ? available : remaining;
            if (take > 0) {
                state.battle_units[hex_index][k][ut] = static_cast<int16_t>(available - take);
                remaining -= take;
                state.kill_xp[killer_faction] += take;
                if (deaths != nullptr) {
                    deaths->push_back(DeathEntry{static_cast<int8_t>(target_faction),
                                                 static_cast<int8_t>(ut),
                                                 static_cast<int16_t>(take),
                                                 static_cast<int8_t>(killer_faction)});
                }
            }
        }
    }
}

int cavalry_of(const GameState& state, int hex_index, int faction) {
    int total = 0;
    const int nslots = state.battle_nslots[hex_index];
    for (int k = 0; k < nslots; ++k) {
        if (state.battle_faction[hex_index][k] == faction) {
            total += state.battle_units[hex_index][k][kCavalry];
        }
    }
    return total;
}

// One dismount roll per cavalry that just died. 14-20 turns it into a live
// infantry joining the same battle immediately - which can keep an otherwise
// defeated army in the fight - subject to the concurrent infantry cap.
void roll_dismounts(GameState& state, int hex_index, int faction, int died_count, Rng& rng,
                    int32_t infantry_counts[MAX_FACTIONS],
                    std::vector<DismountEntry>* dismounts) {
    for (int i = 0; i < died_count; ++i) {
        if (rng.randint(1, 20) < 14) {
            if (dismounts != nullptr) {
                dismounts->push_back(DismountEntry{static_cast<int8_t>(faction), false, false});
            }
            continue;
        }
        if (infantry_counts[faction] >= kSpawnCaps[kInfantry]) {
            if (dismounts != nullptr) {
                dismounts->push_back(DismountEntry{static_cast<int8_t>(faction), false, true});
            }
            continue;
        }
        const int nslots = state.battle_nslots[hex_index];
        for (int k = 0; k < nslots; ++k) {
            if (state.battle_faction[hex_index][k] == faction) {
                state.battle_units[hex_index][k][kInfantry] += 1;
                break;
            }
        }
        infantry_counts[faction] += 1;
        if (dismounts != nullptr) {
            dismounts->push_back(DismountEntry{static_cast<int8_t>(faction), true, false});
        }
    }
}

// Applies kills, then rolls dismounts for whatever cavalry that killed. Used by
// the two pre-round phases; the main round applies all kills first and only then
// rolls dismounts, so it does this in two steps of its own.
void apply_kills_and_dismounts(GameState& state, int hex_index, int target_faction, int num_kills,
                               int killer_faction, Rng& rng,
                               int32_t infantry_counts[MAX_FACTIONS],
                               std::vector<DeathEntry>* deaths,
                               std::vector<DismountEntry>* dismounts) {
    const int before = cavalry_of(state, hex_index, target_faction);
    apply_kills_to_faction(state, hex_index, target_faction, num_kills, killer_faction, deaths);
    const int after = cavalry_of(state, hex_index, target_faction);
    roll_dismounts(state, hex_index, target_faction, before - after, rng, infantry_counts,
                   dismounts);
}

// A capital or outpost gets free defensive shots against whoever is attacking its
// tile, even with no units there to defend with. Same 11-20 math as the Archers
// ability but a deliberately separate mechanic, so the two can be tuned apart.
void apply_structure_defense_shots(GameState& state, int hex_index, Rng& rng,
                                   int32_t infantry_counts[MAX_FACTIONS],
                                   std::vector<DeathEntry>* deaths,
                                   std::vector<DismountEntry>* dismounts) {
    const int owner = state.city_owner[hex_index];
    if (owner == NO_FACTION) return;

    FactionTotals totals;
    faction_totals(state, hex_index, totals);

    // Largest rival army. Ties go to the earliest in first-appearance order,
    // matching Python's max() over an insertion-ordered dict.
    int target = -1, best = -1;
    for (int i = 0; i < totals.count; ++i) {
        if (totals.faction[i] == owner) continue;
        const int t = totals.total_for(i);
        if (t <= 0) continue;
        if (t > best) {
            best = t;
            target = totals.faction[i];
        }
    }
    if (target < 0) return;

    const int shots = state.is_capital[hex_index] ? kCapitalDefenseShots : kOutpostDefenseShots;
    int kills = 0;
    for (int i = 0; i < shots; ++i) {
        if (rng.randint(1, 20) >= 11) ++kills;
    }
    if (kills > 0) {
        apply_kills_and_dismounts(state, hex_index, target, kills, owner, rng, infantry_counts,
                                  deaths, dismounts);
    }
}

// Runs once, before round 1. Only archers that actually MOVED to join this battle
// fire - a stationary defender's archers do not roll at all - but targeting still
// weighs each side's full strength, moved or not.
void apply_archer_abilities(GameState& state, int hex_index, Rng& rng,
                            int32_t infantry_counts[MAX_FACTIONS],
                            std::vector<DeathEntry>* deaths,
                            std::vector<DismountEntry>* dismounts) {
    FactionTotals totals, moved;
    faction_totals(state, hex_index, totals);
    faction_moved_totals(state, hex_index, moved);

    // NOTE: totals is captured ONCE, before any archer fires, and is deliberately
    // not refreshed as kills land. engine_old computes it before the loop, so a
    // later archer targets based on pre-phase strengths. Recomputing would be
    // more intuitive and would diverge.
    for (int i = 0; i < totals.count; ++i) {
        const int faction = totals.faction[i];
        const int mi = moved.index_of(faction);
        const int archers = mi >= 0 ? moved.units[mi][kArchers] : 0;
        if (archers <= 0) continue;
        if (totals.total_for(i) <= 0) continue;  // not alive

        int target = -1, best = -1;
        for (int j = 0; j < totals.count; ++j) {
            if (j == i) continue;
            const int t = totals.total_for(j);
            if (t <= 0) continue;
            if (t > best) {
                best = t;
                target = totals.faction[j];
            }
        }
        if (target < 0) continue;

        int kills = 0;
        for (int a = 0; a < archers; ++a) {
            if (rng.randint(1, 20) >= 11) ++kills;
        }
        if (kills > 0) {
            apply_kills_and_dismounts(state, hex_index, target, kills, faction, rng,
                                      infantry_counts, deaths, dismounts);
        }
    }
}

// One full round: conflicts, rolls, simultaneous kills, then dismounts.
void resolve_round(GameState& state, int hex_index, const int target_choices[MAX_FACTIONS],
                   const FactionTotals& order, Rng& rng, int32_t infantry_counts[MAX_FACTIONS],
                   RoundLog* log) {
    FactionTotals totals;
    faction_totals(state, hex_index, totals);

    // --- resolve targeting conflicts ---------------------------------------
    // No army may be attacked by more than one player in a round: if several pick
    // the same target, only the largest army follows through. Ties go to the
    // LOWEST faction id (Python maximizes on (units, -faction)).
    // Resolved pairs are built in first-appearance order of each TARGET, which is
    // then the roll order below.
    if (log != nullptr) {
        // target_choices_submitted: every living faction, in first-appearance
        // order, with its choice (-1 for an abstain, which serializes as null).
        for (int i = 0; i < order.count; ++i) {
            if (order.total_for(i) <= 0) continue;
            log->choice_faction.push_back(order.faction[i]);
            log->choice_target.push_back(static_cast<int8_t>(target_choices[order.faction[i]]));
        }
    }

    int attacker_of[MAX_FACTIONS];
    int target_of[MAX_FACTIONS];
    int n_resolved = 0;
    int seen_target[MAX_FACTIONS];
    int n_seen = 0;

    for (int i = 0; i < order.count; ++i) {
        const int attacker = order.faction[i];
        const int target = target_choices[attacker];
        if (target < 0) continue;

        bool already = false;
        for (int s = 0; s < n_seen; ++s) {
            if (seen_target[s] == target) already = true;
        }
        if (already) continue;
        seen_target[n_seen++] = target;

        // Everyone who chose this target; keep the strongest.
        int winner = -1, winner_units = -1;
        for (int j = 0; j < order.count; ++j) {
            const int f = order.faction[j];
            if (target_choices[f] != target) continue;
            const int fi = totals.index_of(f);
            const int units = fi >= 0 ? totals.total_for(fi) : 0;
            if (units > winner_units || (units == winner_units && f < winner)) {
                winner_units = units;
                winner = f;
            }
        }
        attacker_of[n_resolved] = winner;
        target_of[n_resolved] = target;
        ++n_resolved;
        if (log != nullptr) {
            log->resolved_attacker.push_back(static_cast<int8_t>(winner));
            log->resolved_target.push_back(static_cast<int8_t>(target));
        }
    }

    // --- rolls -------------------------------------------------------------
    int pending_target[MAX_FACTIONS];
    int pending_kills[MAX_FACTIONS];
    int pending_killer[MAX_FACTIONS];
    int n_pending = 0;

    for (int i = 0; i < n_resolved; ++i) {
        const int attacker = attacker_of[i];
        const int ai = totals.index_of(attacker);
        const int attacker_units = ai >= 0 ? totals.total_for(ai) : 0;
        if (attacker_units <= 0) continue;
        const int roll = static_cast<int>(rng.randint(1, 20));
        const int kills = kills_for_roll(roll, attacker_units);
        if (log != nullptr) {
            log->roll_faction.push_back(static_cast<int8_t>(attacker));
            log->roll_value.push_back(static_cast<int16_t>(roll));
            log->kills_dealt.push_back(static_cast<int16_t>(kills));
        }
        if (kills > 0) {
            pending_target[n_pending] = target_of[i];
            pending_kills[n_pending] = kills;
            pending_killer[n_pending] = attacker;
            ++n_pending;
        }
    }

    // --- simultaneous application ------------------------------------------
    int cav_died_faction[MAX_FACTIONS];
    int cav_died_count[MAX_FACTIONS];
    int n_died = 0;

    for (int i = 0; i < n_pending; ++i) {
        const int target = pending_target[i];
        const int before = cavalry_of(state, hex_index, target);
        apply_kills_to_faction(state, hex_index, target, pending_kills[i], pending_killer[i],
                               log != nullptr ? &log->deaths : nullptr);
        const int after = cavalry_of(state, hex_index, target);
        const int died = before - after;
        if (died <= 0) continue;
        int slot = -1;
        for (int d = 0; d < n_died; ++d) {
            if (cav_died_faction[d] == target) slot = d;
        }
        if (slot < 0) {
            slot = n_died++;
            cav_died_faction[slot] = target;
            cav_died_count[slot] = 0;
        }
        cav_died_count[slot] += died;
    }

    // --- dismounts, in first-death order -----------------------------------
    for (int d = 0; d < n_died; ++d) {
        roll_dismounts(state, hex_index, cav_died_faction[d], cav_died_count[d], rng,
                       infantry_counts, log != nullptr ? &log->dismounts : nullptr);
    }

    state.battle_round[hex_index] += 1;
}

}  // namespace

void faction_totals(const GameState& state, int hex_index, FactionTotals& out) {
    totals_impl(state, hex_index, false, out);
}

void faction_moved_totals(const GameState& state, int hex_index, FactionTotals& out) {
    totals_impl(state, hex_index, true, out);
}

void get_legal_target_actions(const GameState& state, int hex_index, int faction,
                              SmallVec<int8_t, MAX_FACTIONS>& out) {
    out.clear();
    FactionTotals totals;
    faction_totals(state, hex_index, totals);
    for (int i = 0; i < totals.count; ++i) {
        if (totals.faction[i] == faction) continue;
        if (totals.total_for(i) > 0) out.push_back(totals.faction[i]);
    }
}

bool is_battle_over(const GameState& state, int hex_index) {
    FactionTotals totals;
    faction_totals(state, hex_index, totals);
    int alive = 0;
    for (int i = 0; i < totals.count; ++i) {
        if (totals.total_for(i) > 0) ++alive;
    }
    return alive <= 1;
}

int get_winner(const GameState& state, int hex_index) {
    FactionTotals totals;
    faction_totals(state, hex_index, totals);
    int alive = 0, winner = -1;
    for (int i = 0; i < totals.count; ++i) {
        if (totals.total_for(i) > 0) {
            ++alive;
            winner = totals.faction[i];
        }
    }
    return alive == 1 ? winner : -1;
}

void resolve_full_battle(GameState& state, int hex_index, TargetFn target_fn, void* ctx, Rng& rng,
                         int32_t infantry_counts[MAX_FACTIONS], BattleLog* log) {
    apply_structure_defense_shots(state, hex_index, rng, infantry_counts,
                                  log ? &log->structure_phase : nullptr,
                                  log ? &log->structure_dismounts : nullptr);
    apply_archer_abilities(state, hex_index, rng, infantry_counts,
                           log ? &log->archer_phase : nullptr,
                           log ? &log->archer_dismounts : nullptr);

    int rounds_run = 0;
    while (!is_battle_over(state, hex_index) && rounds_run < kMaxRoundsSafetyCap) {
        FactionTotals order;
        faction_totals(state, hex_index, order);

        int target_choices[MAX_FACTIONS];
        for (int f = 0; f < MAX_FACTIONS; ++f) target_choices[f] = -1;
        for (int i = 0; i < order.count; ++i) {
            if (order.total_for(i) <= 0) continue;
            target_choices[order.faction[i]] = target_fn(state, hex_index, order.faction[i], ctx);
        }

        if (log != nullptr) log->rounds.emplace_back();
        resolve_round(state, hex_index, target_choices, order, rng, infantry_counts,
                      log != nullptr ? &log->rounds.back() : nullptr);
        ++rounds_run;
    }
}

void rectify_overflow(GameState& state, int hex_index, int winner_faction, const SendBack& send_back,
                      int cap) {
    FactionTotals totals;
    faction_totals(state, hex_index, totals);
    const int wi = totals.index_of(winner_faction);
    int32_t winning[NUM_UNIT_TYPES] = {0, 0, 0};
    if (wi >= 0) {
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) winning[t] = totals.units[wi][t];
    }

    for (int e = 0; e < send_back.size(); ++e) {
        const SendBackEntry& entry = send_back[e];
        const int origin = entry.origin_hex;
        const bool valid_origin = origin >= 0 && origin < state.num_hexes;
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
            const int take = entry.units[t] < winning[t] ? entry.units[t] : winning[t];
            winning[t] -= take;
            if (!valid_origin || take <= 0) continue;
            if (state.army_faction[origin] == NO_FACTION) {
                state.army_faction[origin] = static_cast<int8_t>(winner_faction);
            }
            if (state.army_faction[origin] == winner_faction) {
                // The 6-unit limit is strict outside battle: a returning unit that
                // does not fit at its origin simply does not make it back.
                const int room = MAX_STACK_SIZE - state.units_at(origin);
                const int deposit = take < room ? take : (room > 0 ? room : 0);
                if (deposit > 0) state.army_units[origin][t] += static_cast<int16_t>(deposit);
            }
            // else: the origin is held by someone else, or has no room - those
            // units are simply lost.
        }
    }

    int total_remaining = winning[0] + winning[1] + winning[2];
    if (total_remaining > cap) {
        int excess = total_remaining - cap;
        for (int p = 0; p < NUM_UNIT_TYPES && excess > 0; ++p) {
            const int ut = kDeathPriority[p];
            const int take = winning[ut] < excess ? winning[ut] : excess;
            winning[ut] -= take;
            excess -= take;
        }
    }

    if (winning[0] + winning[1] + winning[2] > 0) {
        state.army_faction[hex_index] = static_cast<int8_t>(winner_faction);
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
            state.army_units[hex_index][t] = static_cast<int16_t>(winning[t]);
        }
    } else {
        state.army_faction[hex_index] = NO_FACTION;
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.army_units[hex_index][t] = 0;
    }
    state.frozen[hex_index] = false;
    state.locked[hex_index] = false;

    for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
        state.battle_faction[hex_index][k] = NO_FACTION;
        state.battle_origin[hex_index][k] = NO_ORIGIN;
        state.battle_moved[hex_index][k] = false;
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.battle_units[hex_index][k][t] = 0;
    }
    state.battle_nslots[hex_index] = 0;
    state.battle_round[hex_index] = 0;

    // Order-preserving compacting erase - Python's list.remove keeps order, and
    // battle order affects outcomes near the shared dismount cap.
    int w = 0;
    for (int i = 0; i < state.num_battles; ++i) {
        if (state.battle_order[i] == hex_index) continue;
        state.battle_order[w++] = state.battle_order[i];
    }
    state.num_battles = static_cast<int16_t>(w);
}

}  // namespace oo
