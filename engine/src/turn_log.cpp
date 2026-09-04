// run_turn_and_log - the same turn as run_turn, plus a full replay record.
//
// Deliberately a separate entry point rather than a flag on run_turn: logging is
// run.py's dominant cost, and the tournament / self-play path must not pay a
// branch for it.
//
// The battle phase is duplicated here rather than calling run_battle_phase,
// because the log needs each battle's contributions captured BEFORE it resolves
// and its outcome after. The rules themselves are not duplicated - every step
// below calls the same shared functions run_battle_phase does.

#include "oo/log.hpp"

#include "oo/battle.hpp"
#include "oo/buy.hpp"
#include "oo/collect.hpp"
#include "oo/movement.hpp"
#include "oo/terrain.hpp"
#include "oo/turn.hpp"

namespace oo {

void snapshot_sparse(const GameState& state, std::vector<HexSnapshot>& out) {
    out.clear();
    for (int h = 0; h < state.num_hexes; ++h) {
        HexSnapshot e{};
        const HexCoord& c = state.grid->coord_of(h);
        e.q = c.q;
        e.r = c.r;
        e.s = c.s;

        if (state.city_owner[h] != NO_FACTION) {
            e.has_city = true;
            e.city_faction = state.city_owner[h];
            e.is_capital = state.is_capital[h];
            e.upgrade = state.outpost_upgrade[h];
        }

        if (state.locked(h)) {
            FactionTotals totals;
            faction_totals(state, h, totals);
            if (totals.count > 0) {
                e.has_battle = true;
                for (int i = 0; i < totals.count; ++i) {
                    ContributionStart c2{};
                    c2.faction = totals.faction[i];
                    c2.origin_hex = NO_ORIGIN;  // unused in a checkpoint entry
                    for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                        c2.units[t] = static_cast<int16_t>(totals.units[i][t]);
                    }
                    e.contributions.push_back(c2);
                }
            }
        } else if (state.army_faction[h] != NO_FACTION) {
            e.has_troops = true;
            e.troop_faction = state.army_faction[h];
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) e.troops[t] = state.army_units[h][t];
            e.frozen = state.frozen[h];
        }

        // sparse_hexes: keep only hexes with something on them.
        if (e.has_city || e.has_troops || e.has_battle) out.push_back(std::move(e));
    }
}

void snapshot_player_stats(const GameState& state, std::vector<PlayerStats>& out) {
    out.clear();
    for (int f = 0; f < state.num_factions; ++f) {
        PlayerStats p{};
        p.gold = state.gold[f];
        for (int r = 0; r < NUM_RESOURCES; ++r) p.resources[r] = state.resources[f][r];
        p.kill_xp = state.kill_xp[f];
        p.victory_points = state.victory_points[f];
        p.alive = state.alive[f];
        out.push_back(p);
    }
}

namespace {

void capture(const GameState& state, TurnRecord& out) {
    out.checkpoints.emplace_back();
    snapshot_sparse(state, out.checkpoints.back());
    out.player_stats.emplace_back();
    snapshot_player_stats(state, out.player_stats.back());
}

// The battle phase, with each battle's before/after captured. Mirrors
// run_battle_phase step for step.
void run_battle_phase_logged(GameState& state, const TurnDecisions& decisions, Rng& rng,
                             std::vector<BattleEvent>& events) {
    int32_t infantry_counts[MAX_FACTIONS];
    for (int f = 0; f < MAX_FACTIONS; ++f) {
        infantry_counts[f] = f < state.num_factions ? count_units_in_play(state, f, kInfantry) : 0;
    }

    int16_t pending[MAX_ACTIVE_BATTLES];
    const int n_pending = state.num_battles;
    for (int i = 0; i < n_pending; ++i) pending[i] = state.battles[i].hex;

    for (int i = 0; i < n_pending; ++i) {
        const int hex_index = pending[i];
        if (!state.locked(hex_index)) continue;

        BattleEvent ev{};
        const HexCoord& c = state.grid->coord_of(hex_index);
        ev.q = c.q;
        ev.r = c.r;
        ev.s = c.s;
        const Battle* b = state.battle_at(hex_index);
        for (int k = 0; b != nullptr && k < b->nslots; ++k) {
            if (b->slots[k].faction == NO_FACTION) continue;
            ContributionStart cs{};
            cs.faction = b->slots[k].faction;
            cs.origin_hex = b->slots[k].origin;
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) cs.units[t] = b->slots[k].units[t];
            ev.contributions_start.push_back(cs);
        }

        resolve_full_battle(state, hex_index, decisions.target, decisions.ctx, rng, infantry_counts,
                            &ev.log);

        const int winner = get_winner(state, hex_index);
        ev.winner = winner;
        if (winner < 0) {
            state.army_faction[hex_index] = NO_FACTION;
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.army_units[hex_index][t] = 0;
            state.erase_battle(hex_index);
        } else {
            const int owner = state.city_owner[hex_index];
            int cap = MAX_STACK_SIZE;
            if (owner != NO_FACTION && owner != winner) {
                if (state.is_capital[hex_index]) {
                    cap = 0;
                } else {
                    state.city_owner[hex_index] = NO_FACTION;
                    state.victory_points[winner] += kOutpostDestroyVp;
                }
            }
            SendBack send_back;
            send_back.clear();
            decisions.rectification(state, hex_index, winner, cap, send_back, decisions.ctx);
            for (int e = 0; e < send_back.size(); ++e) {
                SendBackLog sb{};
                sb.origin_hex = send_back[e].origin_hex;
                for (int t = 0; t < NUM_UNIT_TYPES; ++t) sb.units[t] = send_back[e].units[t];
                ev.rectification.push_back(sb);
            }
            rectify_overflow(state, hex_index, winner, send_back, cap);
        }

        events.push_back(std::move(ev));
    }
}

}  // namespace

void run_turn_and_log(GameState& state, const TurnDecisions& decisions, Rng& rng, TurnRecord& out) {
    out.turn_number = state.turn_number;
    out.checkpoints.clear();
    out.player_stats.clear();
    out.battle_events.clear();

    // "Start": before this turn's Buy phase. Because Collect runs at the END of a
    // turn, this already reflects the PREVIOUS turn's income - there is no
    // separate "before income" moment.
    capture(state, out);

    {
        ChosenBuyActions chosen[MAX_FACTIONS];
        LegalBuyActions legal;
        for (int f = 0; f < state.num_factions; ++f) {
            chosen[f].clear();
            get_legal_buy_actions(state, f, legal);
            decisions.buy(state, f, legal, chosen[f], decisions.ctx);
        }
        apply_buy_phase(state, chosen);
    }
    capture(state, out);

    for (int step = 0; step < kMovementSteps; ++step) {
        MoveActions actions;
        actions.clear();
        LegalMask legal;
        for (int f = 0; f < state.num_factions; ++f) {
            legal_movement_mask(state, f, legal);
            Move chosen{};
            if (decisions.movement(state, f, step, legal, chosen, decisions.ctx)) {
                actions.set(f, chosen.hex, chosen.dir);
            }
        }
        apply_movement_step(state, actions, rng, /*cavalry_only=*/false);
        capture(state, out);
    }

    for (int step = 0; step < kCavalrySteps; ++step) {
        MoveActions actions;
        actions.clear();
        LegalMask legal;
        for (int f = 0; f < state.num_factions; ++f) {
            legal_cavalry_mask(state, f, legal);
            Move chosen{};
            if (decisions.cavalry(state, f, step, legal, chosen, decisions.ctx)) {
                actions.set(f, chosen.hex, chosen.dir);
            }
        }
        apply_movement_step(state, actions, rng, /*cavalry_only=*/true);
        capture(state, out);
    }

    run_battle_phase_logged(state, decisions, rng, out.battle_events);
    apply_terrain_effects(state);
    apply_collect_phase(state, decisions.resource_choice, decisions.ctx);
    capture(state, out);

    state.turn_number += 1;
}

}  // namespace oo
