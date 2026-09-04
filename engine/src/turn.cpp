#include "oo/turn.hpp"

#include "oo/buy.hpp"
#include "oo/movement.hpp"
#include "oo/terrain.hpp"

namespace oo {

void run_battle_phase(GameState& state, const TurnDecisions& decisions, Rng& rng) {
    // One shared tally for the whole turn - see the header note on battle order.
    int32_t infantry_counts[MAX_FACTIONS];
    for (int f = 0; f < MAX_FACTIONS; ++f) {
        infantry_counts[f] = f < state.num_factions ? count_units_in_play(state, f, kInfantry) : 0;
    }

    // Snapshot of the pending list: resolving a battle mutates battle_order, and
    // reverting/rectifying can add or remove entries.
    int16_t pending[MAX_ACTIVE_BATTLES];
    const int n_pending = state.num_battles;
    for (int i = 0; i < n_pending; ++i) pending[i] = state.battles[i].hex;

    for (int i = 0; i < n_pending; ++i) {
        const int hex_index = pending[i];
        if (!state.locked(hex_index)) continue;

        resolve_full_battle(state, hex_index, decisions.target, decisions.ctx, rng,
                            infantry_counts);

        const int winner = get_winner(state, hex_index);
        if (winner < 0) {
            // Everyone wiped out simultaneously: the tile is left empty with no
            // one victorious.
            state.army_faction[hex_index] = NO_FACTION;
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) state.army_units[hex_index][t] = 0;
            state.erase_battle(hex_index);
            continue;
        }

        const int owner = state.city_owner[hex_index];
        int cap = MAX_STACK_SIZE;
        if (owner != NO_FACTION && owner != winner) {
            if (state.is_capital[hex_index]) {
                cap = 0;  // uncapturable: the winner is evicted whole
            } else {
                state.city_owner[hex_index] = NO_FACTION;  // outpost destroyed, not captured
                state.victory_points[winner] += kOutpostDestroyVp;
            }
        }

        SendBack send_back;
        send_back.clear();
        decisions.rectification(state, hex_index, winner, cap, send_back, decisions.ctx);
        rectify_overflow(state, hex_index, winner, send_back, cap);
    }
}

void run_turn(GameState& state, const TurnDecisions& decisions, Rng& rng) {
    // --- Buy ---------------------------------------------------------------
    {
        static_assert(MAX_FACTIONS <= 10, "chosen[] is sized for the rulebook's 10-player max");
        ChosenBuyActions chosen[MAX_FACTIONS];
        LegalBuyActions legal;
        for (int f = 0; f < state.num_factions; ++f) {
            chosen[f].clear();
            get_legal_buy_actions(state, f, legal);
            decisions.buy(state, f, legal, chosen[f], decisions.ctx);
        }
        apply_buy_phase(state, chosen);
    }

    // --- Movement ----------------------------------------------------------
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
    }

    // --- Cavalry -----------------------------------------------------------
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
    }

    run_battle_phase(state, decisions, rng);
    apply_terrain_effects(state);
    apply_collect_phase(state, decisions.resource_choice, decisions.ctx);

    state.turn_number += 1;
}

int get_game_winner(const GameState& state) {
    int top = state.victory_points[0];
    for (int f = 1; f < state.num_factions; ++f) {
        if (state.victory_points[f] > top) top = state.victory_points[f];
    }
    if (top < kVpToWin) return -1;

    int winner = -1;
    int best_settle = -1;
    int contenders = 0;
    for (int f = 0; f < state.num_factions; ++f) {
        if (state.victory_points[f] != top) continue;
        ++contenders;
        // Tie-break: whoever settled their capital LATER wins.
        if (winner < 0 || state.capital_settle_order[f] > best_settle) {
            winner = f;
            best_settle = state.capital_settle_order[f];
        }
    }
    (void)contenders;
    return winner;
}

bool check_game_end(const GameState& state, int max_turns) {
    if (get_game_winner(state) >= 0) return true;
    return max_turns >= 0 && state.turn_number >= max_turns;
}

}  // namespace oo
