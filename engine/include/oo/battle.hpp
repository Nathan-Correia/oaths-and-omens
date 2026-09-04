// Battle resolution - the port of engine_old/battle.py.
//
// THE ORDER OF EVERY randint(1, 20) IS LOAD-BEARING. Two independently written
// engines fed identically seeded RNGs only agree if they consume draws in the same
// sequence, and the parity tests key on exactly that. The order is:
//
//   1. structure defence shots  (capital 2, outpost 1; 11-20 = 1 kill each)
//   2. the real Archers ability (each MOVED archer once; 11-20 = 1 kill)
//   3. per round, until one faction is left:
//        a. targeting conflicts resolved (no dice)
//        b. one roll per attacker, in resolved-target insertion order
//        c. all kills applied simultaneously
//        d. one dismount roll per cavalry that died, per faction, in
//           first-death order (14-20 = the cavalry dismounts into an infantry)
//
// Where engine_old iterates `battle.contributions` (a list, in append order), this
// iterates slots 0..K-1 - the same order by construction, since movement fills
// slots from 0 upward and nothing ever frees an individual slot.

#pragma once

#include "oo/actions.hpp"
#include "oo/rng.hpp"
#include "oo/state.hpp"

namespace oo {

inline constexpr int kMaxRoundsSafetyCap = 50;  // infinite-loop guard, never hit in practice
inline constexpr int kCapitalDefenseShots = 2;
inline constexpr int kOutpostDefenseShots = 1;

// Per-faction unit totals in a battle, in FIRST-APPEARANCE (slot) order. That
// order drives targeting, roll order and tie-breaks, so it is part of the
// contract, not an implementation detail.
struct FactionTotals {
    int8_t faction[MAX_FACTIONS];
    int32_t units[MAX_FACTIONS][NUM_UNIT_TYPES];
    int count = 0;

    int total_for(int i) const { return units[i][0] + units[i][1] + units[i][2]; }
    int index_of(int f) const {
        for (int i = 0; i < count; ++i) {
            if (faction[i] == f) return i;
        }
        return -1;
    }
};

void faction_totals(const GameState& state, int hex_index, FactionTotals& out);

// Like faction_totals, but summing only slots flagged battle_moved - the units
// that actually moved to join this fight. Gates the Archers ability to the
// attacking side.
void faction_moved_totals(const GameState& state, int hex_index, FactionTotals& out);

// Valid targets for `faction` this round: any other faction still alive here.
void get_legal_target_actions(const GameState& state, int hex_index, int faction,
                              SmallVec<int8_t, MAX_FACTIONS>& out);

bool is_battle_over(const GameState& state, int hex_index);

// The single surviving faction, or -1 if none or several remain.
int get_winner(const GameState& state, int hex_index);

// The agent decision point: which faction is `faction` attacking this round?
// Return -1 to abstain.
using TargetFn = int (*)(const GameState& state, int hex_index, int faction, void* ctx);

// Runs the whole battle to completion, in place, crediting kill-XP as units die.
// `infantry_counts` is a running per-faction tally the CALLER maintains and shares
// across every battle in the turn - that sharing is why battle order matters.
void resolve_full_battle(GameState& state, int hex_index, TargetFn target_fn, void* ctx, Rng& rng,
                         int32_t infantry_counts[MAX_FACTIONS]);

// After a battle resolves, if the winning stack exceeds `cap`, the winner sends
// the excess back to their own contributing origin hexes. Units whose origin is
// invalid, that `send_back` does not account for, or that would push the origin
// itself past MAX_STACK_SIZE are trimmed off (infantry -> cavalry -> archers)
// rather than left stacked over the cap.
//
// `cap` is normally MAX_STACK_SIZE, but is 0 when the winner just took a battle on
// a foreign CAPITAL - capitals are uncapturable, so that winner is evicted whole.
// City ownership is not touched here; that is turn.cpp's job.
void rectify_overflow(GameState& state, int hex_index, int winner_faction, const SendBack& send_back,
                      int cap = MAX_STACK_SIZE);

}  // namespace oo
