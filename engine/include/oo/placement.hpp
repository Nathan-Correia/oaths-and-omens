// Capital setup - the port of engine_old/placement.py.
//
// Two phases, each with its own independently random faction order:
//   1. PLACEMENT. Each faction in turn places one colourless (unowned) city.
//      Nobody owns anything yet; city_placer just records who placed what.
//   2. DRAFT. Each faction in turn claims an already-placed city as its capital,
//      never the one it placed itself. The last faction is a special case: if the
//      only city left is its own placement it simply takes it, otherwise it may
//      force a SWAP - taking what the placer already drafted and bumping the
//      placer onto the leftover.
//
// As in engine_old, whatever a callback returns is re-validated against the legal
// set actually computed; an invalid answer falls back to a uniform random legal
// choice from `rng` rather than raising.

#pragma once

#include "oo/actions.hpp"  // SmallVec
#include "oo/rng.hpp"
#include "oo/state.hpp"

#include <vector>

namespace oo {

inline constexpr int kCapitalMinDist = 3;      // between any two placed cities
inline constexpr int kEdgeBanMinFactions = 5;  // 5-7 players: no capitals on the edge ring
inline constexpr int kEdgeBanMaxFactions = 7;

// Hexes a colourless city could legally be placed on right now: passable, not
// already placed on, at least kCapitalMinDist from every placed city, and - only
// for 5-7 factions - not on the board's edge ring.
//
// Relaxes in tiers if the strict mask would be empty: the edge ban drops first,
// then the distance rule. "Not already placed" and "passable" never relax.
// Whichever tier is non-empty is what an agent is offered, so it is never shown a
// hex that is not really legal.
void legal_placement_mask(const GameState& state, bool out[MAX_HEXES]);

using PlacementFn = int (*)(const GameState& state, int faction, const bool legal[MAX_HEXES],
                            void* ctx);
using DraftFn = int (*)(const GameState& state, int faction, const int16_t* pool, int pool_size,
                        void* ctx);
// True forces the swap; false keeps the leftover as-is. Only ever called for the
// single last-drafting faction, and only when the leftover is not its own placement.
using SwapFn = bool (*)(const GameState& state, int faction, int leftover_hex, int placer_faction,
                        int placer_hex, void* ctx);

struct SetupDecisions {
    PlacementFn placement = nullptr;
    DraftFn draft = nullptr;
    SwapFn swap = nullptr;
    void* ctx = nullptr;
};

// One placement/draft step, for city_placement_log.json (§1.3).
struct PlacementLogEntry {
    enum Kind : int8_t { kPlace = 0, kDraft = 1, kDraftAuto = 2, kKeep = 3, kSwap = 4 };
    Kind kind;
    int8_t faction;
    int8_t q, r, s;
    // Only meaningful for kSwap: who gets bumped, and onto which hex.
    int8_t placer_faction = NO_FACTION;
    int8_t placer_q = 0, placer_r = 0, placer_s = 0;
};

// Mutates `state`. Called once, before any turn is played.
void run_city_setup(GameState& state, const SetupDecisions& decisions, Rng& rng,
                    std::vector<PlacementLogEntry>* log = nullptr);

}  // namespace oo
